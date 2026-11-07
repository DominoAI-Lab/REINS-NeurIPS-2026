import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from torch.autograd import forward_ad
from tqdm import tqdm


def aggregate_to_subject_level(distances, labels, subject_ids, agg='median'):
    from collections import defaultdict
    subj_to_dists = defaultdict(list)
    subj_to_labels = defaultdict(list)
    for i, sid in enumerate(subject_ids):
        subj_to_dists[sid].append(distances[i])
        subj_to_labels[sid].append(labels[i])

    subj_ids_sorted = sorted(subj_to_dists.keys())
    if agg == 'median':
        agg_fn = np.median
    elif agg == 'mean':
        agg_fn = np.mean
    elif agg == 'max':
        agg_fn = np.max
    else:
        raise ValueError(f'unknown agg: {agg}')

    subj_distances = np.array([agg_fn(subj_to_dists[s]) for s in subj_ids_sorted])
    subj_labels = np.array([np.bincount(subj_to_labels[s]).argmax() for s in subj_ids_sorted])
    return subj_distances, subj_labels


def build_space(z, poses_np, Mq_np, hc_train_mask, n_pca,
                drop_pca_dims=1, ridge_alpha=1.0):
    D_orig = z.shape[1]

    pca_proj = PCA(n_components=n_pca).fit(z[hc_train_mask])
    z_proj = pca_proj.transform(z)

    if drop_pca_dims > 0:
        pca_drop = PCA(n_components=n_pca).fit(z_proj[hc_train_mask])
        z_pca = pca_drop.transform(z_proj)
        z_pca[:, :drop_pca_dims] = 0.0
        z_use = pca_drop.inverse_transform(z_pca)
    else:
        z_use = z_proj

    reg = Ridge(alpha=ridge_alpha, fit_intercept=True)
    reg.fit(z_use[hc_train_mask], poses_np[hc_train_mask])
    J_ridge = reg.coef_

    import constants
    M_bar = Mq_np[hc_train_mask].mean(0) + 1e-6 * np.eye(75)
    J_q = J_ridge[np.asarray(constants.smpl2q), :]
    G_const = J_q.T @ M_bar @ J_q + 1e-6 * np.eye(n_pca)

    assert pca_proj.components_.shape == (n_pca, D_orig)
    return z_use, G_const, M_bar, pca_proj


def pairwise_G(z_sub, G):
    Gz = z_sub @ G
    s = (Gz * z_sub).sum(1)
    return np.sqrt(np.maximum(s[:, None] + s[None, :] - 2.0 * (z_sub @ Gz.T), 0.0))


def pairwise_pointwise(z_sub, G_sub, block=64):
    n = z_sub.shape[0]
    A = np.empty((n, n))
    for s in range(0, n, block):
        g = G_sub[s:s + block]
        zb = z_sub[s:s + block]
        Zg = np.einsum('jd,bde->bje', z_sub, g)
        t1 = (Zg * z_sub[None]).sum(-1)
        t2 = np.einsum('bje,be->bj', Zg, zb)
        t3 = np.einsum('bd,bde,be->b', zb, g, zb)
        A[s:s + block] = t1 - 2.0 * t2 + t3[:, None]
    np.fill_diagonal(A, 0.0)
    return np.sqrt(np.maximum(0.5 * (A + A.T), 0.0))


def knn_from_dist(d_full, k):
    n = d_full.shape[0]
    d = d_full.copy()
    np.fill_diagonal(d, np.inf)
    k = min(k, n - 1)
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    for i in range(n):
        idx[i] = idx[i][np.argsort(d[i, idx[i]])]
    return idx


def M_at_pose(smpl_model, q_hat, device):
    import constants
    from assets.utils import batch_roteulerSMPL
    B = q_hat.shape[0]
    rotmat, rot_ind = batch_roteulerSMPL(q_hat[:, 3:])
    out = smpl_model.forward(betas=torch.zeros(B, 10, device=device), rotmat=rotmat)
    mass, _, inertia = smpl_model.compute_massinertia(out.v_shaped, out.joint_smpl_shaped)
    _, Mq, _, _, _ = smpl_model.compute_Euler_Lagrange(
        mass, inertia, out.vertices, out.joints_smpl, rot_ind,
        q_dot=torch.zeros(max(B - 1, 0), 75, device=device), dt=constants.dt, T=None)
    return Mq


def compute_G_pointwise(z_use, pca_proj, model, smpl_model, device,
                        T=16, batch=64, poses_ref=None, Mq_ref=None,
                        check_only=False, z_raw=None):
    import constants
    from assets.utils import rot6d_to_rotmat, rotmat2eulerSMPL

    n_pca = z_use.shape[1]
    comp_t = torch.tensor(pca_proj.components_, dtype=torch.float32, device=device)
    mean_t = torch.tensor(pca_proj.mean_, dtype=torch.float32, device=device)
    perm = torch.tensor(constants.smpl2q, dtype=torch.long, device=device)

    override = {}

    def _hook(module, inp, out):
        if 'z' in override:
            return override['z']

    h = model.transformer_encoder.register_forward_hook(_hook)
    model.eval()
    model.mode = 'test'

    def _decode(zb):
        B = zb.shape[0]
        z_full = zb @ comp_t + mean_t
        override['z'] = z_full.unsqueeze(0).expand(T, B, -1)
        dummy_src = torch.zeros(T, B, 279, device=device)
        dummy_shape = torch.zeros(B, 10, device=device)
        pred_q, _ = model(src=dummy_src, tgt=dummy_src, gt_shape=dummy_shape,
                          gt_gender_id=None, smplh_m=None, smplh_f=None,
                          smpl=smpl_model)
        pq = pred_q.reshape(B * T, 147)
        rotmat = rot6d_to_rotmat(pq[:, 3:].reshape(-1, 6)).reshape(B * T, 24, 3, 3)
        euler = rotmat2eulerSMPL(rotmat).reshape(B, T, 72)
        return torch.cat([pred_q[:, :, :3], euler], dim=-1)

    def _decode_mean(zb):
        return _decode(zb).mean(dim=1)

    N = z_use.shape[0]
    G_all = np.zeros((N, n_pca, n_pca), dtype=np.float64)
    z_t = torch.tensor(z_use, dtype=torch.float32, device=device)

    if poses_ref is not None:
        with torch.no_grad():
            q_hat = _decode_mean(z_t[:64]).cpu().numpy()
        ref = poses_ref[:64]
        print('  decode(z_use) vs stored poses : mean|diff| = %.4e '
              '(pose scale %.4e)' % (np.abs(q_hat - ref).mean(),
                                     np.abs(ref).mean()))
    if poses_ref is not None and z_raw is not None:
        zr = torch.tensor(z_raw[:64], dtype=torch.float32, device=device)
        with torch.no_grad():
            override['z'] = zr.unsqueeze(0).expand(T, zr.shape[0], -1)
            B = zr.shape[0]
            pred_q, _ = model(src=torch.zeros(T, B, 279, device=device),
                              tgt=torch.zeros(T, B, 279, device=device),
                              gt_shape=torch.zeros(B, 10, device=device),
                              gt_gender_id=None, smplh_m=None, smplh_f=None,
                              smpl=smpl_model)
            pq = pred_q.reshape(B * T, 147)
            rotmat = rot6d_to_rotmat(pq[:, 3:].reshape(-1, 6)).reshape(B * T, 24, 3, 3)
            euler = rotmat2eulerSMPL(rotmat).reshape(B, T, 72)
            q_raw = torch.cat([pred_q[:, :, :3], euler], dim=-1).mean(dim=1).cpu().numpy()
        print('  decode(raw z) vs stored poses : mean|diff| = %.4e' %
              np.abs(q_raw - poses_ref[:64]).mean())
    if Mq_ref is not None:
        p = np.asarray(constants.q2smpl)
        ref = Mq_ref[:64]
        with torch.no_grad():
            Mh = M_at_pose(smpl_model, _decode_mean(z_t[:64]), device).cpu().numpy()
        print('  M(q_hat) vs stored Mq : mean|diff| q-order %.4e, euler-order %.4e '
              '(M scale %.4e)' % (np.abs(Mh - ref).mean(),
                                  np.abs(Mh[:, p][:, :, p] - ref).mean(), np.abs(ref).mean()))
        if poses_ref is not None:
            with torch.no_grad():
                Ms = M_at_pose(smpl_model, torch.tensor(poses_ref[:64], dtype=torch.float32,
                                                        device=device), device).cpu().numpy()
            print('  M(stored pose) vs stored Mq : mean|diff| q-order %.4e, euler-order %.4e'
                  % (np.abs(Ms - ref).mean(), np.abs(Ms[:, p][:, :, p] - ref).mean()))
    if check_only:
        h.remove()
        return None

    with torch.no_grad():
        for s in tqdm(range(0, N, batch), desc='J(z)'):
            zb = z_t[s:s + batch]
            B = zb.shape[0]
            cols = []
            for i in range(n_pca):
                tan = torch.zeros_like(zb)
                tan[:, i] = 1.0
                with forward_ad.dual_level():
                    zd = forward_ad.make_dual(zb, tan)
                    qd = _decode(zd)
                    q_hat, col = forward_ad.unpack_dual(qd)
                if col is None:
                    raise RuntimeError('forward-AD returned no tangent; the '
                                       'decoder graph is detached somewhere')
                cols.append(col)
            J = torch.stack(cols, dim=-1)
            M_b = M_at_pose(smpl_model, q_hat.reshape(B * T, 75), device)
            M_b = M_b.reshape(B, T, 75, 75)
            J_q = J[:, :, perm, :]
            G = torch.einsum('btji,btjk,btkl->bil', J_q, M_b, J_q)
            G_all[s:s + batch] = G.double().cpu().numpy()

    h.remove()
    return G_all


def edge_weights(knn_idx, z_use, G_all):
    n, k = knn_idx.shape
    w = np.zeros((n, k))
    for i in range(n):
        nb = knn_idx[i]
        dz = z_use[i][None, :] - z_use[nb]
        t_i = np.einsum('kd,de,ke->k', dz, G_all[i], dz)
        t_j = np.einsum('kd,kde,ke->k', dz, G_all[nb], dz)
        w[i] = np.sqrt(np.maximum(0.5 * (t_i + t_j), 0.0))
    return w


def score(z_use, G_const, G_all, hc_train_mask, subject_ids, eval_mask,
          knn_k=20, mode='per_subject', knn_metric='pointwise',
          dump=(), dump_store=None):
    N = z_use.shape[0]
    src = np.where(hc_train_mask)[0]
    out = np.full(N, np.nan)

    def cand(z_sub, G_sub):
        return (pairwise_pointwise(z_sub, G_sub) if knn_metric == 'pointwise'
                else pairwise_G(z_sub, G_const))

    if mode == 'transductive':
        idx = knn_from_dist(cand(z_use, G_all), knn_k)
        w = edge_weights(idx, z_use, G_all)
        k = idx.shape[1]
        g = csr_matrix((w.ravel(), (np.repeat(np.arange(N), k), idx.ravel())),
                       shape=(N, N))
        g = g.maximum(g.T)
        d = dijkstra(g, indices=src, directed=False, min_only=True)
        if np.isinf(d).any():
            raise RuntimeError('unreachable node in the global graph')
        return d

    if mode == 'nograph':
        for i in np.where(eval_mask)[0]:
            dz = z_use[i][None, :] - z_use[src]
            t_i = np.einsum('kd,de,ke->k', dz, G_all[i], dz)
            t_s = np.einsum('kd,kde,ke->k', dz, G_all[src], dz)
            out[i] = np.sqrt(np.maximum(0.5 * (t_i + t_s), 0.0)).min()
        return out

    n_forced = 0
    for s in np.unique(subject_ids[eval_mask]):
        tgt = np.where((subject_ids == s) & eval_mask)[0]
        sub = np.concatenate([src, tgt])
        n_src, n = len(src), len(sub)

        d_full = cand(z_use[sub], G_all[sub])
        idx = knn_from_dist(d_full, knn_k)
        w = edge_weights(idx, z_use[sub], G_all[sub])

        k = idx.shape[1]
        rows = np.repeat(np.arange(n), k)
        cols = idx.ravel()
        vals = w.ravel()

        tgt_local = np.arange(n_src, n)
        no_src = ~(idx[tgt_local] < n_src).any(1)
        n_forced += int(no_src.sum())
        if no_src.any():
            er = tgt_local[no_src]
            ec = d_full[er][:, :n_src].argmin(1)
            dz = z_use[sub][er] - z_use[sub][ec]
            t_i = np.einsum('kd,kde,ke->k', dz, G_all[sub][er], dz)
            t_j = np.einsum('kd,kde,ke->k', dz, G_all[sub][ec], dz)
            ew = np.sqrt(np.maximum(0.5 * (t_i + t_j), 0.0))
            rows = np.concatenate([rows, er])
            cols = np.concatenate([cols, ec])
            vals = np.concatenate([vals, ew])

        g = csr_matrix((vals, (rows, cols)), shape=(n, n))
        g = g.maximum(g.T)
        if str(s) in dump:
            d, pred, srcs = dijkstra(g, indices=np.arange(n_src), directed=False,
                                     min_only=True, return_predecessors=True)
            dump_store[str(s)] = dict(
                z=z_use[sub], gidx=sub, is_ref=np.arange(n) < n_src,
                edge_i=rows, edge_j=cols, edge_w=vals,
                edge_forced=np.arange(len(vals)) >= n * k,
                dist=d, pred=pred, src=srcs,
                g=G_all[sub].astype(np.float32), gconst=G_const,
                knn_metric=knn_metric)
        else:
            d = dijkstra(g, indices=np.arange(n_src), directed=False, min_only=True)
        if np.isinf(d).any():
            raise RuntimeError('unreachable node after forced attachment')
        out[tgt] = d[n_src:]

    print(f'[per_subject] {n_forced} windows needed a forced edge')
    return out


TASKS = [('HC/PD', lambda lb: (lb == 0) | (lb == 1)),
         ('HC/all', lambda lb: np.ones(len(lb), dtype=bool))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', nargs='+', required=True,
                    help='label=<output dir of train.py>, e.g. '
                         'reins=out/reins')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--mode', default='per_subject',
                    choices=['transductive', 'per_subject', 'nograph'],
                    help='per_subject (default) is the paper protocol: one graph per '
                         'test subject over the reference windows and its own windows')
    ap.add_argument('--knn_k', type=int, default=20)
    ap.add_argument('--knn_metric', default='pointwise', choices=['pointwise', 'const'])
    ap.add_argument('--ridge_alpha', type=float, default=1.0)
    ap.add_argument('--var_threshold', type=float, default=0.95)
    ap.add_argument('--model_checkpoint', required=True)
    ap.add_argument('--physpt_dir', default='.',
                    help='directory holding PhysPT\'s models/, assets/, config.py, constants.py')
    ap.add_argument('--geo_batch', type=int, default=64)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--boot_seed', type=int, default=0)
    ap.add_argument('--recompute_G', action='store_true',
                    help='ignore the cached G(z) and recompute')
    ap.add_argument('--check', action='store_true',
                    help='print alignment diagnostics and exit before Dijkstra')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--dump', nargs='*', default=[],
                    help='per_subject only: export the graph of these test '
                         'subjects to <out_root>/<lab>_k<k>_dump_<subject>.npz')
    args = ap.parse_args()
    if args.dump and args.mode != 'per_subject':
        ap.error('--dump needs --mode per_subject')

    sys.path.insert(0, args.physpt_dir)
    from models.PhysPT import PhysPT
    from models.smpl_phys import SMPL as PhysSMPL

    os.makedirs(args.out_root, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for item in args.conditions:
        lab, base = item.split('=', 1)
        kdir = os.path.join(base, f'k{args.k}')
        d = torch.load(os.path.join(kdir, 'carepd_latents.pt'), map_location='cpu')
        si = torch.load(os.path.join(kdir, 'split_info.pt'), map_location='cpu')

        labels = d['labels'].numpy().astype(int)
        Mq_np = d['Mq'].numpy()
        poses_np = d['poses'].numpy()
        sid = d['subject_ids']
        sid = np.asarray(sid.numpy() if hasattr(sid, 'numpy') else sid)
        latents = d['latents']
        z = latents.mean(dim=1).numpy()
        T = latents.shape[1]

        train_hc = set(si['train_hc_subjects'])
        ev = set(si['eval_hc_subjects']) | set(si['eval_pd_subjects']) \
            | set(si['eval_other_subjects'])
        hc_train_mask = np.array([s in train_hc for s in sid]) & (labels == 0)
        eval_mask = np.array([s in ev for s in sid])

        N, D = z.shape
        pf = PCA(n_components=min(200, int(hc_train_mask.sum()), D)).fit(z[hc_train_mask])
        n_pca = int(np.searchsorted(np.cumsum(pf.explained_variance_ratio_),
                                    args.var_threshold)) + 1
        print(f'[{lab}] N={N} D={D} T={T} n_pca={n_pca} '
              f'train-HC={int(hc_train_mask.sum())} eval={int(eval_mask.sum())}')

        z_use, G_const, M_bar, pca_proj = build_space(
            z, poses_np, Mq_np, hc_train_mask, n_pca,
            drop_pca_dims=1, ridge_alpha=args.ridge_alpha)

        model = PhysPT(device=device, seqlen=T, mode='test', f_dim=279,
                       d_model=D, nhead=8, d_hid=1024, nlayers=6,
                       dropout=0.1).to(device)
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        smpl = PhysSMPL(device=device)

        cache = os.path.join(args.out_root, f'{lab}_k{args.k}_p{n_pca}_GMqT.npy')
        if os.path.exists(cache) and not args.recompute_G:
            G_all = np.load(cache)
            print(f'  loaded cached G(z) from {cache}')
        elif args.check:
            compute_G_pointwise(z_use, pca_proj, model, smpl, device, T=T,
                                batch=args.geo_batch, poses_ref=poses_np,
                                Mq_ref=Mq_np, check_only=True, z_raw=z)
            print('  ||G_const||_F      = %.4e' % np.linalg.norm(G_const))
            return
        else:
            G_all = compute_G_pointwise(z_use, pca_proj, model, smpl,
                                        device, T=T, batch=args.geo_batch,
                                        poses_ref=poses_np, Mq_ref=Mq_np)
            np.save(cache, G_all)
            print(f'  cached G(z) to {cache}')

        if args.check:
            print('  ||G_const||_F      = %.4e' % np.linalg.norm(G_const))
            print('  mean ||G(z_i)||_F  = %.4e' % np.mean(
                [np.linalg.norm(G_all[i]) for i in range(0, N, 200)]))
            print('  min eig G(z_i)     = %.4e' % min(
                np.linalg.eigvalsh(G_all[i]).min() for i in range(0, N, 200)))
            return

        import time
        _t0 = time.time()
        dumps = {}
        dist = score(z_use, G_const, G_all, hc_train_mask, sid, eval_mask,
                     knn_k=args.knn_k, mode=args.mode, knn_metric=args.knn_metric,
                     dump=set(args.dump), dump_store=dumps)
        print(f'  scoring stage: {time.time()-_t0:.1f} s '
              f'(excludes the one-off G(z) pass)')
        for s_id, dd in dumps.items():
            path = os.path.join(args.out_root, f'{lab}_k{args.k}_dump_{s_id}.npz')
            np.savez(path, subject=s_id, knn_k=args.knn_k,
                     evr=pca_proj.explained_variance_ratio_,
                     labels=labels[dd['gidx']], subj=sid[dd['gidx']], **dd)
            print(f'  dumped {s_id}: {path}')
        missing = set(args.dump) - set(dumps)
        if missing:
            raise RuntimeError(f'--dump subjects not in eval set: {sorted(missing)}')

        s_sub, lb = aggregate_to_subject_level(dist[eval_mask], labels[eval_mask],
                                               sid[eval_mask], agg='median')
        results[lab] = {}
        for task, mask_fn in TASKS:
            keep = mask_fn(lb)
            y = (lb[keep] != 0).astype(int)
            sc = s_sub[keep]
            hc, pos = np.where(y == 0)[0], np.where(y == 1)[0]
            rng = np.random.RandomState(args.boot_seed)
            bs = []
            for _ in range(args.n_boot):
                dr = np.concatenate([rng.choice(hc, len(hc), True),
                                     rng.choice(pos, len(pos), True)])
                bs.append(np.nan if len(np.unique(y[dr])) < 2
                          else roc_auc_score(y[dr], sc[dr]))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            pt = roc_auc_score(y, sc)
            results[lab][task] = dict(auroc=float(pt), lo=float(lo), hi=float(hi))
            print(f'  {task:7s} {pt:.4f} [{lo:.4f}, {hi:.4f}]  '
                  f'(n_HC={len(hc)}, n_pos={len(pos)})')

        tag = f'{args.mode}_{args.knn_metric}'
        np.savez(os.path.join(args.out_root, f'{lab}_k{args.k}_scores_{tag}.npz'),
                 distances=dist, labels=labels, subject_ids=sid, eval_mask=eval_mask,
                 n_pca=n_pca, knn_k=args.knn_k, knn_metric=args.knn_metric,
                 z_use=z_use.astype(np.float32), hc_train_mask=hc_train_mask)

    with open(os.path.join(args.out_root,
                           f'summary_k{args.k}_{tag}.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
