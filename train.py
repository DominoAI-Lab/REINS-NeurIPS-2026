import os
import sys
import torch
import numpy as np

from core import (
    SEQLEN, D_MODEL, NHEAD, D_HID, NLAYERS, DROPOUT, F_DIM, DT,
    CarePDDataset,
    finetune_carepd,
    extract_latents,
    eval_all_metrics,
)
from models.PhysPT import PhysPT
from models.smpl_phys import SMPL


def build_split(pkl_path, data_dir, n_train_hc,
                                 hc_eval_ratio=0.30, seed=42):
    import pickle, random

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    hc_subjects, pd_subjects, other_subjects = [], [], []
    label_map = {}
    for s in data:
        label = list(data[s].values())[0]['other']
        if label == 'healthy-control':
            hc_subjects.append(s)
        elif label == 'parkinson':
            pd_subjects.append(s)
        else:
            other_subjects.append(s)

    for s in hc_subjects:    label_map[s] = 0
    for s in pd_subjects:    label_map[s] = 1
    for s in other_subjects: label_map[s] = 2

    rng = random.Random(seed)
    hc_shuffled = sorted(hc_subjects)
    rng.shuffle(hc_shuffled)

    n_hc_eval = max(1, int(len(hc_shuffled) * hc_eval_ratio))
    eval_hc_subjects = set(hc_shuffled[:n_hc_eval])
    pool_hc = hc_shuffled[n_hc_eval:]

    if n_train_hc > len(pool_hc):
        raise ValueError(
            f'n_train_hc={n_train_hc} exceeds pool size {len(pool_hc)} '
            f'(HC total={len(hc_shuffled)}, eval={n_hc_eval})'
        )

    train_hc_subjects = set(pool_hc[:n_train_hc])
    eval_pd_subjects = set(pd_subjects)
    eval_other_subjects = set(other_subjects)

    all_npz = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    train_files = sorted(f for f in all_npz if f.split('__')[0] in train_hc_subjects)
    eval_files = sorted(
        f for f in all_npz
        if (f.split('__')[0] in eval_hc_subjects
            or f.split('__')[0] in eval_pd_subjects
            or f.split('__')[0] in eval_other_subjects)
    )

    print(f'HC-only split (K={n_train_hc}, hc_eval_ratio={hc_eval_ratio}, seed={seed}):')
    print(f'  Total HC pool: {len(pool_hc)} subjects (eval HC fixed: {n_hc_eval})')
    print(f'  Train HC: {n_train_hc} subjects, {len(train_files)} files')
    print(f'  Eval HC (held-out): {len(eval_hc_subjects)} subjects')
    print(f'  Eval PD (held-out): {len(eval_pd_subjects)} subjects')
    print(f'  Eval other (held-out): {len(eval_other_subjects)} subjects')
    print(f'  Total eval files: {len(eval_files)}')

    split_info = {
        'train_hc_subjects':   train_hc_subjects,
        'eval_hc_subjects':    eval_hc_subjects,
        'eval_pd_subjects':    eval_pd_subjects,
        'eval_other_subjects': eval_other_subjects,
        'pool_hc_subjects':    set(pool_hc),
    }
    return train_files, eval_files, label_map, split_info


def _suffix_k(path, K):
    base, ext = os.path.splitext(path)
    return f'{base}_k{K}{ext}'


def run_one_k(args, K, device, smpl, dt, base_output_dir, base_checkpoint_out):
    output_dir_k = os.path.join(base_output_dir, f'k{K}')
    checkpoint_out_k = _suffix_k(base_checkpoint_out, K)
    os.makedirs(output_dir_k, exist_ok=True)

    print(f'\n{"#"*72}')
    print(f'# K = {K}')
    print(f'# output_dir   = {output_dir_k}')
    print(f'# checkpoint   = {checkpoint_out_k}')
    print(f'{"#"*72}')

    train_files, eval_files, label_map, split_info = build_split(
        pkl_path=args.pkl_path,
        data_dir=args.data_dir,
        n_train_hc=K,
        hc_eval_ratio=args.hc_eval_ratio,
        seed=args.split_seed,
    )

    train_ds = CarePDDataset(args.data_dir, train_files, label_map)
    held_out_hc_files = [
        f for f in eval_files if f.split('__')[0] in split_info['eval_hc_subjects']
    ]
    val_ds  = CarePDDataset(args.data_dir, held_out_hc_files, label_map)
    test_ds = val_ds

    print(f'\nDataset sizes (windows):')
    print(f'  train (K={K} HC): {len(train_ds)}')
    print(f'  val/test (held-out HC): {len(val_ds)}')

    model = PhysPT(
        device=device, seqlen=SEQLEN, mode='train', f_dim=F_DIM,
        d_model=D_MODEL, nhead=NHEAD, d_hid=D_HID, nlayers=NLAYERS, dropout=DROPOUT,
    ).to(device)

    if args.eval_only:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f'\nLoaded checkpoint from {args.checkpoint}')
        print(f'\nMetrics on held-out HC:')
        eval_metrics = eval_all_metrics(model, smpl, val_ds, device, dt=dt, verbose=True)
        return {'K': K, 'output_dir': output_dir_k,
                'final_test_metrics': eval_metrics,
                'oos_results': None, 'status': 'eval_only'}

    if not args.extract_only:
        finetune_carepd(
            model, smpl, train_ds, val_ds, test_ds, device,
            ft_epochs=args.ft_epochs,
            pretrained_path=args.checkpoint,
            checkpoint_out=checkpoint_out_k,
            from_scratch=args.from_scratch,
            dt=dt,
            lambda_force=args.lambda_force,
            lambda_contact=args.lambda_contact,
            lambda_euler=args.lambda_euler,
            lambda_rec=args.lambda_rec,
            lambda_act=args.lambda_act,
            gamma_act=args.gamma_act,
            lambda_iso=args.lambda_iso,
            patience=args.patience,
            min_epochs=args.min_epochs,
            min_delta=args.min_delta,
        )
    else:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f'Loaded checkpoint from {args.checkpoint}')

    print(f'\n--- Final metrics on best checkpoint (held-out HC) ---')
    final_test_metrics = eval_all_metrics(model, smpl, val_ds, device, dt=dt, verbose=True)

    all_files = train_files + eval_files
    full_ds = CarePDDataset(args.data_dir, all_files, label_map)
    print(f'\nExtracting latents on {len(full_ds)} windows '
          f'({len(all_files)} files = {len(train_files)} train + {len(eval_files)} eval)')
    (latents, labels, Mq, poses, subj_ids,
     ke_pred, ke_gt, el_res, accel, qdots) = extract_latents(model, full_ds, device, dt=dt)

    latents_path = os.path.join(output_dir_k, 'carepd_latents.pt')
    torch.save({
        'latents':      latents,
        'labels':       labels,
        'Mq':           Mq,
        'poses':        poses,
        'subject_ids':  subj_ids,
        'phys_ke_pred': ke_pred,
        'phys_ke_gt':   ke_gt,
        'phys_el_res':  el_res,
        'phys_accel':   accel,
        'qdots':        qdots,
    }, latents_path)
    print(f'Latents saved to {latents_path}: {latents.shape}')

    split_path = os.path.join(output_dir_k, 'split_info.pt')
    torch.save({
        'train_hc_subjects':   list(split_info['train_hc_subjects']),
        'eval_hc_subjects':    list(split_info['eval_hc_subjects']),
        'eval_pd_subjects':    list(split_info['eval_pd_subjects']),
        'eval_other_subjects': list(split_info['eval_other_subjects']),
        'pool_hc_subjects':    list(split_info['pool_hc_subjects']),
        'n_train_hc':          K,
        'hc_eval_ratio':       args.hc_eval_ratio,
        'split_seed':          args.split_seed,
    }, split_path)
    print(f'Split info saved to {split_path}')

    oos_results = None
    if args.auto_analyze:
        print(f'\n{"="*70}')
        print(f'Auto-analyze: running OOS subject AUROC for K={K}')
        print(f'{"="*70}')
        try:
            from analyze_latents import run_oos_subject_auroc
        except ImportError as e:
            print(f'ERROR: cannot import analyze_latents.run_oos_subject_auroc: {e}')
            print(f'Skipping auto_analyze for K={K}.')
            return {'K': K, 'output_dir': output_dir_k,
                    'final_test_metrics': final_test_metrics,
                    'oos_results': None,
                    'status': 'ok_no_analyze'}

        methods = None
        if args.auto_analyze_methods is not None:
            methods = [m.strip() for m in args.auto_analyze_methods.split(',') if m.strip()]

        oos_results = run_oos_subject_auroc(
            latents_path=latents_path,
            train_hc_subjects=split_info['train_hc_subjects'],
            eval_hc_subjects=split_info['eval_hc_subjects'],
            eval_pd_subjects=split_info['eval_pd_subjects'],
            eval_other_subjects=split_info['eval_other_subjects'],
            output_dir=output_dir_k,
            methods=methods,
            knn_k=args.auto_analyze_knn_k,
            ridge_alpha=args.auto_analyze_ridge_alpha,
            run_sensitivity=args.auto_analyze_sensitivity,
            verbose=True,
        )

        print(f'\n{"="*80}')
        print(f'K={K} OOS Summary (subject-level / window-level AUROC)')
        print(f'{"="*80}')
        print(f'{"Method":40s} {"subj HC/PD":>11s} {"subj all":>11s} '
              f'{"win HC/PD":>11s} {"win all":>11s}')
        print('-' * 80)
        sorted_methods = sorted(oos_results.items(),
                                  key=lambda kv: -kv[1]['auroc_pd_subj'])
        for m, v in sorted_methods:
            print(f'{m:40s} {v["auroc_pd_subj"]:11.4f} {v["auroc_all_subj"]:11.4f} '
                  f'{v["auroc_pd_win"]:11.4f} {v["auroc_all_win"]:11.4f}')

    return {'K': K, 'output_dir': output_dir_k,
            'final_test_metrics': final_test_metrics,
            'oos_results': oos_results,
            'status': 'ok'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pkl_path', type=str,
                        default='/data/CARE-PD/assets/datasets/DNE.pkl')
    parser.add_argument('--checkpoint', type=str,
                        default='assets/checkpoint/PhysPT.pt',
                        help='AMASS pretrained checkpoint (FIXED across K-sweep)')
    parser.add_argument('--checkpoint_out', type=str,
                        default='checkpoints/reins.pt')
    parser.add_argument('--output_dir', type=str, default='outputs/reins')
    parser.add_argument('--ft_epochs', type=int, default=100,
                        help='fine-tune epochs hard ceiling (default 100); '
                             'with --patience early-stopping kicks in well before this')
    parser.add_argument('--from_scratch', action='store_true',
                        help='do not load pretrained checkpoint (sanity check only)')

    parser.add_argument('--patience', type=int, default=10,
                        help='epochs without val MPJPE improvement before early stop '
                             '(default 10 = 2x empirical max gap)')
    parser.add_argument('--min_epochs', type=int, default=20,
                        help='minimum epochs before early-stopping can trigger '
                             '(default 20 protects small-data K from premature stop)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='minimum MPJPE improvement (mm) to count as improved '
                             '(default 0 = any decrease counts)')

    parser.add_argument('--lambda_force',   type=float, default=1.0)
    parser.add_argument('--lambda_contact', type=float, default=1.0)
    parser.add_argument('--lambda_euler',   type=float, default=1.0)
    parser.add_argument('--lambda_rec',     type=float, default=0.0)
    parser.add_argument('--lambda_act',     type=float, default=0.0)
    parser.add_argument('--gamma_act',      type=float, default=0.1)
    parser.add_argument('--lambda_iso',     type=float, default=0.0)

    parser.add_argument('--fps', type=int, default=20)

    parser.add_argument('--n_train_hc', type=int, default=None,
                        help='K = number of HC subjects for fine-tune (single-K mode); '
                             'mutually exclusive with --n_train_hc_sweep')
    parser.add_argument('--n_train_hc_sweep', type=str, default=None,
                        help='comma-separated K list for K-sweep (e.g. "1,5,10,20"). '
                             '每 K 独立 output dir + checkpoint, 串行训练. '
                             '与 --n_train_hc 互斥.')
    parser.add_argument('--hc_eval_ratio', type=float, default=0.30,
                        help='fraction of HC held-out for eval (固定 across K)')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='random seed for HC split (固定 across K → nested)')

    parser.add_argument('--extract_only', action='store_true',
                        help='skip training, just extract latents from --checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                        help='skip training, eval MPJPE on held-out HC from --checkpoint')

    parser.add_argument('--auto_analyze', action='store_true',
                        help='auto run OOS subject AUROC after extract latents')
    parser.add_argument('--auto_analyze_knn_k', type=int, default=20,
                        help='knn_k for OOS pullback geodesic (default 20)')
    parser.add_argument('--auto_analyze_ridge_alpha', type=float, default=1.0,
                        help='Ridge alpha for OOS pullback (default 1.0)')
    parser.add_argument('--auto_analyze_methods', type=str, default=None,
                        help='comma-separated estimator names; default = full 23 methods '
                             '(see analyze_latents.run_oos_subject_auroc)')
    parser.add_argument('--auto_analyze_sensitivity', action='store_true',
                        help='additionally run knn_k sensitivity sweep on best mode '
                             '(adds 4 sweep methods, mirrors main Path G)')

    args = parser.parse_args()

    if args.n_train_hc is None and args.n_train_hc_sweep is None:
        parser.error('Must specify either --n_train_hc (single K) or '
                     '--n_train_hc_sweep (K list, e.g. "1,5,10,20")')
    if args.n_train_hc is not None and args.n_train_hc_sweep is not None:
        parser.error('--n_train_hc and --n_train_hc_sweep are mutually exclusive')

    if args.n_train_hc_sweep is not None:
        try:
            k_list = [int(s.strip()) for s in args.n_train_hc_sweep.split(',') if s.strip()]
        except ValueError as e:
            parser.error(f'invalid --n_train_hc_sweep "{args.n_train_hc_sweep}": {e}')
        if len(k_list) == 0:
            parser.error('--n_train_hc_sweep is empty after parsing')
        sweep_mode = True
    else:
        k_list = [args.n_train_hc]
        sweep_mode = False

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    dt = 1.0 / args.fps
    print(f'Using dt={dt:.4f} (fps={args.fps})')

    if sweep_mode:
        print(f'\n{"="*72}')
        print(f'K-SWEEP MODE: K ∈ {k_list}')
        print(f'  base output_dir = {args.output_dir}')
        print(f'  base checkpoint = {args.checkpoint_out}')
        print(f'  per-K dir       = {args.output_dir}/k{{K}}/')
        print(f'  per-K checkpt   = <base>_k{{K}}.pt')
        print(f'{"="*72}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPL(device=device)

    sweep_results = []
    for K in k_list:
        try:
            result = run_one_k(
                args, K, device, smpl, dt,
                base_output_dir=args.output_dir,
                base_checkpoint_out=args.checkpoint_out,
            )
            sweep_results.append(result)
        except Exception as e:
            import traceback
            err_msg = f'{type(e).__name__}: {e}'
            print(f'\n{"!"*72}')
            print(f'! K={K} FAILED: {err_msg}')
            print(f'! Continuing to next K. Traceback:')
            traceback.print_exc()
            print(f'{"!"*72}')
            sweep_results.append({
                'K': K, 'output_dir': os.path.join(args.output_dir, f'k{K}'),
                'final_test_metrics': None, 'oos_results': None,
                'status': f'error: {err_msg}',
            })

    if sweep_mode and len(sweep_results) > 1:
        print(f'\n\n{"="*100}')
        print(f'K-SWEEP CROSS-K SUMMARY')
        print(f'{"="*100}')

        print(f'\n--- Held-out HC physics metrics across K (best checkpoint) ---')
        metric_keys = ['MPJPE (mm)', 'PA-MPJPE (mm)', 'ACCEL (mm/s²)',
                        'EL residual (N·m)', 'KE pred (J)', 'KE gt (J)', 'KE rel err']
        header = f'{"K":>4s}  ' + '  '.join(f'{k:>16s}' for k in metric_keys)
        print(header)
        print('-' * len(header))
        for r in sweep_results:
            if r['final_test_metrics'] is None:
                print(f'{r["K"]:>4d}  ' + '  '.join(f'{"--":>16s}' for _ in metric_keys)
                      + f'  [{r["status"]}]')
                continue
            metrics = r['final_test_metrics']
            row = f'{r["K"]:>4d}  '
            for k in metric_keys:
                v = metrics.get(k)
                if v is None:
                    row += f'{"--":>16s}  '
                elif 'rel err' in k:
                    row += f'{v:>15.1%}  '
                else:
                    row += f'{v:>16.4f}  '
            print(row.rstrip())

        if any(r.get('oos_results') for r in sweep_results):
            paper_methods = [
                ('Euclidean (HC center)',           'mean'),
                ('Euclidean (GMM)',                 'mean_gmm'),
                ('Riemannian (linear, M)',          'pullback_riem_M'),
                ('Riemannian (linear, M^-1)',       'pullback_riem'),
                ('Riemannian (geodesic, HC center)','pullback_geo_pca22_drop1_hc_center'),
                ('Riemannian (geodesic, ours)',     'pullback_geo_pca22_drop1'),
            ]
            print(f'\n--- OOS Subject HC/PD AUROC across K (paper main methods) ---')
            header = f'{"Method":40s}' + ''.join(
                f'  {"K=" + str(r["K"]):>10s}' for r in sweep_results
            )
            print(header)
            print('-' * len(header))
            for label, key in paper_methods:
                row = f'{label:40s}'
                for r in sweep_results:
                    oos = r.get('oos_results')
                    if oos is None or key not in oos:
                        row += f'  {"--":>10s}'
                    else:
                        row += f'  {oos[key]["auroc_pd_subj"]:>10.4f}'
                print(row)

            print(f'\n--- OOS Subject HC/all-anom AUROC across K ---')
            print(header)
            print('-' * len(header))
            for label, key in paper_methods:
                row = f'{label:40s}'
                for r in sweep_results:
                    oos = r.get('oos_results')
                    if oos is None or key not in oos:
                        row += f'  {"--":>10s}'
                    else:
                        row += f'  {oos[key]["auroc_all_subj"]:>10.4f}'
                print(row)

        import json
        summary_path = os.path.join(args.output_dir, 'sweep_summary.json')
        json_payload = []
        for r in sweep_results:
            entry = {
                'K':           r['K'],
                'output_dir':  r['output_dir'],
                'status':      r['status'],
                'final_test_metrics': r['final_test_metrics'],
            }
            if r['oos_results'] is not None:
                entry['oos_summary'] = {
                    m: {k: float(v[k]) for k in v if k.startswith('auroc_')}
                    for m, v in r['oos_results'].items()
                }
            else:
                entry['oos_summary'] = None
            json_payload.append(entry)
        with open(summary_path, 'w') as f:
            json.dump(json_payload, f, indent=2)
        print(f'\nSweep summary saved to {summary_path}')

    print(f'\nDone. Output: {args.output_dir}')
