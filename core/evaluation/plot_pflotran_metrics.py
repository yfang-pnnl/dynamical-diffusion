import os, glob, re
import numpy as np
import matplotlib.pyplot as plt

def load_stats(stats_path):
    stats = np.load(stats_path, allow_pickle=True).item()
    mu_c = stats["mu_c"]; std_c = stats["std_c"]
    return mu_c, std_c, stats

def inv_transform(logc_norm, mu_c, std_c):
    """normalized log1p(conc) -> conc"""
    logc = logc_norm * std_c + mu_c
    conc = np.expm1(logc)
    return conc

def rmse(a, b, eps=1e-12):
    return np.sqrt(np.mean((a - b) ** 2) + eps)

def crps_ensemble(samples, truth):
    """
    samples: (K, H, W) or (K, N)
    truth:   (H, W) or (N,)
    Ensemble CRPS: E|X-y| - 0.5 E|X-X'|
    """
    X = samples.reshape(samples.shape[0], -1)
    y = truth.reshape(-1)[None, :]
    term1 = np.mean(np.abs(X - y))
    term2 = 0.5 * np.mean(np.abs(X[:, None, :] - X[None, :, :]))
    return term1 - term2

def plume_mass(conc_hw, poro_hw, cell_area=1.0):
    return np.sum(conc_hw * poro_hw) * cell_area

def plume_extent(conc_hw, thresh=1e-8):
    return np.mean(conc_hw > thresh)

def list_case_ids(root):
    gts = sorted(glob.glob(os.path.join(root, "gt_*.npy")))
    ids = []
    for p in gts:
        m = re.search(r"gt_(\d+)\.npy$", os.path.basename(p))
        if m:
            ids.append(int(m.group(1)))
    return ids

def load_case(root, case_id):
    gt = np.load(os.path.join(root, f"gt_{case_id}.npy"))
    # expected: (B, total_len, C, H, W), with B=1, C=1
    gt = gt[0]  # (total_len, C, H, W)

    pred_paths = sorted(glob.glob(os.path.join(root, f"pred_{case_id}_sample_*.npy")))
    if len(pred_paths) == 0:
        raise RuntimeError(f"No predictions found for case {case_id}")
    preds = np.stack([np.load(p)[0] for p in pred_paths], axis=0)
    # preds: (K, total_len, C, H, W)
    return gt, preds

def evaluate_and_plot(
    model_output_root,
    stats_path,
    input_length,
    pred_length=20,
    thresh=1e-8,
    out_dir=None,
    max_cases=None,
):
    if out_dir is None:
        out_dir = os.path.join(model_output_root, "plots")
    os.makedirs(out_dir, exist_ok=True)

    mu_c, std_c, stats = load_stats(stats_path)

    case_ids = list_case_ids(model_output_root)
    if max_cases is not None:
        case_ids = case_ids[:max_cases]
    if len(case_ids) == 0:
        raise RuntimeError(f"No gt_*.npy found in {model_output_root}")

    # Store metrics per lead time (1..pred_length)
    rmse_by_lead = np.zeros(pred_length, dtype=np.float64)
    crps_by_lead = np.zeros(pred_length, dtype=np.float64)
    mass_rmse_by_lead = np.zeros(pred_length, dtype=np.float64)
    extent_rmse_by_lead = np.zeros(pred_length, dtype=np.float64)
    cover90_by_lead = np.zeros(pred_length, dtype=np.float64)  # 5-95% interval coverage

    n_cases_used = 0

    for cid in case_ids:
        gt_log, preds_log = load_case(model_output_root, cid)

        # Slice future window (log-space, normalized)
        gt_f_log = gt_log[input_length:input_length+pred_length, 0]             # (S,H,W)
        preds_f_log = preds_log[:, input_length:input_length+pred_length, 0]    # (K,S,H,W)

        # Inverse transform to physical conc
        gt_f = inv_transform(gt_f_log, mu_c, std_c)              # (S,H,W)
        preds_f = inv_transform(preds_f_log, mu_c, std_c)        # (K,S,H,W)

        # Porosity: not in gt/preds when you pass perm/poro as conditioning.
        # If you saved poro separately in evaluation, load it there.
        # For now: plume mass/ext require poro. You can provide poro_hw via an external file.
        # Workaround: skip mass/ext if poro not available.
        poro_hw = None

        # If you DO have poro for test windows, set poro_hw here (H,W) for this realization.
        # Example: poro_hw = np.load("test_static.npy")[real_idx, 1]  # if stored

        # Compute per-lead metrics
        for l in range(pred_length):
            gt_hw = gt_f[l]
            samples_hw = preds_f[:, l]          # (K,H,W)
            mean_hw = samples_hw.mean(axis=0)

            rmse_by_lead[l] += rmse(mean_hw, gt_hw)
            crps_by_lead[l] += crps_ensemble(samples_hw, gt_hw)

            # Coverage for 90% interval
            lo = np.quantile(samples_hw, 0.05, axis=0)
            hi = np.quantile(samples_hw, 0.95, axis=0)
            cover90_by_lead[l] += np.mean((gt_hw >= lo) & (gt_hw <= hi))

            # Plume metrics if poro exists
            if poro_hw is not None:
                gt_m = plume_mass(gt_hw, poro_hw)
                pred_m = plume_mass(mean_hw, poro_hw)
                mass_rmse_by_lead[l] += (pred_m - gt_m) ** 2

                gt_e = plume_extent(gt_hw, thresh)
                pred_e = plume_extent(mean_hw, thresh)
                extent_rmse_by_lead[l] += (pred_e - gt_e) ** 2

        n_cases_used += 1

    # Average over cases
    rmse_by_lead /= n_cases_used
    crps_by_lead /= n_cases_used
    cover90_by_lead /= n_cases_used

    # finalize plume metrics (if poro existed)
    if np.any(mass_rmse_by_lead):
        mass_rmse_by_lead = np.sqrt(mass_rmse_by_lead / n_cases_used)
    else:
        mass_rmse_by_lead = None

    if np.any(extent_rmse_by_lead):
        extent_rmse_by_lead = np.sqrt(extent_rmse_by_lead / n_cases_used)
    else:
        extent_rmse_by_lead = None

    lead = np.arange(1, pred_length + 1)

    # --- Plot 1: RMSE vs lead
    plt.figure()
    plt.plot(lead, rmse_by_lead, marker="o")
    plt.xlabel("Lead time (steps)")
    plt.ylabel("RMSE (conc)")
    plt.title("RMSE vs lead time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_vs_lead.png"), dpi=200)
    plt.close()

    # --- Plot 2: CRPS vs lead
    plt.figure()
    plt.plot(lead, crps_by_lead, marker="o")
    plt.xlabel("Lead time (steps)")
    plt.ylabel("CRPS (conc)")
    plt.title("CRPS vs lead time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "crps_vs_lead.png"), dpi=200)
    plt.close()

    # --- Plot 3: Coverage vs lead (calibration)
    plt.figure()
    plt.plot(lead, cover90_by_lead, marker="o")
    plt.axhline(0.90, linestyle="--")
    plt.xlabel("Lead time (steps)")
    plt.ylabel("Empirical coverage (5–95%)")
    plt.title("90% interval coverage vs lead time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coverage90_vs_lead.png"), dpi=200)
    plt.close()

    # --- Optional plots: plume mass/ext RMSE if poro provided
    if mass_rmse_by_lead is not None:
        plt.figure()
        plt.plot(lead, mass_rmse_by_lead, marker="o")
        plt.xlabel("Lead time (steps)")
        plt.ylabel("RMSE (plume mass)")
        plt.title("Plume mass RMSE vs lead time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plume_mass_rmse_vs_lead.png"), dpi=200)
        plt.close()

    if extent_rmse_by_lead is not None:
        plt.figure()
        plt.plot(lead, extent_rmse_by_lead, marker="o")
        plt.xlabel("Lead time (steps)")
        plt.ylabel("RMSE (plume extent)")
        plt.title(f"Plume extent RMSE vs lead time (threshold={thresh:g})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plume_extent_rmse_vs_lead.png"), dpi=200)
        plt.close()

    # Save arrays too
    np.save(os.path.join(out_dir, "rmse_by_lead.npy"), rmse_by_lead)
    np.save(os.path.join(out_dir, "crps_by_lead.npy"), crps_by_lead)
    np.save(os.path.join(out_dir, "coverage90_by_lead.npy"), cover90_by_lead)

    print(f"Evaluated {n_cases_used} cases")
    print("Plots saved to:", out_dir)
    print("RMSE (lead 1..S):", rmse_by_lead)
    print("CRPS (lead 1..S):", crps_by_lead)
    print("Coverage90:", cover90_by_lead)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_output_root", required=True,
                    help="Directory containing gt_*.npy and pred_*_sample_*.npy")
    ap.add_argument("--stats_path", required=True,
                    help="normalization_stats.npy used for inverse transform")
    ap.add_argument("--input_length", type=int, required=True)
    ap.add_argument("--pred_length", type=int, default=20)
    ap.add_argument("--thresh", type=float, default=1e-8)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_cases", type=int, default=None)
    args = ap.parse_args()

    evaluate_and_plot(
        model_output_root=args.model_output_root,
        stats_path=args.stats_path,
        input_length=args.input_length,
        pred_length=args.pred_length,
        thresh=args.thresh,
        out_dir=args.out_dir,
        max_cases=args.max_cases,
    )

if __name__ == "__main__":
    main()

