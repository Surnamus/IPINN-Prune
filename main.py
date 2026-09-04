import os
import re
import csv
import glob
import torch
import torch.nn as nn
from cfdsolver import generategrid

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(2, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 20), nn.Tanh(),
        nn.Linear(20, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


IPINN = PINN  # the inverse models use the exact same architecture

CHECKPOINT_DIR = "checkpoints"
CSV_DIR = "results_csv"

# Known ground-truth viscosity for this Burgers benchmark.
NU_TRUE = 0.01 / torch.pi


def load_checkpoint(path, model, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run the corresponding training script first."
        )
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


X_RANGE = (-1.0, 1.0)
T_RANGE = (0.0, 3.0 / torch.pi)


def sample_random_points(n_points, device, x_range=X_RANGE, t_range=T_RANGE):
    x_lo, x_hi = x_range
    t_lo, t_hi = t_range
    rand_x = x_lo + (x_hi - x_lo) * torch.rand(n_points, 1, device=device)
    rand_t = t_lo + (t_hi - t_lo) * torch.rand(n_points, 1, device=device)
    return rand_x, rand_t


def evaluate_model(model, x, t, label):
    with torch.no_grad():
        u_pred = model(x, t)
    print(f"  {label}: shape={tuple(u_pred.shape)}, "
          f"min={u_pred.min().item():.4f}, max={u_pred.max().item():.4f}, mean={u_pred.mean().item():.4f}")
    return u_pred


def find_finished_variants(pattern):
    """Glob checkpoints matching `pattern` under CHECKPOINT_DIR, keeping only
    the FINISHED, non-intermediate saves (per-step checkpoints have '_step'
    in the filename and are skipped)."""
    paths = glob.glob(os.path.join(CHECKPOINT_DIR, pattern))
    return sorted(p for p in paths if "_step" not in os.path.basename(p))


def parse_model1_name(path):
    m = re.search(r"sparsity([0-9.]+)_angle([0-9.]+)", os.path.basename(path))
    if not m:
        return None
    return {"sparsity": float(m.group(1).rstrip('.')), "angle": float(m.group(2).rstrip('.'))}


def parse_model2_name(path):
    m = re.search(r"sparsity([0-9.]+)", os.path.basename(path))
    if not m:
        return None
    return {"sparsity": float(m.group(1).rstrip('.'))}


def mse(a, b):
    return torch.mean((a - b) ** 2).item()


# ---------------------------------------------------------------------------
# Ground-truth error metrics — standard, named quantities only.
#   Absolute Error      AE  = |nu - nu_ref|
#   Relative Error (%)  RE% = AE / nu_ref * 100
# ---------------------------------------------------------------------------
def compute_metrics(nu_pred, nu_ref=NU_TRUE):
    abs_err = abs(nu_pred - nu_ref)
    rel_err_pct = abs_err / nu_ref * 100.0
    return {"abs_err": abs_err, "rel_err_pct": rel_err_pct}


def print_table(headers, rows, title=None):
    if title:
        print(f"\n{title}")
    if not rows:
        print("  (no entries)")
        return
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, floatfmt=".6f", tablefmt="github"))
        return
    def cell(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v)
    widths = [
        max(len(str(h)), max((len(cell(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    def fmt_row(row):
        return "  ".join(f"{cell(v):<{w}}" for v, w in zip(row, widths))
    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def write_csv(filename, headers, rows):
    os.makedirs(CSV_DIR, exist_ok=True)
    path = os.path.join(CSV_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  [csv] wrote {path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Ground-truth nu = 0.01/pi = {NU_TRUE:.6f}")

    torch.manual_seed(0)  # remove/change this for different random points each run
    N_RANDOM_POINTS = 1000

    X, T, vu, points = generategrid()
    grid_x = torch.tensor(X.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    grid_t = torch.tensor(T.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    rand_x, rand_t = sample_random_points(N_RANDOM_POINTS, device)

    print("--- Testing Inverse PINN (modelbase) ---")
    inverse_model = IPINN().to(device)
    ckpt = load_checkpoint("checkpoints/modelbase_checkpoint.pt", inverse_model, device)
    nu_base = ckpt["nu"]
    inverse_model.eval()
    print(f"  Discovered nu: {nu_base.item():.6f}")
    base_grid_pred = evaluate_model(inverse_model, grid_x, grid_t, "grid points")
    evaluate_model(inverse_model, rand_x, rand_t, "random points")

    print("\n--- Testing LBFGS-refined Inverse PINN (modelbase) ---")
    lbfgs_base_model = IPINN().to(device)
    ckpt_lbfgs_base = load_checkpoint("checkpoints/modelbase_lbfgs.pt", lbfgs_base_model, device)
    nu_lbfgs_base = ckpt_lbfgs_base["nu"]
    lbfgs_base_model.eval()
    print(f"  Refined nu (modelbase): {nu_lbfgs_base.item():.6f}")
    evaluate_model(lbfgs_base_model, grid_x, grid_t, "grid points")
    evaluate_model(lbfgs_base_model, rand_x, rand_t, "random points")

    # -----------------------------------------------------------------
    # Loop-eval every FINISHED (non-step) model1 variant: one entry per
    # (sparsity, angle) combo, for both the raw pruned checkpoint and its
    # lbfgs-refined counterpart.
    # -----------------------------------------------------------------
    print("\n--- Evaluating all finished model1 variants (train + lbfgs) ---")
    model1_results = []
    model1_paths = (
        find_finished_variants("model1_checkpoint_sparsity*_angle*.pt")
        + find_finished_variants("model1_lbfgs_sparsity*_angle*.pt")
    )
    for path in model1_paths:
        info = parse_model1_name(path)
        if info is None:
            continue
        stage = "lbfgs" if "_lbfgs_" in os.path.basename(path) else "train"
        m = IPINN().to(device)
        ckpt_v = load_checkpoint(path, m, device)
        m.eval()
        nu_v = ckpt_v["nu"]
        label = f"model1 sparsity={info['sparsity']} angle={info['angle']}deg [{stage}]"
        print(f"  {label}: nu={nu_v.item():.6f}")
        grid_pred = evaluate_model(m, grid_x, grid_t, label + " grid")
        evaluate_model(m, rand_x, rand_t, label + " random")
        model1_results.append({
            "path": path, "model": "model1", "stage": stage,
            "sparsity": info["sparsity"], "angle": info["angle"],
            "nu": nu_v.item(), "grid_pred": grid_pred,
        })

    # -----------------------------------------------------------------
    # Same for model2: one entry per sparsity, train + lbfgs.
    # -----------------------------------------------------------------
    print("\n--- Evaluating all finished model2 variants (train + lbfgs) ---")
    model2_results = []
    model2_paths = (
        find_finished_variants("model2_checkpoint_sparsity*.pt")
        + find_finished_variants("model2_lbfgs_sparsity*.pt")
    )
    for path in model2_paths:
        info = parse_model2_name(path)
        if info is None:
            continue
        stage = "lbfgs" if "_lbfgs_" in os.path.basename(path) else "train"
        m = IPINN().to(device)
        ckpt_v = load_checkpoint(path, m, device)
        m.eval()
        nu_v = ckpt_v["nu"]
        label = f"model2 sparsity={info['sparsity']} [{stage}]"
        print(f"  {label}: nu={nu_v.item():.6f}")
        grid_pred = evaluate_model(m, grid_x, grid_t, label + " grid")
        evaluate_model(m, rand_x, rand_t, label + " random")
        model2_results.append({
            "path": path, "model": "model2", "stage": stage,
            "sparsity": info["sparsity"], "angle": None,
            "nu": nu_v.item(), "grid_pred": grid_pred,
        })

    # -----------------------------------------------------------------
    # Comparison against modelbase (field-prediction MSE + nu relative error).
    # -----------------------------------------------------------------
    print("\n--- Accuracy comparison against modelbase ---")
    all_results = model1_results + model2_results
    for r in all_results:
        r["field_mse_vs_base"] = mse(r["grid_pred"], base_grid_pred)
        r["nu_diff_vs_base"] = abs(r["nu"] - nu_base.item())
        r["rel_err_vs_base_pct"] = r["nu_diff_vs_base"] / nu_base.item() * 100.0

    all_results.sort(key=lambda r: r["field_mse_vs_base"])

    print("\nRanking (closest to modelbase first, by grid-field MSE):")
    for rank, r in enumerate(all_results, start=1):
        tag = os.path.basename(r["path"])
        print(f"  {rank}. {tag}  field_mse={r['field_mse_vs_base']:.6e}  nu_diff={r['nu_diff_vs_base']:.6f}"
              f"  rel_err_vs_base={r['rel_err_vs_base_pct']:.2f}%")

    if all_results:
        best = all_results[0]
        print(f"\nClosest to modelbase (ground-truth proxy): {os.path.basename(best['path'])} "
              f"(field_mse={best['field_mse_vs_base']:.6e}, nu_diff={best['nu_diff_vs_base']:.6f})")
    else:
        print("\nNo finished model1/model2 variant checkpoints found yet.")

    write_csv(
        "vs_modelbase_ranking.csv",
        ["rank", "checkpoint", "field_mse_vs_base", "nu_diff_vs_base", "rel_err_vs_base_pct"],
        [[i + 1, os.path.basename(r["path"]), r["field_mse_vs_base"], r["nu_diff_vs_base"], r["rel_err_vs_base_pct"]]
         for i, r in enumerate(all_results)],
    )

    # ===================================================================
    # Ground-truth (nu = 0.01/pi) error tables — Absolute Error and
    # Relative Error % only. (No "accuracy" construct: it isn't a
    # standard named metric, just the complement of error, and doesn't
    # add information beyond the error itself.)
    # ===================================================================
    gt_entries = []

    def add_gt_entry(label, nu_value, extra=None):
        m = compute_metrics(nu_value, NU_TRUE)
        row = {"label": label, "nu": nu_value, **m}
        if extra:
            row.update(extra)
        gt_entries.append(row)

    add_gt_entry("modelbase [train]", nu_base.item())
    add_gt_entry("modelbase [lbfgs]", nu_lbfgs_base.item())
    for r in model1_results:
        add_gt_entry(
            f"model1 sparsity={r['sparsity']} angle={r['angle']} [{r['stage']}]",
            r["nu"], extra={"model": "model1", "sparsity": r["sparsity"], "angle": r["angle"], "stage": r["stage"]},
        )
    for r in model2_results:
        add_gt_entry(
            f"model2 sparsity={r['sparsity']} [{r['stage']}]",
            r["nu"], extra={"model": "model2", "sparsity": r["sparsity"], "angle": None, "stage": r["stage"]},
        )

    headers_common = ["Rank", "Checkpoint", "nu", "nu_true"]

    print("\n" + "=" * 70)
    print("GROUND-TRUTH COMPARISON TABLES (nu_true = 0.01/pi = {:.6f})".format(NU_TRUE))
    print("=" * 70)

    # Table 1: Absolute Error (ascending — lower is better)
    by_abs_err = sorted(gt_entries, key=lambda e: e["abs_err"])
    rows = [[i + 1, e["label"], e["nu"], NU_TRUE, e["abs_err"]] for i, e in enumerate(by_abs_err)]
    print_table(headers_common + ["AbsoluteError"], rows,
                title="Table 1/2 — Absolute Error |nu - nu_true|  (sorted best-first)")
    write_csv("table1_absolute_error.csv", headers_common + ["AbsoluteError"], rows)

    # Table 2: Relative Error % (ascending — lower is better)
    by_rel_err = sorted(gt_entries, key=lambda e: e["rel_err_pct"])
    rows = [[i + 1, e["label"], e["nu"], NU_TRUE, e["rel_err_pct"]] for i, e in enumerate(by_rel_err)]
    print_table(headers_common + ["RelativeError_%"], rows,
                title="Table 2/2 — Relative Error %  (sorted best-first)")
    write_csv("table2_relative_error.csv", headers_common + ["RelativeError_%"], rows)

    # -------------------------------------------------------------
    # model1-only and model2-only detail tables: both error metrics
    # as columns, sorted by relative error (ascending — best first).
    # -------------------------------------------------------------
    detail_headers = ["Rank", "Sparsity", "Angle", "Stage", "nu", "AbsErr", "RelErr_%"]

    model1_gt = [e for e in gt_entries if e.get("model") == "model1"]
    model1_gt.sort(key=lambda e: e["rel_err_pct"])
    rows = [[i + 1, e["sparsity"], e["angle"], e["stage"], e["nu"], e["abs_err"], e["rel_err_pct"]]
            for i, e in enumerate(model1_gt)]
    print_table(detail_headers, rows,
                title="\nTable 3/4 — model1 variants: error vs ground truth (sorted by RelErr%, best first)")
    write_csv("table3_model1_detail.csv", detail_headers, rows)

    model2_gt = [e for e in gt_entries if e.get("model") == "model2"]
    model2_gt.sort(key=lambda e: e["rel_err_pct"])
    rows = [[i + 1, e["sparsity"], e["angle"], e["stage"], e["nu"], e["abs_err"], e["rel_err_pct"]]
            for i, e in enumerate(model2_gt)]
    print_table(detail_headers, rows,
                title="Table 4/4 — model2 variants: error vs ground truth (sorted by RelErr%, best first)")
    write_csv("table4_model2_detail.csv", detail_headers, rows)

    if gt_entries:
        overall_best = min(gt_entries, key=lambda e: e["rel_err_pct"])
        print(f"\nBest overall vs ground truth: {overall_best['label']} "
              f"(nu={overall_best['nu']:.6f}, RelErr={overall_best['rel_err_pct']:.2f}%)")

    print(f"\nAll tables exported to ./{CSV_DIR}/")
