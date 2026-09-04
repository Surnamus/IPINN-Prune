#!/usr/bin/env python3
"""
Post-LBFGS pruning-mask verification.

Checks, AFTER the fact and without rerunning anything, whether LBFGS
refinement respected the pruning mask -- i.e. whether every weight that
was exactly zero (pruned) in the pre-LBFGS checkpoint is STILL zero
(up to float32 noise) in the LBFGS-refined checkpoint.

How the mask is recovered: model1lbfgs.py / model2lbfgs.py never save
`masks` themselves, but they build it identically every time from the
pre-LBFGS checkpoint:

    masks = [(p != 0).clone() for p in model.parameters() if p.dim() == 2]

Filtering a state_dict to dim()==2 entries reproduces that exact same
set of tensors (Linear .weight, in order) -- so we don't need to
instantiate the model or run anything, just load two .pt files and
compare tensors directly. This makes the check essentially free.

What "violated" means here: a pruned position (mask == False, i.e. the
pre-LBFGS weight was exactly 0) whose POST-LBFGS value has drifted
beyond `TOL` in absolute value. If LBFGS's grad-masking
(`p.grad *= masks[i]`) worked as intended, every pruned position's
gradient -- and therefore every (s_k, y_k) curvature pair contributed
to L-BFGS's history at that position -- is exactly zero on every
internal iteration, so the position should stay at its original value
up to floating-point rounding only. A small nonzero residue (~1e-7,
float32's own precision floor) is expected and harmless; a residue
comparable in magnitude to the network's ALIVE weights means pruning
was effectively undone, which is what this script is meant to catch.

Usage:
    python verify_lbfgs_masks.py [--tol 1e-6] [--checkpoint-dir checkpoints]
"""
import argparse
import glob
import os
import re

import torch

DEFAULT_TOL = 1e-6


def parse_model1_lbfgs(path):
    m = re.search(r"model1_lbfgs_sparsity([0-9.]+)_angle([0-9.]+)\.pt$", os.path.basename(path))
    if not m:
        return None
    return {"sparsity": m.group(1), "angle": m.group(2)}


def parse_model2_lbfgs(path):
    m = re.search(r"model2_lbfgs_sparsity([0-9.]+)\.pt$", os.path.basename(path))
    if not m:
        return None
    return {"sparsity": m.group(1)}


def weight_tensors(state_dict):
    """[(name, tensor)] for every 2D parameter, in state_dict order --
    the exact same filter *_lbfgs.py used to build `masks` in the first
    place (Linear.bias is 1D and is skipped, same as the original code)."""
    return [(k, v) for k, v in state_dict.items() if v.dim() == 2]


def verify_pair(pre_path, post_path, tol):
    pre_ckpt = torch.load(pre_path, map_location="cpu")
    post_ckpt = torch.load(post_path, map_location="cpu")
    pre_weights = weight_tensors(pre_ckpt["model_state_dict"])
    post_weights = weight_tensors(post_ckpt["model_state_dict"])

    if len(pre_weights) != len(post_weights):
        return {"ok": False, "error": "layer count mismatch between pre/post checkpoints -- architectures don't match"}

    layer_reports = []
    worst_violation = 0.0
    total_pruned = 0
    total_violated = 0

    for (name, w_pre), (_, w_post) in zip(pre_weights, post_weights):
        pruned_mask = (w_pre == 0)          # exactly what *_lbfgs.py's masks list marks as "pruned"
        n_pruned = int(pruned_mask.sum().item())
        if n_pruned == 0:
            continue  # this layer wasn't pruned at all (e.g. RigL keeps the first layer fully dense)

        post_at_pruned = w_post[pruned_mask]
        max_abs = float(post_at_pruned.abs().max().item()) if post_at_pruned.numel() else 0.0
        n_violated = int((post_at_pruned.abs() > tol).sum().item())

        worst_violation = max(worst_violation, max_abs)
        total_pruned += n_pruned
        total_violated += n_violated

        layer_reports.append({
            "layer": name, "n_pruned": n_pruned, "n_violated": n_violated,
            "max_abs_at_pruned": max_abs,
        })

    return {
        "ok": total_violated == 0,
        "total_pruned": total_pruned,
        "total_violated": total_violated,
        "worst_violation": worst_violation,
        "layers": layer_reports,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                         help="max |weight| at a pruned position before it counts as 'violated' (default: 1e-6)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    lbfgs_paths = (
        sorted(glob.glob(os.path.join(ckpt_dir, "model1_lbfgs_sparsity*_angle*.pt")))
        + sorted(glob.glob(os.path.join(ckpt_dir, "model2_lbfgs_sparsity*.pt")))
    )

    if not lbfgs_paths:
        print(f"No model1_lbfgs_* / model2_lbfgs_* checkpoints found under {ckpt_dir}/")
        return

    print(f"tolerance = {args.tol:.1e}  (a pruned weight's |value| must stay below this to count as 'held')\n")
    print(f"{'checkpoint':45s} {'pruned':>8s} {'violated':>9s} {'max|w| at pruned':>18s} {'verdict':>10s}")

    any_bad = False
    any_checked = False
    for post_path in lbfgs_paths:
        name = os.path.basename(post_path)
        info1 = parse_model1_lbfgs(post_path)
        info2 = None if info1 else parse_model2_lbfgs(post_path)

        if info1:
            pre_path = os.path.join(ckpt_dir, f"model1_checkpoint_sparsity{info1['sparsity']}_angle{info1['angle']}.pt")
        elif info2:
            pre_path = os.path.join(ckpt_dir, f"model2_checkpoint_sparsity{info2['sparsity']}.pt")
        else:
            print(f"{name:45s}  (unrecognized filename pattern, skipped)")
            continue

        if not os.path.exists(pre_path):
            print(f"{name:45s}  (pre-LBFGS checkpoint not found: {pre_path}, skipped)")
            continue

        report = verify_pair(pre_path, post_path, args.tol)
        if "error" in report:
            print(f"{name:45s}  ERROR: {report['error']}")
            any_bad = True
            continue

        any_checked = True
        verdict = "OK" if report["ok"] else "VIOLATED"
        if not report["ok"]:
            any_bad = True
        print(f"{name:45s} {report['total_pruned']:8d} {report['total_violated']:9d} "
              f"{report['worst_violation']:18.3e} {verdict:>10s}")

        if not report["ok"]:
            for lr in report["layers"]:
                if lr["n_violated"] > 0:
                    print(f"    -> {lr['layer']}: {lr['n_violated']}/{lr['n_pruned']} pruned entries "
                          f"exceed tol, max|w|={lr['max_abs_at_pruned']:.3e}")

    print()
    if not any_checked:
        print("RESULT: no checkpoint pairs could be checked (see skip reasons above).")
    elif any_bad:
        print("RESULT: at least one checkpoint has pruned weights that moved well beyond the float32")
        print("        noise floor -- pruning was likely undone (or partially undone) during LBFGS.")
    else:
        print("RESULT: every checked checkpoint kept its pruned weights at (numerically) zero --")
        print("        masking held throughout LBFGS for all of them.")


if __name__ == "__main__":
    main()
