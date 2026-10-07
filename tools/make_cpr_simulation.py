from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "components"))

from multi_scale_point_refiner import MultiScalePointRefiner


PATCH_SIZE = 21
SIM_SEED = 20260916
TARGET_CENTER = (10.35, 10.62)
TARGET_SIGMA = 1.50
TARGET_AMPLITUDE = 0.75
BACKGROUND = 0.085
GRADIENT = (0.020, -0.010)
NOISE_STD = 0.011
CLUTTER = False
HOT_PIXELS = ((4, 5, 0.62), (16, 4, 0.55), (15, 17, 0.70), (5, 16, 0.48))


SIGMA_RANGE = (1.4, 2.6)
OFFSET_RANGE = (1.0, 3.0)
BORDER = 2
MAX_TRIES_FACTOR = 4


def make_patch(seed: int = 1,
               patch_size: int = PATCH_SIZE,
               center=TARGET_CENTER,
               sigma: float = TARGET_SIGMA,
               amplitude: float = TARGET_AMPLITUDE,
               background: float = BACKGROUND,
               grad=GRADIENT,
               clutter: bool = CLUTTER,
               noise_std: float = NOISE_STD,
               hot_pixels=HOT_PIXELS) -> np.ndarray:
    n = patch_size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    img = (background
           + grad[0] * (xx / (n - 1))
           + grad[1] * (yy / (n - 1))
           + rng.normal(0.0, noise_std, (n, n)))

    if clutter:
        for _ in range(2):
            cx, cy = rng.uniform(1.5, n - 2.5, 2)
            sg, am = rng.uniform(2.6, 4.6), rng.uniform(0.07, 0.16)
            img += am * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg * sg))

    img += amplitude * np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
                                / (2 * sigma * sigma)))

    for hx, hy, hv in hot_pixels:
        img[hy, hx] = hv

    return np.clip(img, 0.0, 1.0)


def sample_coarse_point(rng, center=TARGET_CENTER, patch_size=PATCH_SIZE,
                        offset_range=OFFSET_RANGE, border=BORDER):
    ang = rng.uniform(0.0, 2.0 * math.pi)
    d = rng.uniform(*offset_range)
    x = int(round(center[0] + d * math.cos(ang)))
    y = int(round(center[1] + d * math.sin(ang)))
    if not (border <= x <= patch_size - border - 1
            and border <= y <= patch_size - border - 1):
        return None
    return (x, y)


def offset_to_center(point, center=TARGET_CENTER) -> float:
    return math.hypot(point[0] - center[0], point[1] - center[1])


def run_simulation(n: int = 200,
                   seed: int = SIM_SEED,
                   sigma_range=SIGMA_RANGE,
                   use_clutter: bool = True,
                   offset_range=OFFSET_RANGE,
                   max_tries_factor: int = MAX_TRIES_FACTOR,
                   verbose: bool = False) -> dict:
    refiner = MultiScalePointRefiner()
    rng = np.random.default_rng(seed)

    records = []
    tries = 0
    max_tries = n * max_tries_factor
    while len(records) < n and tries < max_tries:
        tries += 1

        patch_seed = int(rng.integers(1, 10 ** 6))
        sigma = float(rng.uniform(*sigma_range))

        clutter = bool(rng.integers(0, 2)) and use_clutter
        raw = make_patch(seed=patch_seed, sigma=sigma, clutter=clutter)

        coarse = sample_coarse_point(rng, offset_range=offset_range)
        if coarse is None:
            continue

        refined = refiner.refine(raw, coarse)

        records.append({
            "patch_seed": patch_seed,
            "target_sigma": sigma,
            "clutter": clutter,
            "coarse_point": list(coarse),
            "refined_point": list(refined),
            "offset_before": offset_to_center(coarse),
            "offset_after": offset_to_center(refined),
        })
        if verbose and len(records) % 50 == 0:
            print("  ... %d/%d" % (len(records), n))

    return {"summary": _summarize(records, n, tries, seed), "records": records}


def _summarize(records, n, tries, seed) -> dict:
    before = np.array([r["offset_before"] for r in records])
    after = np.array([r["offset_after"] for r in records])
    delta = after - before
    worse = delta > 0
    return {
        "n": len(records),
        "requested_n": n,
        "tries": tries,
        "seed": seed,
        "offset_before": {"mean": float(before.mean()),
                          "median": float(np.median(before)),
                          "max": float(before.max())},
        "offset_after": {"mean": float(after.mean()),
                         "median": float(np.median(after)),
                         "max": float(after.max())},
        "improved": int((delta < 0).sum()),
        "improved_fraction": float((delta < 0).mean()),
        "worsened": int(worse.sum()),
        "unchanged": int((delta == 0).sum()),
        "worsened_mean_increase": float(delta[worse].mean()) if worse.any() else 0.0,
        "worsened_max_increase": float(delta[worse].max()) if worse.any() else 0.0,
    }


def format_report(summary: dict) -> str:
    b, a = summary["offset_before"], summary["offset_after"]
    lines = [
        "CPR localization error on synthetic patches",
        "  n = %d   (attempts: %d, seed: %s)"
        % (summary["n"], summary["tries"], summary.get("seed")),
        "",
        "  offset to target center (px)   before    after",
        "  %-30s %7.3f  %7.3f" % ("mean", b["mean"], a["mean"]),
        "  %-30s %7.3f  %7.3f" % ("median", b["median"], a["median"]),
        "  %-30s %7.3f  %7.3f" % ("max", b["max"], a["max"]),
        "",
        "  reduced in %d/%d samples (%.1f%%)"
        % (summary["improved"], summary["n"], 100 * summary["improved_fraction"]),
        "  increased in %d (mean +%.2f, max +%.2f); unchanged %d"
        % (summary["worsened"], summary["worsened_mean_increase"],
           summary["worsened_max_increase"], summary["unchanged"]),
    ]
    return "\n".join(lines)


def write_json(path: str, result: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)


def write_csv(path: str, records) -> None:
    fields = ["patch_seed", "target_sigma", "clutter", "coarse_point",
              "refined_point", "offset_before", "offset_after"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="统计组合式点校正（CPR）在合成红外图像块上的定位误差。")
    ap.add_argument("--n", type=int, default=200, help="图像块数量（默认 200）")
    ap.add_argument("--seed", type=int, default=SIM_SEED, help="抽样随机种子")
    ap.add_argument("--no-clutter", action="store_true",
                    help="不叠加低频背景杂波（默认一半样本叠加）")
    ap.add_argument("--json", metavar="PATH", help="把逐样本记录与汇总写入 JSON")
    ap.add_argument("--csv", metavar="PATH", help="把逐样本记录写入 CSV")
    ap.add_argument("--quiet", action="store_true", help="只输出汇总")
    args = ap.parse_args(argv)

    result = run_simulation(n=args.n, seed=args.seed,
                            use_clutter=not args.no_clutter,
                            verbose=not args.quiet)
    print(format_report(result["summary"]))

    if args.json:
        write_json(args.json, result)
        print("\nwrote %s" % args.json)
    if args.csv:
        write_csv(args.csv, result["records"])
        print("wrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
