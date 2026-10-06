"""
Permutation test for SiE paper: NI vs LRU on AIME 2024-2025.

Per-problem sign-flip test.  For each of the 60 AIME problems we
compute the mean correctness (across seeds) for NI and for LRU, then
test whether the mean of (NI_mean[i] - LRU_mean[i]) is significantly
positive.  Under H0, the NI/LRU labels are exchangeable for each
problem independently; we approximate the null by randomly flipping
the sign of each per-problem difference.

GPU-free: reads existing binary outcome vectors from results_per_sample.jsonl.

Usage:
    python permutation_test.py \
        --deepseek results/aime_ablation_deepseek/results_per_sample.jsonl \
        --qwen     results/aime_ablation/results_per_sample.jsonl
"""

import argparse
import json
import numpy as np


# --------------------------------------------------------------------------- #
def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_binary_vectors(rows: list[dict], method: str, budget: int,
                        seeds: list[int]) -> dict[int, np.ndarray]:
    """Return {seed: np.array shape (n_problems,)} with 0/1 correctness."""
    result: dict[int, list] = {}
    for r in rows:
        if r["method"] != method:
            continue
        if r.get("budget") != budget:
            continue
        s = r["seed"]
        if s not in seeds:
            continue
        result.setdefault(s, []).append((r["idx"], int(r["correct"])))
    return {
        s: np.array([c for _, c in sorted(items)])
        for s, items in result.items()
    }


def permutation_test(ni_vecs: list[np.ndarray],
                     lru_vecs: list[np.ndarray],
                     n_perm: int = 200_000,
                     rng_seed: int = 42) -> tuple[float, float]:
    """
    One-sided (NI > LRU) per-problem sign-flip permutation test.

    Parameters
    ----------
    ni_vecs, lru_vecs : list of np.ndarray, each shape (60,)
    n_perm            : number of Monte-Carlo permutations
    rng_seed          : reproducibility seed

    Returns
    -------
    obs_stat : observed mean(NI - LRU) per problem, averaged over seeds
    p_value  : one-sided p-value (fraction of null stats >= obs_stat)
    """
    rng = np.random.default_rng(rng_seed)
    n_problems = len(ni_vecs[0])

    ni_mean  = np.mean(ni_vecs, axis=0)   # (n_problems,)
    lru_mean = np.mean(lru_vecs, axis=0)  # (n_problems,)
    diff     = ni_mean - lru_mean          # (n_problems,)
    obs_stat = float(diff.mean())

    count = 0
    for _ in range(n_perm):
        signs = rng.integers(0, 2, size=n_problems) * 2 - 1  # ±1
        if float((diff * signs).mean()) >= obs_stat:
            count += 1

    return obs_stat, count / n_perm


# --------------------------------------------------------------------------- #
def run_analysis(label: str, per_sample_path: str,
                 seeds: list[int], budgets: list[int]) -> None:
    rows = load_jsonl(per_sample_path)
    print(f"\n{'=' * 62}")
    print(f"  {label}  |  seeds={seeds}")
    print(f"{'=' * 62}")
    print(f"  {'Budget':>7}  {'NI (mean±std)':>15}  {'LRU (mean±std)':>15}  "
          f"{'diff':>7}  {'p-value':>8}  sig")
    print(f"  {'-'*7}  {'-'*15}  {'-'*15}  {'-'*7}  {'-'*8}  ---")

    for bud in budgets:
        ni_vecs  = get_binary_vectors(rows, "novelty_inv", bud, seeds)
        lru_vecs = get_binary_vectors(rows, "lru",         bud, seeds)

        seeds_have = [s for s in seeds if s in ni_vecs and s in lru_vecs]
        if len(seeds_have) < len(seeds):
            missing = set(seeds) - set(seeds_have)
            print(f"  B={bud:5d}  WARNING: missing seeds {missing} for NI or LRU — skipping")
            continue

        ni_list  = [ni_vecs[s]  for s in seeds_have]
        lru_list = [lru_vecs[s] for s in seeds_have]

        ni_acc  = np.array([v.mean() for v in ni_list])
        lru_acc = np.array([v.mean() for v in lru_list])

        obs, p = permutation_test(ni_list, lru_list)

        sig = "p<.05" if p < 0.05 else ("p<.10" if p < 0.10 else "")
        print(f"  B={bud:5d}  "
              f"{ni_acc.mean():.3f}±{ni_acc.std(ddof=1):.3f}    "
              f"{lru_acc.mean():.3f}±{lru_acc.std(ddof=1):.3f}    "
              f"{obs:+.4f}  {p:.4f}   {sig}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek", required=True,
                        help="DeepSeek AIME results_per_sample.jsonl")
    parser.add_argument("--qwen", required=True,
                        help="Qwen3 AIME results_per_sample.jsonl")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 1234, 5678],
                        help="Seeds to include (default: 42 1234 5678)")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[2048, 4096, 8192, 16384],
                        help="KV budgets to test")
    parser.add_argument("--n_perm", type=int, default=200_000,
                        help="Number of Monte-Carlo permutations")
    args = parser.parse_args()

    print(f"Permutation test  |  n_perm={args.n_perm:,}  |  rng_seed=42")
    print("H0: NI = LRU per problem (one-sided, NI > LRU)")

    run_analysis("DeepSeek-R1-Distill-Llama-8B",
                 args.deepseek, args.seeds, args.budgets)
    run_analysis("Qwen3-8B",
                 args.qwen, args.seeds, args.budgets)

    print("\n")


if __name__ == "__main__":
    main()