#!/usr/bin/env python3
"""
R-KV importance / redundancy 스케일 진단
=======================================

목적
----
r_kv.compute_rkv_scores 의 joint score

    z = lam * imp_think - (1 - lam) * redundancy

에서 두 항의 실제 크기를 측정한다.

배경: redundancy 는 softmax(dim=0) 이므로 합이 정확히 1.0 이다.
반면 imp_think 는 output_attentions(전체 캐시 softmax)에서 think 구간만
슬라이스한 값이라 합이 1 보다 작다 (프롬프트가 attention mass 를 가져감).
원본 R-KV 는 candidate-only 재softmax 라 두 항이 같은 스케일이다.

따라서 우리 구현에서는 imp 항만 축소되어 z 가 redundancy-only 로
퇴화했을 가능성이 있다. 이 스크립트는 그 여부를 정량 확인한다.

방법
----
generation.py 와 r_kv.py 를 수정하지 않는다.
compute_rkv_scores 를 lam 만 바꿔 세 번 호출하면 두 항이 분리된다:

    lam=RKV_LAMBDA -> z   = lam*imp - (1-lam)*red   (실제 사용값)
    lam=1.0        -> z   = imp                     (importance 단독)
    lam=0.0        -> z   = -red                    (redundancy 단독)

generation.py 가 `from ..methods.r_kv import compute_rkv_scores` 로
이름을 자기 네임스페이스에 바인딩하므로,
src.core.generation.compute_rkv_scores 를 패치해야 한다.

핵심 지표
--------
  ratio_std   = std((1-lam)*red) / std(lam*imp)
                두 항이 z 의 순위에 기여하는 상대 크기.
                >> 1 이면 redundancy 가 지배.
  rho_z_red   = Spearman(z, -red)   1.0 에 가까우면 redundancy-only 로 퇴화
  rho_z_imp   = Spearman(z,  imp)
  imp_sum     = imp 합. 원본이면 1.0 이어야 함. << 1 이면 스케일 손실 확인.

사용법
-----
  # 리포 루트에서
  python diag_rkv_scale.py --n-samples 10 --budget 512

  # DeepSeek 로
  python diag_rkv_scale.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B

결과 해석
--------
  ratio_std > 10 이고 rho_z_red > 0.95  -> importance 가 사실상 무시됨.
      스케일 정규화 버그 확정. r_kv.py 의 max_pool 직후에
      imp_think = imp_think / imp_think.sum().clamp(min=1e-9) 추가 후 재실행 필요.
  ratio_std ~ 1-3                      -> 스케일은 정상. R-KV 저성능은 다른 원인.
"""
import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# ─────────────────────────────────────────────────────────────────────────────
# 순위 상관 (scipy 의존성 회피)
# ─────────────────────────────────────────────────────────────────────────────

def _rankdata(a: np.ndarray) -> np.ndarray:
    """average-tie ranking. scipy.stats.rankdata 와 동일 결과."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # tie 평균 처리
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


# ─────────────────────────────────────────────────────────────────────────────
# 진단 수집기
# ─────────────────────────────────────────────────────────────────────────────

class RKVScaleProbe:
    """compute_rkv_scores 를 감싸 두 항을 분리 기록하는 래퍼."""

    def __init__(self, orig_fn, lam: float):
        self.orig_fn = orig_fn
        self.lam = lam
        self.events = []          # eviction 시점마다 한 건
        self._sample_idx = None

    def set_sample(self, idx):
        self._sample_idx = idx

    def __call__(self, history, cache, prompt_len, n_think, lam=None, alpha=None):
        lam = self.lam if lam is None else lam

        kw = {}
        if alpha is not None:
            kw["alpha"] = alpha

        # 실제 사용되는 z (반환값은 이것을 그대로 돌려준다)
        z = self.orig_fn(history, cache, prompt_len, n_think, lam, **kw)

        try:
            imp = self.orig_fn(history, cache, prompt_len, n_think, 1.0, **kw)
            neg_red = self.orig_fn(history, cache, prompt_len, n_think, 0.0, **kw)

            z_a = np.asarray(z, dtype=np.float64)
            imp_a = np.asarray(imp, dtype=np.float64)
            red_a = -np.asarray(neg_red, dtype=np.float64)   # red = -(z at lam=0)

            if len(z_a) >= 3:
                term_imp = lam * imp_a
                term_red = (1.0 - lam) * red_a

                s_imp = float(term_imp.std())
                s_red = float(term_red.std())

                self.events.append(dict(
                    sample=self._sample_idx,
                    n_think=int(len(z_a)),
                    lam=float(lam),
                    # 원시 항 통계
                    imp_sum=float(imp_a.sum()),
                    imp_mean=float(imp_a.mean()),
                    imp_std=float(imp_a.std()),
                    imp_max=float(imp_a.max()),
                    imp_nonzero_frac=float((imp_a > 0).mean()),
                    red_sum=float(red_a.sum()),
                    red_mean=float(red_a.mean()),
                    red_std=float(red_a.std()),
                    # 가중 후 항 (z 순위에 실제로 기여하는 크기)
                    term_imp_std=s_imp,
                    term_red_std=s_red,
                    ratio_std=float(s_red / s_imp) if s_imp > 1e-15 else float("inf"),
                    # z 가 어느 항을 따라가는지
                    rho_z_imp=spearman(z_a, imp_a),
                    rho_z_red=spearman(z_a, -red_a),
                    rho_imp_red=spearman(imp_a, red_a),
                ))
        except Exception as e:                      # 진단 실패해도 생성은 계속
            print(f"  [probe warn] {type(e).__name__}: {e}", file=sys.stderr)

        return z


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로딩 — 리포 로더가 있으면 쓰고, 없으면 HF 로 폴백
# ─────────────────────────────────────────────────────────────────────────────

def load_math500(n: int):
    """(question, answer) 리스트 반환."""
    try:
        from src.datasets.math500 import load_math500 as _repo_loader   # noqa
        data = _repo_loader()
        out = []
        for d in data[:n]:
            q = d.get("problem") or d.get("question")
            out.append((q, d.get("answer")))
        print(f"[data] 리포 로더 사용, {len(out)}건")
        return out
    except Exception:
        pass

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = [(ds[i]["problem"], ds[i].get("answer")) for i in range(min(n, len(ds)))]
    print(f"[data] HuggingFaceH4/MATH-500 사용, {len(out)}건")
    return out


def build_prompt(tokenizer, question: str, enable_thinking: bool) -> str:
    msgs = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 리포트
# ─────────────────────────────────────────────────────────────────────────────

def report(events, lam):
    if not events:
        print("\n[!] eviction 이벤트가 한 건도 기록되지 않았다.")
        print("    budget 을 낮추거나 (--budget 512) 샘플 수를 늘려라.")
        return

    def col(k):
        v = [e[k] for e in events if not (isinstance(e[k], float) and np.isnan(e[k]))]
        return np.asarray(v, dtype=np.float64) if v else np.array([np.nan])

    def line(label, k, fmt="{:.6g}"):
        a = col(k)
        finite = a[np.isfinite(a)]
        if len(finite) == 0:
            print(f"  {label:<22s}  (전부 inf/nan)")
            return
        print(f"  {label:<22s}  중앙값={fmt.format(np.median(finite)):>12s}"
              f"  평균={fmt.format(finite.mean()):>12s}"
              f"  [{fmt.format(finite.min())}, {fmt.format(finite.max())}]")

    n_ev = len(events)
    n_samp = len(set(e["sample"] for e in events))
    print("\n" + "=" * 78)
    print(f"R-KV 스케일 진단  —  eviction 이벤트 {n_ev}건 / 샘플 {n_samp}개 / lam={lam}")
    print("=" * 78)

    print("\n[원시 항]")
    line("imp_sum", "imp_sum")
    print("    ^ 원본(candidate-only 재softmax)이면 1.0. 1보다 작으면 스케일 손실.")
    line("imp_std", "imp_std")
    line("imp_nonzero_frac", "imp_nonzero_frac")
    line("red_sum", "red_sum")
    print("    ^ softmax(dim=0) 이므로 1.0 이어야 정상.")
    line("red_std", "red_std")

    print("\n[가중 후 항 — z 순위에 실제 기여하는 크기]")
    line("std(lam*imp)", "term_imp_std")
    line("std((1-lam)*red)", "term_red_std")
    line("ratio_std", "ratio_std", "{:.3f}")
    print("    ^ (1-lam)*red 가 lam*imp 의 몇 배인가.")

    print("\n[z 가 따라가는 항 — Spearman]")
    line("rho(z, imp)", "rho_z_imp", "{:.4f}")
    line("rho(z, -red)", "rho_z_red", "{:.4f}")
    line("rho(imp, red)", "rho_imp_red", "{:.4f}")

    # 판정
    ratio = col("ratio_std")
    ratio = ratio[np.isfinite(ratio)]
    rho_red = col("rho_z_red")
    rho_red = rho_red[np.isfinite(rho_red)]
    imp_sum = col("imp_sum")
    imp_sum = imp_sum[np.isfinite(imp_sum)]

    med_ratio = float(np.median(ratio)) if len(ratio) else float("nan")
    med_rho = float(np.median(rho_red)) if len(rho_red) else float("nan")
    med_isum = float(np.median(imp_sum)) if len(imp_sum) else float("nan")

    print("\n" + "-" * 78)
    print("판정")
    print("-" * 78)
    print(f"  ratio_std 중앙값   = {med_ratio:.3f}")
    print(f"  rho(z,-red) 중앙값 = {med_rho:.4f}")
    print(f"  imp_sum 중앙값     = {med_isum:.6g}")
    print()

    if med_ratio > 10 and med_rho > 0.95:
        print("  >> REDUNDANCY-ONLY 퇴화 확정.")
        print("     importance 항이 z 순위에 사실상 기여하지 못한다.")
        print("     r_kv.py 의 max_pool1d 직후에 다음 한 줄을 추가하고 재실행:")
        print("         imp_think = imp_think / imp_think.sum().clamp(min=1e-9)")
        print("     현재 R-KV 결과는 원본 R-KV 가 아니라 redundancy-only ablation 이다.")
    elif med_ratio > 3 or med_rho > 0.85:
        print("  >> redundancy 우세하나 importance 도 일부 기여.")
        print("     lam sweep (0.07 / 0.3 / 0.5 / 0.7) 으로 민감도 확인 권장.")
    else:
        print("  >> 두 항 스케일 균형. R-KV 저성능은 스케일 버그가 원인이 아니다.")
        print("     이 결과 자체가 재구현 충실도의 근거가 되므로 논문 appendix 에 넣어라.")

    if med_isum < 0.5:
        print()
        print(f"  참고: imp_sum={med_isum:.4g} << 1.0 — think 구간 슬라이스로")
        print("        attention mass 가 손실되고 있다 (프롬프트가 가져감).")
    print()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--out", default="rkv_scale_diag.json")
    ap.add_argument("--repo-root", default=".",
                    help="src/ 가 있는 디렉토리")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo_root).resolve()))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import src.core.generation as G
    from src.methods.r_kv import RKV_LAMBDA

    lam = RKV_LAMBDA
    print(f"[cfg] model={args.model}  budget={args.budget}  "
          f"n_samples={args.n_samples}  lam={lam}")

    # ── monkey-patch: generation 네임스페이스에 바인딩된 이름을 교체 ──────────
    probe = RKVScaleProbe(G.compute_rkv_scores, lam)
    G.compute_rkv_scores = probe
    print("[patch] src.core.generation.compute_rkv_scores 교체 완료 "
          "(원본 파일은 수정하지 않음)")

    # ── 모델 로드 (rkv 는 eager 필수) ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    enable_thinking = "Qwen" in args.model

    data = load_math500(args.n_samples)

    for i, (question, _gold) in enumerate(data):
        probe.set_sample(i)
        prompt = build_prompt(tokenizer, question, enable_thinking)
        before = len(probe.events)
        print(f"[run] sample {i+1}/{len(data)} ...", end="", flush=True)
        try:
            out = G.generate_with_scored_eviction(
                model, tokenizer, prompt, "rkv", args.budget, device,
                max_new_tokens=args.max_new_tokens,
            )
            got = len(probe.events) - before
            print(f" think={out['think_tokens_generated']:6d} "
                  f"trunc={out['truncated']} eviction_events={got}")
        except Exception as e:
            print(f" 실패: {type(e).__name__}: {e}")

    report(probe.events, lam)

    with open(args.out, "w") as f:
        json.dump(dict(
            model=args.model, budget=args.budget, lam=lam,
            n_samples=args.n_samples, events=probe.events,
        ), f, indent=2)
    print(f"[out] 이벤트 원본 저장: {args.out}")


if __name__ == "__main__":
    main()