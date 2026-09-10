"""
analysis/detector_calib.py — n-gram loop detector 오프라인 캘리브레이션
=====================================================================
GPU 불필요. 저장된 novelty_inv 생성 텍스트를 재생해서
(n, window, enter_threshold) 조합을 전수 스윕한다.

핵심 최적화:
  - rep_rate를 incremental 계산 (naive 대비 12x)
  - (sample, n, window)당 rep 시계열을 1회만 계산 후
    threshold 스윕에서 재사용 (추가 7x)

Usage:
  python analysis/detector_calib.py                 # 전체
  python analysis/detector_calib.py --quick         # 스윕 범위 축소
  python analysis/detector_calib.py --counts-only   # 라벨 분포만 확인
"""
import sys, json, argparse, itertools
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from transformers import AutoTokenizer

QWEN     = "Qwen/Qwen3-8B"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# ── 데이터셋 정의: (이름, 텍스트 경로, 토크나이저, budget 목록) ──────────
DATASETS = [
    ("math500_qwen3",    "results/math500_ablation/results_texts.jsonl",
     QWEN,     [512, 1024, 2048, 4096]),
    ("math500_deepseek", "results/math500_ablation_deepseek/results_texts.jsonl",
     DEEPSEEK, [512, 1024, 2048, 4096]),
    ("gpqa_qwen3",       "results/gpqa_ablation/results_texts.jsonl",
     QWEN,     [512, 1024, 2048, 4096]),
    ("gpqa_deepseek",    "results/gpqa_ablation_deepseek/results_texts.jsonl",
     DEEPSEEK, [512, 1024, 2048, 4096]),
    ("aime_qwen3",       "results/aime_ablation/results_texts.jsonl",
     QWEN,     [2048, 4096, 8192, 16384]),
    ("aime_deepseek",    "results/aime_ablation_deepseek/results_texts.jsonl",
     DEEPSEEK, [2048, 4096, 8192, 16384]),
]

# ── 스윕 범위 ──────────────────────────────────────────────────────────
NGRAMS  = [3, 4, 5, 8]
WINDOWS = [200, 400, 800, 1600, 3200]
ENTERS  = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
EXIT_RATIO = 0.4        # exit_threshold = enter * EXIT_RATIO
MIN_CONFIRM = 32

# circular 라벨 기준: truncated 이면서 전체 8-gram(단어단위) 반복률이 이 값 초과
CIRC_REP8_MIN = 0.15
MIN_SAMPLES_PER_CLASS = 5   # 이보다 적으면 해당 budget 스윕 생략

# ── 선정 기준 ──────────────────────────────────────────────────────────
DUTY_NORM_MAX = 0.05   # 정상 샘플에서 in_loop 상태 비율 상한 (헛개입 제약)
DUTY_PENALTY  = 2.0    # score = youden - DUTY_PENALTY * duty_norm


# ═══════════════════════════════════════════════════════════════════════
# rep_rate
# ═══════════════════════════════════════════════════════════════════════
def word_rep_rate(text, n=8):
    """라벨링용: 전체 텍스트의 단어 단위 n-gram 반복률."""
    w = text.split()
    if len(w) < n:
        return 0.0
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(g)) / len(g)


def rep_series(ids, n, window):
    """
    각 스텝에서의 sliding-window n-gram 반복률 시계열.
    LoopDetector.rep_rate()와 정확히 동일한 값을 incremental 하게 계산.
    """
    T = len(ids)
    out = np.zeros(T, dtype=np.float32)
    cnt = Counter()
    distinct = 0
    start = 0                      # 윈도우 시작 인덱스 (ids 상의)
    for k in range(T):
        end = k + 1                # 윈도우 = ids[start:end]
        if end - start > window:
            # 가장 오래된 n-gram 제거
            if end - start - 1 >= n:
                old = tuple(ids[start:start + n])
                cnt[old] -= 1
                if cnt[old] == 0:
                    distinct -= 1
                    del cnt[old]
            start += 1
        size = end - start
        if size >= n:
            new = tuple(ids[end - n:end])
            if cnt[new] == 0:
                distinct += 1
            cnt[new] += 1
            out[k] = 1.0 - distinct / (size - n + 1)
    return out


def simulate(series, enter, exit_thr, min_confirm=MIN_CONFIRM):
    """
    LoopDetector 상태 기계를 시계열에 재생.
    Returns: (발동 여부, 첫 발동 스텝, loop 상태였던 스텝 수)
    """
    in_loop, confirm = False, 0
    first_fire, loop_steps = -1, 0
    for i, rep in enumerate(series):
        thr = enter if not in_loop else exit_thr
        hit = (rep > thr) if not in_loop else (rep < thr)
        if hit:
            confirm += 1
            if confirm >= min_confirm:
                in_loop = not in_loop
                confirm = 0
                if in_loop and first_fire < 0:
                    first_fire = i
        else:
            confirm = 0
        if in_loop:
            loop_steps += 1
    return first_fire >= 0, first_fire, loop_steps


# ═══════════════════════════════════════════════════════════════════════
# 데이터 로딩
# ═══════════════════════════════════════════════════════════════════════
def load_samples(path, tok, budget, method="novelty_inv"):
    """해당 budget의 샘플을 circ/norm 으로 라벨링 + 토큰화."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("method") != method or r.get("budget") != budget:
                continue
            text = r.get("text", "")
            if not text:
                continue
            if r.get("truncated"):
                label = "circ" if word_rep_rate(text) > CIRC_REP8_MIN else None
            else:
                label = "norm"
            if label is None:
                continue
            rows.append({
                "idx":   r.get("idx"),
                "label": label,
                "ids":   tok.encode(text, add_special_tokens=False),
            })
    return rows


def split_train_test(rows):
    """idx 짝/홀로 분할 — 튜닝셋과 검증셋 분리."""
    train = [r for r in rows if (r["idx"] or 0) % 2 == 0]
    test  = [r for r in rows if (r["idx"] or 0) % 2 == 1]
    return train, test


# ═══════════════════════════════════════════════════════════════════════
# 스윕
# ═══════════════════════════════════════════════════════════════════════
def sweep(rows, ngrams, windows, enters):
    """
    Returns: list of dict(n, window, enter, tpr, fpr, youden,
                          duty_circ, duty_norm, score)
    duty_norm = 정상 샘플에서 in_loop 상태로 보낸 스텝 비율 (헛개입 시간).
    FPR은 '한 번이라도 켜졌나'만 보므로, 얼마나 오래 켜져 있었는지를
    duty로 따로 잰다. window가 클수록 duty가 부풀 수 있어 반드시 확인.
    """
    n_circ = sum(1 for r in rows if r["label"] == "circ")
    n_norm = sum(1 for r in rows if r["label"] == "norm")
    results = []

    for n, w in itertools.product(ngrams, windows):
        series_by_row = [(r["label"], rep_series(r["ids"], n, w)) for r in rows]
        for e in enters:
            ex = e * EXIT_RATIO
            fc = fn_ = 0
            dc, dn = [], []
            for label, s in series_by_row:
                fired, _, steps = simulate(s, e, ex)
                duty = steps / max(len(s), 1)
                if label == "circ":
                    fc += int(fired); dc.append(duty)
                else:
                    fn_ += int(fired); dn.append(duty)
            tpr = fc / max(n_circ, 1)
            fpr = fn_ / max(n_norm, 1)
            duty_circ = float(np.mean(dc)) if dc else 0.0
            duty_norm = float(np.mean(dn)) if dn else 0.0
            youden = tpr - fpr
            results.append({
                "n": n, "window": w, "enter": e,
                "tpr": tpr, "fpr": fpr, "youden": youden,
                "duty_circ": duty_circ, "duty_norm": duty_norm,
                "score": youden - DUTY_PENALTY * duty_norm,
            })
    return results


def print_table(results, top=12, title="", key="score"):
    if title:
        print(f"  {title}")
    print(f"    {'n':<4}{'win':<7}{'enter':<7}{'TPR':<7}{'FPR':<7}"
          f"{'T-F':<7}{'dutyC':<8}{'dutyN':<8}{'score':<7}")
    print("    " + "-" * 61)
    for r in sorted(results, key=lambda x: -x[key])[:top]:
        print(f"    {r['n']:<4}{r['window']:<7}{r['enter']:<7.2f}"
              f"{r['tpr']:<7.3f}{r['fpr']:<7.3f}{r['youden']:<7.3f}"
              f"{r['duty_circ']:<8.3f}{r['duty_norm']:<8.3f}{r['score']:<7.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="스윕 범위 축소 (n=[4,8], window=[200,400])")
    ap.add_argument("--counts-only", action="store_true",
                    help="라벨 분포만 출력하고 종료")
    ap.add_argument("--no-split", action="store_true",
                    help="train/test 분할 없이 전체로 스윕")
    args = ap.parse_args()

    ngrams  = [4, 8] if args.quick else NGRAMS
    windows = [200, 400] if args.quick else WINDOWS
    enters  = ENTERS

    tok_cache = {}
    all_results = {}        # (dataset, budget) → train sweep 결과

    for name, path, model, budgets in DATASETS:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {name}: {path} 없음")
            continue

        if model not in tok_cache:
            tok_cache[model] = AutoTokenizer.from_pretrained(model)
        tok = tok_cache[model]

        print(f"\n{'=' * 68}")
        print(f"{name}   ({model.split('/')[-1]})")
        print("=" * 68)

        for budget in budgets:
            rows = load_samples(p, tok, budget)
            n_c = sum(1 for r in rows if r["label"] == "circ")
            n_n = sum(1 for r in rows if r["label"] == "norm")
            if n_c == 0 and n_n == 0:
                continue

            print(f"\n  budget={budget}:  circular={n_c}, normal={n_n}", end="")
            if min(n_c, n_n) < MIN_SAMPLES_PER_CLASS:
                print(f"   → 샘플 부족(<{MIN_SAMPLES_PER_CLASS}), 스윕 생략")
                continue
            print()

            if args.counts_only:
                continue

            if args.no_split:
                res = sweep(rows, ngrams, windows, enters)
                print_table(res, title="(전체)")
                all_results[(name, budget)] = res
            else:
                train, test = split_train_test(rows)
                tc = sum(1 for r in train if r["label"] == "circ")
                tn = sum(1 for r in train if r["label"] == "norm")
                vc = sum(1 for r in test  if r["label"] == "circ")
                vn = sum(1 for r in test  if r["label"] == "norm")
                if min(tc, tn, vc, vn) < 3:
                    print("    → 분할 후 샘플 부족, 전체로 스윕")
                    res = sweep(rows, ngrams, windows, enters)
                    print_table(res, title="(전체)")
                    all_results[(name, budget)] = res
                    continue

                res_tr = sweep(train, ngrams, windows, enters)
                print_table(res_tr, top=8,
                            title=f"train (circ={tc}, norm={tn})")

                # duty_norm 제약을 만족하는 것 중 youden 최대
                feasible = [r for r in res_tr if r["duty_norm"] <= DUTY_NORM_MAX]
                if feasible:
                    best = max(feasible, key=lambda x: x["youden"])
                else:
                    best = max(res_tr, key=lambda x: x["score"])
                    print(f"    [warn] duty_norm<={DUTY_NORM_MAX} 만족 설정 없음")

                res_te = sweep(test, [best["n"]], [best["window"]],
                               [best["enter"]])[0]
                print(f"    → train 최적 {best['n']}-gram/win{best['window']}"
                      f"/enter{best['enter']:.2f} 을 test 적용: "
                      f"TPR={res_te['tpr']:.3f} FPR={res_te['fpr']:.3f} "
                      f"dutyN={res_te['duty_norm']:.3f} "
                      f"(train TPR={best['tpr']:.3f} FPR={best['fpr']:.3f} "
                      f"dutyN={best['duty_norm']:.3f})")
                all_results[(name, budget)] = res_tr

    if args.counts_only or not all_results:
        return

    # ── 전 조건 공통 최적값 ────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print("전 조건 통합 (평균 youden / 최악 youden / 평균 duty_norm)")
    print("=" * 68)
    agg = {}
    for res in all_results.values():
        for r in res:
            k = (r["n"], r["window"], r["enter"])
            agg.setdefault(k, {"y": [], "dn": []})
            agg[k]["y"].append(r["youden"])
            agg[k]["dn"].append(r["duty_norm"])

    rows_agg = [
        (k, float(np.mean(v["y"])), float(np.min(v["y"])),
         float(np.mean(v["dn"])), float(np.max(v["dn"])))
        for k, v in agg.items()
    ]
    print(f"  조건 수 = {len(all_results)}")

    # 제약 만족 집합
    feasible = [r for r in rows_agg if r[4] <= DUTY_NORM_MAX]
    pool, tag = (feasible, f"duty_norm(최악) <= {DUTY_NORM_MAX}") if feasible \
                else (rows_agg, "제약 만족 없음 — 전체")
    pool.sort(key=lambda x: -x[1])

    print(f"  [{tag}] 후보 {len(pool)}개, 평균 youden 내림차순\n")
    print(f"    {'n':<4}{'win':<7}{'enter':<7}{'평균Y':<9}{'최악Y':<9}"
          f"{'평균dN':<9}{'최악dN':<9}")
    print("    " + "-" * 54)
    for (n, w, e), my, miny, mdn, maxdn in pool[:12]:
        print(f"    {n:<4}{w:<7}{e:<7.2f}{my:<9.3f}{miny:<9.3f}"
              f"{mdn:<9.3f}{maxdn:<9.3f}")

    (bn, bw, be), bmy, bminy, bmdn, bmaxdn = pool[0]
    print(f"\n  권장 설정: {bn}-gram, window={bw}, enter={be:.2f}, "
          f"exit={be * EXIT_RATIO:.2f}")
    print(f"    평균 youden={bmy:.3f} (최악 {bminy:.3f}), "
          f"평균 duty_norm={bmdn:.3f} (최악 {bmaxdn:.3f})")

    # 경계 경고
    if bw == max(WINDOWS) or be == max(ENTERS) or be == min(ENTERS):
        print(f"\n  [warn] 최적값이 그리드 경계에 있음 "
              f"(window 범위 {min(WINDOWS)}~{max(WINDOWS)}, "
              f"enter {min(ENTERS)}~{max(ENTERS)}) — 범위 확장 검토 필요")


if __name__ == "__main__":
    main()