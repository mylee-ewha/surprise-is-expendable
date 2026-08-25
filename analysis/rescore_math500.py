#!/usr/bin/env python3
"""
MATH500 재채점 스크립트
=======================

목적
----
results_math500_ablation/results_texts.jsonl 에 저장된 생성 텍스트를
개선된 extractor 로 재채점하고, 원본 correct 필드와 비교한다.

GPU 불필요. 텍스트 파일 + gold dataset 만 있으면 된다.

원본 extractor 의 알려진 문제
------------------------------
1. 중간에 등장한 \boxed{} 를 최종 답으로 픽업 (idx=35 등)
2. 수식 표현 순서 차이로 동치인데 다르게 판정 (idx=52: x^8+...+1 vs 1+x+...+x^8)
3. LaTeX 서수 suffix 불일치 (idx=496: \boxed{12} vs \boxed{12^{\text{th}}})
4. ellipse 처럼 텍스트로 쓴 답의 불일치 처리 (idx=59 반복 오류)
5. "52_8" 같은 표기 정규화 누락 (idx=264)

개선 사항
----------
- 항상 마지막 \boxed{} 사용
- LaTeX 서수 제거: 12^{\\text{th}} -> 12
- 다항식 정규화: 항 정렬 후 비교
- 수치 정규화: 1.0 == 1, 정수 문자열 동일 처리

사용법
------
  # 리포 루트에서 (results_texts.jsonl 경로 직접 지정)
  python rescore_math500.py \\
      --texts results/math500_ablation/results_texts.jsonl \\
      --out   rescore_math500_out.json

  # gold 소스 선택 (기본: HuggingFace 다운로드)
  # 리포에 이미 데이터가 있으면:
  python rescore_math500.py \\
      --texts results/math500_ablation/results_texts.jsonl \\
      --gold-src repo           # src/datasets/math500.py 의 로더 사용

결과 파일
---------
rescore_math500_out.json:
  per_sample: idx, method, budget, old_correct, new_correct, old_pred, new_pred, gold
  summary   : method × budget 별 old_acc, new_acc, delta, n_flipped
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 개선된 extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_last_boxed(text: str):
    """마지막 \\boxed{...} 내용 반환 (중첩 브레이스 지원)."""
    idxs = [m.start() for m in re.finditer(r'\\boxed\{', text)]
    if not idxs:
        return None
    start = idxs[-1] + len(r'\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


_ORDINAL_RE = re.compile(r'\^\{(?:\\text|\\mathrm)\{(?:st|nd|rd|th)\}\}')
_SPACE_RE   = re.compile(r'\s+')


def normalize(s):
    if s is None: return None
    s = re.sub(r'\\dfrac', r'\\frac', s)       # dfrac → frac
    s = re.sub(r',\s+', ',', s)                # (2, 4) → (2,4)
    s = re.sub(r'\s*\+\s*', '+', s)            # x + 3 → x+3
    s = re.sub(r'\s*-\s*', '-', s)             # x - 3 → x-3 (주의: 음수 케이스)
    s = re.sub(r'\s+(?=[a-zA-Z\\{(])', '', s)  # 3 \sqrt → 3\sqrt
    s = _ORDINAL_RE.sub('', s)
    s = s.strip()
    return s


def _poly_key(s: str) -> str:
    """다항식 항 정렬 (순서만 다른 동치 표현 통일). 비다항식에는 원본 반환."""
    # x^8 + x^7 + ... 형태만 처리
    if re.fullmatch(r'[\w\^\+\s\{\}\\]+', s):
        terms = sorted(t.strip() for t in s.split('+'))
        return '+'.join(terms)
    return s


def equal_answers(pred: str | None, gold: str | None) -> bool:
    """정규화 후 두 답이 같은지 판정."""
    if pred is None or gold is None:
        return False
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    # 수치 비교: "1.0" == "1"
    try:
        if abs(float(p) - float(g)) < 1e-9:
            return True
    except ValueError:
        pass
    # 다항식 항 순서 무관 비교
    if _poly_key(p) == _poly_key(g):
        return True
    # 대소문자 무관 (Ellipse / ellipse)
    if p.lower() == g.lower():
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Gold 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_gold_hf() -> dict[int, str]:
    """HuggingFace 에서 MATH-500 gold 로드."""
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    return {i: str(ds[i]['answer']) for i in range(len(ds))}


def load_gold_repo() -> dict[int, str]:
    """리포의 src/datasets/math500.py 로더 사용."""
    from src.datasets.math500 import load_fixed_subset as load_math500
    data = load_math500()
    out = {}
    for i, d in enumerate(data):
        ans = d.get('answer') or d.get('solution') or ''
        # solution 에 \boxed{} 가 있으면 추출
        boxed = extract_last_boxed(str(ans))
        out[i] = normalize(boxed or ans) or str(ans)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 재채점
# ─────────────────────────────────────────────────────────────────────────────

def rescore(texts_path: str, gold: dict[int, str]) -> list[dict]:
    results = []
    with open(texts_path) as f:
        for line in f:
            r = json.loads(line)
            if r['method'] == 'baseline':
                continue   # baseline 은 재채점 대상에서 제외 (optional: 포함 가능)

            idx = r['idx']
            g = gold.get(idx)

            new_pred  = normalize(extract_last_boxed(r['text']))
            new_corr  = equal_answers(new_pred, g)
            old_corr  = r['correct']

            results.append(dict(
                idx=idx,
                method=r['method'],
                budget=r.get('budget'),
                old_correct=old_corr,
                new_correct=new_corr,
                old_pred=r.get('pred'),        # 원본 extractor 결과 (없을 수 있음)
                new_pred=new_pred,
                gold=g,
                truncated=r.get('truncated', False),
            ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 요약
# ─────────────────────────────────────────────────────────────────────────────

def summarize(results: list[dict]) -> list[dict]:
    from collections import Counter

    groups = defaultdict(list)
    for r in results:
        groups[(r['method'], r['budget'])].append(r)

    summary = []
    for (method, budget), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        n = len(rows)
        old_acc = sum(r['old_correct'] for r in rows) / n
        new_acc = sum(r['new_correct'] for r in rows) / n
        # 뒤집힌 케이스
        old_t_new_f = [r for r in rows if r['old_correct'] and not r['new_correct']]
        old_f_new_t = [r for r in rows if not r['old_correct'] and r['new_correct']]
        summary.append(dict(
            method=method, budget=budget, n=n,
            old_acc=round(old_acc, 4), new_acc=round(new_acc, 4),
            delta=round(new_acc - old_acc, 4),
            n_old_t_new_f=len(old_t_new_f),
            n_old_f_new_t=len(old_f_new_t),
            flip_indices_old_t_new_f=[r['idx'] for r in old_t_new_f],
            flip_indices_old_f_new_t=[r['idx'] for r in old_f_new_t],
        ))
    return summary


def print_summary(summary: list[dict]) -> None:
    print()
    print("=" * 90)
    print(f"{'method':<16} {'budget':>7}  {'old':>6}  {'new':>6}  {'delta':>7}  "
          f"{'↑(새 정답)':>9}  {'↓(새 오답)':>9}")
    print("-" * 90)
    for s in summary:
        flag = ''
        if abs(s['delta']) >= 0.002:
            flag = ' ◀'
        print(f"{s['method']:<16} {str(s['budget'] or 'base'):>7}  "
              f"{s['old_acc']:>6.4f}  {s['new_acc']:>6.4f}  "
              f"{s['delta']:>+7.4f}  "
              f"{s['n_old_f_new_t']:>9}  {s['n_old_t_new_f']:>9}{flag}")
    print("=" * 90)

    # ni vs LRU 비교 (논문 핵심)
    print()
    print("── ni vs LRU @ 각 budget (new_acc 기준) ──")
    ni_d  = {s['budget']: s for s in summary if s['method'] == 'novelty_inv'}
    lru_d = {s['budget']: s for s in summary if s['method'] == 'lru'}
    for B in sorted(set(ni_d) | set(lru_d), key=lambda x: x or 0):
        ni_  = ni_d.get(B)
        lru_ = lru_d.get(B)
        if ni_ and lru_:
            gap = ni_['new_acc'] - lru_['new_acc']
            print(f"  B={str(B):>5}  ni={ni_['new_acc']:.4f}  lru={lru_['new_acc']:.4f}  "
                  f"gap={gap:+.4f}  {'ni 우세' if gap > 0.002 else 'lru 우세' if gap < -0.002 else '동률'}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--texts',    required=True,
                    help='results_texts.jsonl 경로')
    ap.add_argument('--out',      default='rescore_math500_out.json')
    ap.add_argument('--gold-src', default='hf', choices=['hf', 'repo'],
                    help='hf=HuggingFace 다운로드 / repo=src/datasets/math500.py')
    ap.add_argument('--repo-root', default='.',
                    help='src/ 가 있는 디렉토리 (--gold-src repo 시)')
    ap.add_argument('--show-flips', action='store_true',
                    help='뒤집힌 케이스 상세 출력')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo_root).resolve()))

    print(f"[gold] 소스: {args.gold_src}")
    if args.gold_src == 'hf':
        gold = load_gold_hf()
    else:
        gold = load_gold_repo()
    print(f"[gold] {len(gold)}건 로드")

    print(f"[texts] {args.texts}")
    results = rescore(args.texts, gold)
    print(f"[rescore] {len(results)}건 처리")

    summary = summarize(results)
    print_summary(summary)

    if args.show_flips:
        print()
        print("── old=False → new=True (새로 정답 처리된 것) ──")
        for r in results:
            if not r['old_correct'] and r['new_correct']:
                print(f"  idx={r['idx']:3d}  {r['method']:<14} B={str(r['budget']):>5}"
                      f"  old_pred={repr(r['old_pred']):<20}"
                      f"  new_pred={repr(r['new_pred']):<20}"
                      f"  gold={repr(r['gold'])}")
        print()
        print("── old=True → new=False (새로 오답 처리된 것) ──")
        for r in results:
            if r['old_correct'] and not r['new_correct']:
                print(f"  idx={r['idx']:3d}  {r['method']:<14} B={str(r['budget']):>5}"
                      f"  old_pred={repr(r['old_pred']):<20}"
                      f"  new_pred={repr(r['new_pred']):<20}"
                      f"  gold={repr(r['gold'])}")

    out = dict(summary=summary, per_sample=results)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == '__main__':
    main()