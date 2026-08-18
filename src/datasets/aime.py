"""
AIME 2024 I+II / 2025 I+II — 60문제 로더
=========================================
Reference:
  - https://huggingface.co/datasets/HuggingFaceH4/aime_2024
  - https://huggingface.co/datasets/opencompass/AIME2025
"""
import re
from pathlib import Path
from datasets import load_dataset

DEFAULT_PATH = Path("data/aime_2024_2025.jsonl")


def _parse_answer(raw: str) -> int | None:
    """'336^\\circ', '  42  ' 등에서 정수 추출. 실패 시 None."""
    raw = str(raw).strip()
    m = re.search(r'\d+', raw)
    if m:
        val = int(m.group())
        if 0 <= val <= 999:
            return val
    return None


def load_aime() -> list[dict]:
    ds_2024    = load_dataset("HuggingFaceH4/aime_2024",  split="train")
    ds_2025_I  = load_dataset("opencompass/AIME2025", "AIME2025-I",  split="test")
    ds_2025_II = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")

    problems = []

    for ex in ds_2024:                          # 컬럼: problem / answer
        ans = _parse_answer(ex["answer"])
        if ans is None:
            print(f"[skip 2024] unparseable: {ex['answer']!r}")
            continue
        problems.append({"problem": ex["problem"], "answer": ans})

    for name, ds in [("2025-I", ds_2025_I), ("2025-II", ds_2025_II)]:
        for ex in ds:
            ans = _parse_answer(ex["answer"])
            if ans is None:
                print(f"[skip {name}] unparseable: {ex['answer']!r}")
                continue
            problems.append({"problem": ex["question"], "answer": ans})

    print(f"[AIME] 총 {len(problems)}문제 로드")
    assert len(problems) >= 58, f"Too few problems: {len(problems)}"
    return problems


def build_prompt(tokenizer, question: str, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": question}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(messages, **kwargs)