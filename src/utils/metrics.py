import re

def extract_boxed_answer(text: str):
    starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return None
    start = starts[-1] + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start:i - 1].strip()


def normalize_answer(s):
    if s is None:
        return None
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\!|\\,|\\;|\\:|\\ ", "", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace(" ", "")
    s = s.rstrip(".")
    return s

def is_correct(pred, gold) -> bool:
    pred_n, gold_n = normalize_answer(pred), normalize_answer(str(gold))
    return pred_n is not None and pred_n == gold_n


def extract_mcq_answer(text: str):
    if "</think>" in text:
        answer_part = text.split("</think>", 1)[-1]
    else:
        answer_part = text[-300:]

    patterns = [
        r"[Tt]he correct answer is\s*\(?([ABCD])\)?",
        r"[Tt]he answer is\s*\(?([ABCD])\)?",
        r"[Aa]nswer[:\s]+\(?([ABCD])\)?",
        r"\*\*\(?([ABCD])\)?\*\*",
        r"^\s*\(?([ABCD])\)[\.\s]",
    ]
    for pat in patterns:
        m = re.search(pat, answer_part, re.MULTILINE)
        if m:
            return m.group(1)

    return None


def is_correct_mcq(pred, gold) -> bool:
    return pred is not None and pred.upper() == gold.upper()
