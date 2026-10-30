"""Step 7: evaluation. Regex first (cheap, deterministic), LLM judge second.

Usage:
  python eval.py --n 50              # regex eval on 50 test questions
  python eval.py --n 50 --judge      # add an LLM-as-judge score
  python eval.py --n 50 --retrieval  # retrieval hit-rate only, no API key needed
"""
import argparse
import json
import random
import re

from datasets import load_dataset

import config
import rag

JUDGE_PROMPT = """You grade a RAG answer. Reply with JSON only:
{{"correct": true|false, "grounded": true|false, "reason": "..."}}
correct  = the answer agrees with the reference answer.
grounded = every claim is supported by the cited passages.

Question: {q}
Reference answer: {ref}
Candidate answer: {a}
Cited passages:
{ctx}"""


def load_tests(n: int, seed: int = 42, skip_yes_no: bool = False):
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
    rows = list(ds)
    if skip_yes_no:  # "yes"/"no" never appears verbatim in a passage
        rows = [r for r in rows if r["answer"].strip().lower().rstrip(".") not in ("yes", "no")]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def regex_for(expected: str) -> str:
    # whole-word, case-insensitive match of the reference answer
    return r"\b" + re.escape(expected.strip().rstrip(".")) + r"\b"


def judge(q, ref, a, hits) -> dict:
    ctx = "\n".join(f"[{i}] {h['text'][:800]}" for i, h in enumerate(hits, 1))
    resp = rag._llm().chat.completions.create(
        model=config.LLM_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, a=a, ctx=ctx)}],
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"correct": False, "grounded": False, "reason": "judge returned bad JSON"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--retrieval", action="store_true",
                    help="only check whether the answer text appears in the top-k chunks")
    args = ap.parse_args()

    tests = load_tests(args.n, skip_yes_no=args.retrieval)
    passed = judged_ok = 0
    for t in tests:
        q, ref = t["question"], t["answer"]
        if args.retrieval:
            hits = rag.retrieve(q)
            ok = any(re.search(regex_for(ref), h["text"], re.I) for h in hits)
            passed += ok
            continue
        out = rag.answer(q)
        a = out["answer"]
        cited = re.search(r"\[\d+\]", a)
        correct = re.search(regex_for(ref), a, re.I)
        ok = bool(cited and correct)
        passed += ok
        line = f"{'PASS' if ok else 'FAIL'} | {q[:60]:60} | ref={ref[:20]!r}"
        if args.judge and out["sources"]:
            j = judge(q, ref, a, out["sources"])
            judged_ok += bool(j.get("correct") and j.get("grounded"))
            line += f" | judge={j.get('correct')}/{j.get('grounded')}"
        print(line)

    label = "retrieval hit rate" if args.retrieval else "pass rate"
    print(f"\n{label}: {passed / len(tests):.0%}  ({passed}/{len(tests)})")
    if args.judge:
        print(f"judge pass rate (correct AND grounded): {judged_ok / len(tests):.0%}")


if __name__ == "__main__":
    main()
