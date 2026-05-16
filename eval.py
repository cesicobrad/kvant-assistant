import json
import re
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from app import build_rag, answer_question

QUESTIONS_FILE = "eval_questions.json"
RESULTS_FILE = "eval_results.json"

JUDGE_PROMPT = """You are evaluating a RAG system answer. Be generous — partial credit counts.

QUESTION: {question}
EXPECTED ANSWER: {expected}
ACTUAL ANSWER: {actual}
CATEGORY: {category}

Scoring:
- unanswerable: correct=true if the answer says it doesn't know or can't find it. correct=false if it invents facts.
- contradiction: correct=true if it mentions both values OR correctly identifies the deployed/final value.
- simple/multi_hop: correct=true if the answer contains the main fact from expected, even if incomplete or differently worded.

Respond with ONLY this JSON on one line:
{{"correct": true, "reasoning": "one sentence"}}
"""

def parse_judgment(text):
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    lower = text.lower()[:300]
    correct = '"correct": true' in lower or '"correct":true' in lower
    return {"correct": correct, "reasoning": "fallback parse: " + text[:120]}

def main():
    print("=== KVANT EVAL ===\n")
    questions = json.loads(Path(QUESTIONS_FILE).read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} questions.\n")

    vector_store, llm, prompt = build_rag()
    judge_llm = llm

    results = []
    for q in questions:
        print(f"[Q{q['id']}] ({q['category']}) {q['question']}")
        try:
            actual, docs = answer_question(q["question"], vector_store, llm, prompt, verbose=False)
        except Exception as e:
            actual = f"ERROR: {e}"
            docs = []

        judge_input = JUDGE_PROMPT.format(
            question=q["question"],
            expected=q["expected"],
            actual=actual,
            category=q["category"],
        )
        try:
            judgment_raw = judge_llm.invoke(judge_input)
        except Exception as e:
            judgment_raw = f'{{"correct": false, "reasoning": "judge error: {e}"}}'

        judgment = parse_judgment(judgment_raw)
        sources_used = list({d.metadata.get("source", "?") for d in docs})

        result = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected": q["expected"],
            "actual": actual,
            "sources_used": sources_used,
            "correct": judgment.get("correct", False),
            "reasoning": judgment.get("reasoning", ""),
        }
        results.append(result)
        mark = "✓" if result["correct"] else "✗"
        print(f"  {mark} {result['reasoning']}\n")

    print("\n=== SUMMARY ===")
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["correct"])
    for cat, vals in by_cat.items():
        pct = 100 * sum(vals) / len(vals)
        print(f"  {cat:15s}  {sum(vals)}/{len(vals)}  ({pct:.0f}%)")
    overall = sum(r["correct"] for r in results)
    print(f"  {'OVERALL':15s}  {overall}/{len(results)}  ({100*overall/len(results):.0f}%)")

    Path(RESULTS_FILE).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()