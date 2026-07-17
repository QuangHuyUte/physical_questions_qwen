import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "result")
import physics_pipeline as p

VERIFIED = Path(r"Retrieve new data v2\verified_golden_expanded.csv")

FILES = [
    Path(r"result\physics_holdout_pipeline_eval_llm_fallback_ready.csv"),
    Path(r"result\physics_stress_test_eval_llm_fallback_ready.csv"),
]

p.prepare_pipeline(str(VERIFIED))

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

for path in FILES:
    print("\n" + "=" * 80)
    print(path)

    df = pd.read_csv(path, dtype=str).fillna("")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    q_col = pick_col(df, ["question", "Question", "prompt"])
    a_col = pick_col(df, ["answer", "true_answer", "expected_answer", "gold_answer"])
    u_col = pick_col(df, ["unit", "true_unit", "expected_unit", "gold_unit"])

    if q_col is None or a_col is None:
        print("SKIP: không tìm thấy cột question/answer phù hợp.")
        continue

    rows = []
    for _, row in df.iterrows():
        question = row[q_col]
        true_answer = row[a_col]
        true_unit = row[u_col] if u_col else ""

        out = p.solve_physics_question(question, true_answer, true_unit)
        is_correct = p.compare_answer(out["answer"], out["unit"], true_answer, true_unit)

        rows.append({
            "id": row.get("id", ""),
            "topic": row.get("topic", out.get("topic_pred", "")),
            "question": question,
            "true_answer": true_answer,
            "true_unit": true_unit,
            "pred_answer": out["answer"],
            "pred_unit": out["unit"],
            "method": out["method"],
            "topic_pred": out.get("topic_pred", ""),
            "prefix_pred": out.get("prefix_pred", ""),
            "confidence": out.get("confidence", ""),
            "is_correct": is_correct,
            "attempted": out["method"] != "unanswered_no_fallback",
        })

    res = pd.DataFrame(rows)

    out_path = path.with_name(path.stem + "_v9_eval.csv")
    res.to_csv(out_path, index=False)

    attempted = res["attempted"].mean()
    correct = res["is_correct"].mean()
    attempted_acc = res.loc[res["attempted"], "is_correct"].mean() if res["attempted"].any() else 0

    print("Attempted:", int(res["attempted"].sum()), "/", len(res), f"= {attempted:.3%}")
    print("Correct:", int(res["is_correct"].sum()), "/", len(res), f"= {correct:.3%}")
    print("Attempted accuracy:", f"{attempted_acc:.3%}")
    print("Saved:", out_path)

    print("\nWrong by topic:")
    wrong = res[res["attempted"] & ~res["is_correct"]]
    if len(wrong):
        print(wrong.groupby(["topic", "method"]).size().reset_index(name="count").sort_values("count", ascending=False).head(20).to_string(index=False))
    else:
        print("No wrong attempted rows.")

    print("\nUnanswered by topic:")
    unanswered = res[~res["attempted"]]
    if len(unanswered):
        print(unanswered.groupby(["topic"]).size().reset_index(name="count").sort_values("count", ascending=False).head(20).to_string(index=False))
    else:
        print("No unanswered rows.")