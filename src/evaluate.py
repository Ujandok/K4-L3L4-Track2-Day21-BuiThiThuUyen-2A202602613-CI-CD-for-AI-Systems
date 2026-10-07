"""
Bonus 3: bao cao chi tiet sau moi lan huan luyen.

Doc models/model.joblib va tap holdout, in confusion matrix dang van ban,
tinh precision / recall rieng cho tung lop va ghi ra outputs/detail.txt.
"""
import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

LABELS = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}


def evaluate(
    model_path: str = "models/model.joblib",
    eval_path: str = "data/holdout.csv",
    out_path: str = "outputs/detail.txt",
) -> str:
    model = joblib.load(model_path)
    df_eval = pd.read_csv(eval_path)
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]
    preds = model.predict(X_eval)

    (tn, fp), (fn, tp) = confusion_matrix(y_eval, preds, labels=[0, 1])

    lines = [
        f"Confusion matrix tren tap holdout ({len(df_eval)} mau)",
        "",
        f"{'':<22}{'du_doan_thap':>14}{'du_doan_cao':>14}",
        f"{'thuc_te_thap (0)':<22}{tn:>14}{fp:>14}",
        f"{'thuc_te_cao  (1)':<22}{fn:>14}{tp:>14}",
        "",
        f"{'Lop':<22}{'precision':>14}{'recall':>14}{'so_mau':>10}",
    ]
    for label, name in LABELS.items():
        precision = precision_score(y_eval, preds, pos_label=label, zero_division=0)
        recall = recall_score(y_eval, preds, pos_label=label, zero_division=0)
        support = int((y_eval == label).sum())
        lines.append(f"{f'{label} ({name})':<22}{precision:>14.4f}{recall:>14.4f}{support:>10}")

    lines += [
        "",
        f"Bo sot nguoi thu nhap cao (FN): {fn} / {fn + tp}",
        f"Gan nham nguoi thu nhap thap thanh cao (FP): {fp} / {fp + tn}",
    ]

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(text)
    return text


if __name__ == "__main__":
    evaluate()
