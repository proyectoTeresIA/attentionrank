import os
from pathlib import Path
from statistics import mean

def read_terms(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        terms = [line.strip().lower() for line in f if line.strip()]
    return terms

def precision_recall_f1_at_k(gold_terms, pred_terms, k):
    pred_k = pred_terms[:k]
    gold_set = set(gold_terms)
    pred_set = set(pred_k)

    tp = len(gold_set & pred_set)

    precision = tp / k if k > 0 else 0
    recall = tp / len(gold_set) if gold_set else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def evaluate(gold_dir, pred_dir, ks=(5,10,15)):
    gold_dir = Path(gold_dir)
    pred_dir = Path(pred_dir)

    results = {k: {"P": [], "R": [], "F1": []} for k in ks}

    files = sorted(os.listdir(gold_dir))

    for fname in files:
        gold_file = gold_dir / fname
        pred_file = pred_dir / fname

        if not pred_file.exists():
            print(f"Missing prediction for {fname}")
            continue

        gold_terms = read_terms(gold_file)
        pred_terms = read_terms(pred_file)

        for k in ks:
            p, r, f1 = precision_recall_f1_at_k(gold_terms, pred_terms, k)
            results[k]["P"].append(p)
            results[k]["R"].append(r)
            results[k]["F1"].append(f1)

    print("\n=== MACRO AVERAGE RESULTS ===")
    for k in ks:
        print(f"\n@{k}")
        print(f"P@{k}:  {mean(results[k]['P']):.4f}")
        print(f"R@{k}:  {mean(results[k]['R']):.4f}")
        print(f"F1@{k}: {mean(results[k]['F1']):.4f}")

    return results


if __name__ == "__main__":

    path="/Users/pablo/Downloads/Teresia/TeresIA-JURI_VersionFinal-AT/"
    evaluate(path+"keys", path+"res15")
