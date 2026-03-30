import csv
from collections import Counter


def analyze_signature_report(csv_path):
    """
    Analyze signature verification results from CSV.
    """

    total = 0
    correct = 0

    genuine_total = 0
    genuine_correct = 0
    forged_total = 0
    forged_correct = 0

    false_positive = 0  # Forged → predicted Genuine
    false_negative = 0  # Genuine → predicted Forged

    per_writer_errors = Counter()

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total += 1

            true_label = row["true_label"]
            predicted_label = row["predicted_label"]
            is_correct = row["correct"] == "True"
            writer_id = row["writer_id"]

            if is_correct:
                correct += 1

            if true_label == "GENUINE":
                genuine_total += 1
                if predicted_label == "GENUINE":
                    genuine_correct += 1
                else:
                    false_negative += 1
                    per_writer_errors[writer_id] += 1

            elif true_label == "FORGED":
                forged_total += 1
                if predicted_label == "FORGED":
                    forged_correct += 1
                else:
                    false_positive += 1
                    per_writer_errors[writer_id] += 1

    # ---- Metrics ----
    overall_accuracy = correct / total if total else 0
    genuine_accuracy = genuine_correct / genuine_total if genuine_total else 0
    forged_accuracy = forged_correct / forged_total if forged_total else 0
    fp_rate = false_positive / forged_total if forged_total else 0
    fn_rate = false_negative / genuine_total if genuine_total else 0

    # ---- Report ----
    print("\n========== SIGNATURE VERIFICATION REPORT ==========\n")
    print(f"Total samples           : {total}")
    print(f"Overall accuracy        : {overall_accuracy:.2%}")
    print()
    print(f"Genuine samples         : {genuine_total}")
    print(f"  ✓ Correct Genuine     : {genuine_correct} ({genuine_accuracy:.2%})")
    print(f"  ✗ False Negatives (FN): {false_negative} ({fn_rate:.2%})")
    print()
    print(f"Forged samples          : {forged_total}")
    print(f"  ✓ Correct Forged      : {forged_correct} ({forged_accuracy:.2%})")
    print(f"  ✗ False Positives (FP): {false_positive} ({fp_rate:.2%})")
    print("\n-----------------------------------------------")

    if per_writer_errors:
        print("\nWriters with most errors:")
        for writer, cnt in per_writer_errors.most_common(5):
            print(f"  Writer {writer}: {cnt} errors")

    print("\n===============================================\n")


if __name__ == "__main__":
    # Example usage
    analyze_signature_report(
        "artifacts/signature_test_report.csv"
    )
