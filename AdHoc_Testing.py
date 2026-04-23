# bulk_test.py
import os
import random
import csv

from Inference import (
    load_model,
    load_embedding_db,
    load_threshold,
    verify_signature
)


DEVICE = "cpu"

MODEL_PATH = "artifacts/signature_embedder.pth"
DB_PATH = "artifacts/embedding_db.pkl"
THRESHOLD_PATH = "artifacts/threshold.txt"

ORG_DIR = "data/split/test/org"
FORG_DIR = "data/split/test/forg"

OUTPUT_CSV = "artifacts/signature_test_report.csv"
MAX_SAMPLES_PER_WRITER = 5


# --------------------
# Utilities
# --------------------
def collect_samples(base_dir, max_samples):
    writer_map = {}

    for f in os.listdir(base_dir):
        if not f.endswith(".png"):
            continue
        writer = f.split("_")[1]
        writer_map.setdefault(writer, []).append(os.path.join(base_dir, f))

    sampled = {}
    for writer, files in writer_map.items():
        sampled[writer] = random.sample(
            files, min(len(files), max_samples)
        )

    return sampled


# --------------------
# Bulk Testing
# --------------------
def run_bulk_test():
    embed_model = load_model(MODEL_PATH, DEVICE)
    embedding_db = load_embedding_db(DB_PATH)
    threshold = load_threshold(THRESHOLD_PATH)

    org_samples = collect_samples(ORG_DIR, MAX_SAMPLES_PER_WRITER)
    forg_samples = collect_samples(FORG_DIR, MAX_SAMPLES_PER_WRITER)

    all_writers = sorted(set(org_samples) | set(forg_samples))
    results = []

    for writer in all_writers:
        for img in org_samples.get(writer, []):
            pred, dist = verify_signature(
                img, writer,
                embed_model, embedding_db,
                threshold, DEVICE
            )
            results.append([
                writer, img, "GENUINE", pred, dist, pred == "GENUINE"
            ])

        for img in forg_samples.get(writer, []):
            pred, dist = verify_signature(
                img, writer,
                embed_model, embedding_db,
                threshold, DEVICE
            )
            results.append([
                writer, img, "FORGED", pred, dist, pred == "FORGED"
            ])

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "writer_id",
            "image_path",
            "true_label",
            "predicted_label",
            "distance",
            "correct"
        ])
        writer.writerows(results)

    print(f"✅ Test report generated: {OUTPUT_CSV}")
    print(f"Total samples tested: {len(results)}")


# --------------------
# Entry Point
# --------------------
if __name__ == "__main__":
    run_bulk_test()
