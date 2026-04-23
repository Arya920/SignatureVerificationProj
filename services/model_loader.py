import torch
from Inference import load_model, load_embedding_db, load_threshold

def load_assets():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("artifacts/signature_embedder.pth", device)
    db = load_embedding_db("artifacts/embedding_db.pkl")
    threshold = load_threshold("artifacts/threshold.txt")
    return model, db, threshold, device
