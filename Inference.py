# inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pickle
from torchvision.models import resnet18
from typing import Tuple


# --------------------
# Model Definition
# --------------------
class WriterNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Linear(512, 256)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)


# --------------------
# Preprocessing
# --------------------
def preprocess_image(path: str, device: str):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)
    crop = th[y:y+h, x:x+w]

    crop = cv2.resize(crop, (224, 224)) / 255.0
    tensor = torch.tensor(crop).float().unsqueeze(0).unsqueeze(0).to(device)
    return tensor


# --------------------
# Loaders
# --------------------
def load_model(model_path: str, device: str):
    model = WriterNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_embedding_db(db_path: str):
    with open(db_path, "rb") as f:
        return pickle.load(f)


def load_threshold(path: str) -> float:
    with open(path, "r") as f:
        return float(f.read())


# --------------------
# Verification
# --------------------
def verify_signature(
    img_path: str,
    writer_id: str,
    embed_model,
    embedding_db: dict,
    threshold: float,
    device: str
) -> Tuple[str, float]:

    if writer_id not in embedding_db:
        return "UNKNOWN_WRITER", None

    img = preprocess_image(img_path, device)

    with torch.no_grad():
        test_emb = embed_model(img).cpu()

    ref_embs = torch.cat(embedding_db[writer_id])
    dist = F.pairwise_distance(test_emb, ref_embs).mean().item()

    decision = "GENUINE" if dist < threshold else "FORGED"
    return decision, dist
