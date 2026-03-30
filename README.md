# Signature Verification System

A deep learning-based system for offline handwritten signature verification using CNN embeddings and metric learning. This project covers the full pipeline: data download, preprocessing, model training, evaluation, and a user-friendly Streamlit web app for inference.

---

## Table of Contents
- [Overview](#overview)
- [Dataset & Download](#dataset--download)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Training Procedure](#training-procedure)
- [Evaluation & Metrics](#evaluation--metrics)
- [Inference Pipeline](#inference-pipeline)
- [Streamlit Frontend](#streamlit-frontend)
- [Artifacts](#artifacts)
- [How to Run](#how-to-run)
- [References](#references)

---

## Overview
This project implements a robust signature verification system using deep metric learning. It leverages a ResNet18-based CNN to extract 256-dimensional embeddings from grayscale signature images, and uses distance-based thresholding to distinguish between genuine and forged signatures.

---

## Dataset & Download
- **Dataset:** [CEDAR Signature Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
- **Download:**
	- The dataset is automatically downloaded and extracted using Kaggle CLI in the training notebook:
		```python
		!kaggle datasets download -d shreelakshmigp/cedardataset
		!unzip -q cedardataset.zip -d data/raw/cedar_raw
		```
	- Directory structure after extraction:
		```
		data/
			raw/
				cedar_raw/
					signatures/
						full_org/   # Genuine signatures
						full_forg/  # Forged signatures
		```

---

## Data Processing
- **Train/Val/Test Split:**
	- Writers are split into train, validation, and test sets to ensure writer-disjoint evaluation.
	- Images are copied into `data/split/{train,val,test}/{org,forg}`.
- **Preprocessing:**
	- Convert to grayscale, binarize, crop to signature region, resize to 224x224, and normalize.
	- See `preprocess_image()` in the notebook and Inference.py.

---

## Model Architecture
- **Backbone:** ResNet18 (first conv layer modified for 1-channel input)
- **Embedding Layer:** 512 → 256-dim fully connected
- **Classifier:** Used only during training for writer classification
- **Embedding Output:** Used for verification (L2-normalized)

```
Input (224x224) → ResNet18 → 512-d → 256-d Embedding →
	├─ Classifier (training)
	└─ Embedding (verification)
```

---

## Training Procedure
- **Stage 1:**
	- Train as a writer classifier using CrossEntropyLoss.
	- Save the backbone and embedding layers for verification.
- **Stage 2 (Optional):**
	- Triplet loss fine-tuning for better separation (did not yield significant improvement in this project).
- **Embedding DB:**
	- For each test writer, compute embeddings for all genuine signatures and store as reference.
- **Threshold Selection:**
	- Compute distances for genuine-genuine and genuine-forgery pairs.
	- Use ROC curve to select the best threshold maximizing TPR-FPR.

---

## Evaluation & Metrics
- **Test Pairs:**
	- Build pairs of genuine-genuine and genuine-forgery signatures for each writer.
- **Metrics:**
	- Accuracy, False Positive Rate (forged predicted as genuine), False Negative Rate (genuine predicted as forged), per-writer error analysis.
- **Report:**
	- See `report.py` for detailed CSV-based analysis.

---

## Inference Pipeline
- **Preprocessing:**
	- Same as training: grayscale, binarize, crop, resize, normalize.
- **Embedding Extraction:**
	- Pass image through trained model to get 256-dim embedding.
- **Verification:**
	- Compute mean L2 distance to all reference embeddings for the claimed writer.
	- If distance < threshold → GENUINE, else FORGED.
	- If writer not enrolled → UNKNOWN_WRITER.
- **Code:** See `Inference.py` for all inference logic.

---

## Streamlit Frontend
- **Features:**
	- Upload one or more signature images.
	- Enter user name (mapped to writer ID).
	- Displays decision (GENUINE/FORGED/UNKNOWN), distance, and model confidence.
	- Downloadable CSV report of results.
- **How it works:**
	- Loads model, embedding DB, and threshold from `artifacts/`.
	- Uses the same inference logic as above.
	- See `app.py` for full UI and backend integration.

---

## Artifacts
- `artifacts/signature_embedder.pth` — Trained model weights
- `artifacts/embedding_db.pkl` — Reference embeddings for all writers
- `artifacts/threshold.txt` — Optimal threshold for verification
- `artifacts/signature_test_report.csv` — Test results (for report.py)

---

## How to Run

### 1. Environment Setup
- Install requirements:
	```bash
	pip install -r requirements.txt
	# or install manually: torch, torchvision, opencv-python, streamlit, pandas, scikit-learn, tqdm, pillow
	```
- (Optional) Download the CEDAR dataset manually if not using the notebook.

### 2. Model Training
- Run `training_notebook.ipynb` step by step to:
	- Download and extract data
	- Preprocess and split data
	- Train the model
	- Build embedding DB and select threshold
	- Save artifacts

### 3. Inference & Web App
- Place the generated artifacts in the `artifacts/` folder.
- Start the Streamlit app:
	```bash
	streamlit run app.py
	```
- Open the provided local URL in your browser.

### 4. Reporting
- After running tests, use `report.py` to analyze results:
	```bash
	python report.py
	```

---

## References
- [CEDAR Signature Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Example Results
- High accuracy on test set with clear separation between genuine and forged signatures.
- See `artifacts/signature_test_report.csv` and Streamlit UI for detailed results and visualizations.

---

## Project Structure
```
├── app.py                # Streamlit frontend
├── Inference.py          # Inference logic
├── report.py             # Evaluation/reporting
├── training_notebook.ipynb # Full training pipeline
├── artifacts/            # Model, embeddings, threshold, reports
├── data/                 # Raw and processed data
└── ...
```
