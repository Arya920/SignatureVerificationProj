# app.py
import streamlit as st
import torch
import os
from PIL import Image
import tempfile
import time
import pandas as pd
from Inference import (
    load_model,
    load_embedding_db,
    load_threshold,
    verify_signature
)

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Signature Verification System",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Global CSS
# -------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {
    padding-top: 1rem;
}
.hero {
    padding: 2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #0ea5e9, #6366f1, #8b5cf6);
    color: white;
    box-shadow: 0 12px 24px rgba(0,0,0,0.15);
}
.hero h1 {
    margin: 0;
    font-weight: 800;
}
.hero p {
    margin-top: 0.5rem;
    opacity: 0.95;
}
.result-card {
    border-radius: 18px;
    padding: 1.4rem;
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    transition: all 0.2s ease-in-out;
    margin-bottom: 1rem;
}
.result-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.12);
}
.result-title {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 600;
}
.result-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.8rem;
}
.result-decision {
    font-size: 1rem;
    font-weight: 800;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    display: inline-block;
}
.genuine {
    background: #dcfce7;
    color: #166534;
}
.forged {
    background: #fee2e2;
    color: #991b1b;
}
.unknown {
    background: #fef9c3;
    color: #854d0e;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Demo User Mapping
# -------------------------------
USER_MAP = {
    "sunit": "19",
    "meenakshi": "30",
}

# -------------------------------
# Load Assets
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("artifacts/signature_embedder.pth", device)
    db = load_embedding_db("artifacts/embedding_db.pkl")
    threshold = load_threshold("artifacts/threshold.txt")
    return model, db, threshold, device

model, embedding_db, threshold, device = load_assets()

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("### ✍️ Signature Verification")
    st.caption("CNN-based writer verification using embedding distance.")
    st.divider()
    st.markdown("**Tips**")
    st.markdown(
        "- Upload clean, cropped signatures\n"
        "- Use same ink & resolution\n"
        "- Avoid background noise"
    )

# -------------------------------
# Hero Section
# -------------------------------
st.markdown("""
<div class="hero">
    <h1>Signature Verification System</h1>
    <p>Verify handwritten signatures using deep metric learning.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Tabs
# -------------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "upload"

tab_upload, tab_results, tab_help = st.tabs(
    ["📤 Upload & Verify", "📊 Results", "❓ Help"]
)

# -------------------------------
# Upload Tab
# -------------------------------
with tab_upload:
    st.subheader("Verification Setup")
    num_tests = st.number_input(
        "How many signatures do you want to verify?",
        min_value=1, max_value=10, step=1
    )
    st.divider()
    
    uploads = []
    for i in range(int(num_tests)):
        with st.container(border=True):
            st.markdown(f"**Signature {i+1}**")
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                user_input = st.text_input(
                    f"User Name #{i+1} (e.g. sunit, meenakshi)",
                    key=f"user_{i}"
                )
            with col2:
                file = st.file_uploader(
                    f"Upload Signature #{i+1}",
                    type=["png", "jpg", "jpeg"],
                    key=f"file_{i}"
                )
                if file:
                    img = Image.open(file)
                    st.image(img, width=120)
            uploads.append((user_input, file))
    
    st.divider()
    if st.button("🔍 Verify Signatures", type="primary", width='stretch'):
        results = []
        with st.spinner("Verifying signatures... Please wait ⏳"):
            for user_input, file in uploads:
                if not user_input or not file:
                    continue
                user_key = user_input.strip().lower()
                writer_id = USER_MAP.get(user_key, user_input.strip())
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(file.getvalue())
                    temp_path = tmp.name
                
                decision, dist = verify_signature(
                    img_path=temp_path,
                    writer_id=writer_id,
                    embed_model=model,
                    embedding_db=embedding_db,
                    threshold=threshold,
                    device=device
                )
                os.remove(temp_path)
                
                confidence = None
                if dist is not None:
                    confidence = max(0, min(100, int((1 - dist) * 100)))
                
                results.append({
                    "User_Input": user_input,
                    "Mapped_ID": writer_id,
                    "Signature_Decision": decision,
                    "Distance": None if dist is None else round(dist, 4),
                    "Confidence": confidence,
                    "Image_Bytes": file.getvalue()
                })
                time.sleep(2)
        
        st.session_state.results = results
        st.session_state.active_tab = "results"

if st.session_state.active_tab == "results":
    st.success("Verification completed successfully. Check the results in the result tab ✅")

# -------------------------------
# Results Tab
# -------------------------------
with tab_results:
    st.subheader("📊 Summary Overview")
    
    if "results" not in st.session_state or not st.session_state.results:
        st.info("No verification results yet.")
    else:
        results = st.session_state.results
        df = pd.DataFrame(results)
        
        # -------- Summary Metrics --------
        total = len(df)
        genuine_count = (df["Signature_Decision"] == "GENUINE").sum()
        forged_count = (df["Signature_Decision"] == "FORGED").sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Signatures", total)
        col2.metric("Genuine", genuine_count, delta=f"{int((genuine_count/total)*100)}%" if total else None)
        col3.metric("Forged", forged_count, delta=f"{int((forged_count/total)*100)}%" if total else None)
        
        st.divider()
        col4, col5 = st.columns([1,1], gap="small")
        
        # -------- Individual Result Blocks --------
        with col4:
            st.subheader("Individual Results")
            st.info(
                "Decision: Based on embedding distance. If :orange-badge[avg distance > 0.608 → Forged, else Genuine.] \n\n"
                
                "Confidence: Indicates how strongly the model supports its decision. "
                "It is derived from the distance score - predictions farther from the threshold (0.608) have higher confidence, "
                "while values near the threshold indicate uncertainty."
            )
            with st.expander("View Individual Results", expanded=False):
                for r in results:
                    decision = r["Signature_Decision"]
                    if decision == "GENUINE":
                        st_color = "green"
                        icon = "✅"
                    elif decision == "FORGED":
                        st_color = "red"
                        icon = "❌"
                    else:
                        st_color = "orange"
                        icon = "⚠️"
                    
                    THRESHOLD = 0.608

                    with st.container(border=True):
                        col1, col2 = st.columns([1, 1], gap="small", vertical_alignment="center")

                        with col2:
                            st.markdown(f"**User Name:** {r['User_Input']}")
                            st.markdown(f"**User ID:** {r['Mapped_ID']}")
                            st.markdown(f"**Decision:** :{st_color}[{icon} {decision}]")

                            if r["Distance"] is not None:
                                dist = r["Distance"]

                                # ---- Confidence based on distance from threshold ----
                                raw_conf = abs(dist - THRESHOLD)
                                max_dev = max(abs(0 - THRESHOLD), abs(1 - THRESHOLD))  # normalization
                                confidence = int((raw_conf / max_dev) * 100)
                                confidence = max(0, min(100, confidence))

                                st.progress(confidence / 100, text=f"Model Confidence: {confidence}%")

                                # Optional: show distance for transparency (good for demo)
                                st.caption(f"Distance: {dist:.3f}")

                        with col1:
                            st.markdown("**Uploaded Signature**")
                            st.image(r["Image_Bytes"], width=220)
        
        # -------- Detailed Report --------
        with col5:
            st.subheader("Detailed Report")
            st.dataframe(df[["User_Input", "Mapped_ID", "Signature_Decision", "Distance"]], width='stretch')
            csv = df.drop(columns=["Image_Bytes"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Report (CSV)",
                data=csv,
                file_name="signature_verification_results.csv",
                mime="text/csv",
                width='stretch'
            )

# -------------------------------
# Help Tab
# -------------------------------
with tab_help:
    st.markdown("""
**How it works**
- CNN extracts a 256-D embedding
- Distance computed against reference signatures
- Threshold-based decision
- The model converts every signature into a mathematical fingerprint (a 256-number vector).
Signatures from the same person produce very similar fingerprints, while different writers produce very different ones.
The system measures the distance between these fingerprints and decides whether the signature matches the stored references.

**Decisions**
- GENUINE → distance < threshold
- FORGED → distance ≥ threshold
- UNKNOWN_WRITER → writer not enrolled

""")