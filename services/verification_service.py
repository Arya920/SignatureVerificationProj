import tempfile
import os
from Inference import verify_signature

def verify_single_signature(
    file,
    user_input,
    model,
    embedding_db,
    threshold,
    device,
    writer_id
):
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

    return {
        "User_Input": user_input,
        "Mapped_ID": writer_id,
        "Signature_Decision": decision,
        "Distance": None if dist is None else round(dist, 4),
        "Image_Bytes": file.getvalue()
    }
