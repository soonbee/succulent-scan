from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

app = FastAPI()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".webp", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/inference")
async def inference(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file extension '{ext}'."
                f" Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail="File size exceeds 10MB limit.",
        )

    # import time

    # time.sleep(20)

    return [
        {"ko": "에오니움", "en": "aeonium", "acc": 78},
        {"ko": "에케베리아", "en": "echeveria", "acc": 48},
        {"ko": "리톱스", "en": "lithops", "acc": 13},
    ]
