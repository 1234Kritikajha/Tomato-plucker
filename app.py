from pathlib import Path
import json
import tempfile

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from auto_learning import LearningManager
from tomato_detector import TomatoDetector, TomatoDetectorError


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Tomato Ripeness Detection API",
    description="Upload a tomato image and classify it as ripe or unripe.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
detector = TomatoDetector()
learning_manager = LearningManager(reload_callback=detector.reload)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api")
def api_info():
    return {
        "status": "running",
        "message": "Tomato Ripeness Detection API",
        "classes": detector.class_names,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": detector.model_path.name,
        "learning": learning_manager.status,
    }


@app.get("/classes")
def classes():
    return {"classes": learning_manager.labels()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        return detector.detect_frame(file.file)
    except TomatoDetectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/learn")
async def learn(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    label: str = Form(...),
    retrain: bool = Form(True),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        saved_path = learning_manager.save_sample(temp_path, label)
        temp_path.unlink(missing_ok=True)
        retraining_started = False
        if retrain:
            retraining_started = learning_manager.train_async()

        return {
            "status": "saved",
            "label": label,
            "sample": saved_path.name,
            "retraining_started": retraining_started,
            "learning": learning_manager.status,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/annotations/learn")
async def learn_annotations(
    file: UploadFile = File(...),
    annotations: str = Form(...),
    retrain: bool = Form(True),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        parsed = json.loads(annotations)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("Annotations must be a non-empty list.")

        image = Image.open(file.file).convert("RGB")
        saved = []
        for index, annotation in enumerate(parsed, start=1):
            label = str(annotation.get("label", "")).strip()
            annotation_id = annotation.get("id", index)
            predicted_label = str(annotation.get("prediction", "")).strip() or None
            confidence = annotation.get("confidence")
            bbox = annotation.get("bbox", {})
            x = max(0, int(float(bbox.get("x", 0))))
            y = max(0, int(float(bbox.get("y", 0))))
            width = max(1, int(float(bbox.get("width", image.width))))
            height = max(1, int(float(bbox.get("height", image.height))))
            x2 = min(image.width, x + width)
            y2 = min(image.height, y + height)
            if x >= x2 or y >= y2:
                raise ValueError(f"Annotation #{index} has an invalid box.")

            crop = image.crop((x, y, x2, y2))
            normalized_bbox = {
                "x": x,
                "y": y,
                "width": x2 - x,
                "height": y2 - y,
            }
            saved_path = learning_manager.save_image(
                crop,
                label,
                f"{Path(file.filename).stem}_{index}",
                metadata={
                    "source": "manual_annotation",
                    "original_filename": file.filename,
                    "annotation_id": annotation_id,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "bbox": normalized_bbox,
                    "image": {"width": image.width, "height": image.height},
                },
            )
            saved.append({
                "id": annotation_id,
                "label": label,
                "sample": saved_path.name,
                "bbox": normalized_bbox,
            })

        retraining_started = False
        if retrain:
            retraining_started = learning_manager.train_async()

        return {
            "status": "saved",
            "saved": saved,
            "classes": learning_manager.labels(),
            "retraining_started": retraining_started,
            "learning": learning_manager.status,
        }
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/learning/status")
def learning_status():
    return learning_manager.status


@app.get("/annotations/history")
def annotation_history(limit: int = 50):
    safe_limit = min(max(limit, 1), 500)
    return {
        "records": learning_manager.annotation_history(limit=safe_limit),
        "total_records": learning_manager.count_annotation_records(),
    }
