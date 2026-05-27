from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import base64
from io import BytesIO
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchvision import transforms


class ImageCNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return torch.relu(self.fc(x))


class TabularMLPBranch(nn.Module):
    def __init__(self, input_size: int = 6, h1: int = 128, h2: int = 64, out_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return torch.relu(self.fc3(x))


class Multimodal3ClassModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        tabular_input_size: int = 6,
        tab_h1: int = 128,
        tab_h2: int = 64,
        tab_out: int = 32,
        fusion_h1: int = 160,
        fusion_h2: int = 80,
        dropout: float = 0.45,
    ):
        super().__init__()
        self.image_branch = ImageCNNBranch()
        self.tabular_branch = TabularMLPBranch(tabular_input_size, tab_h1, tab_h2, tab_out)
        self.tabular_gate = nn.Sequential(
            nn.Linear(tab_out, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
        )
        self.image_head = nn.Linear(128, num_classes)
        self.tabular_head = nn.Linear(tab_out, num_classes)
        self.blend_gate = nn.Sequential(
            nn.Linear(tab_out, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + tab_out, fusion_h1),
            nn.BatchNorm1d(fusion_h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_h1, fusion_h2),
            nn.BatchNorm1d(fusion_h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_h2, num_classes),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor):
        image_features = self.image_branch(image)
        tabular_features = self.tabular_branch(tabular)

        image_logits = self.image_head(image_features)
        tabular_logits = self.tabular_head(tabular_features)

        image_probs = torch.softmax(image_logits, dim=1)
        image_conf = torch.max(image_probs, dim=1, keepdim=True).values
        boundary_strength = (1.0 - image_conf).clamp(0.0, 1.0)

        gate = self.tabular_gate(tabular_features)
        uncertainty_suppression = 1.0 - 0.25 * boundary_strength
        gated_image = image_features * gate * uncertainty_suppression

        fusion_logits = self.fusion(torch.cat([gated_image, tabular_features], dim=1))

        blend = self.blend_gate(tabular_features)
        tabular_boost = 0.38 * boundary_strength
        image_weight = (blend * (1.0 - tabular_boost)).clamp(0.08, 0.92)
        direct_mix = image_weight * image_logits + (1.0 - image_weight) * tabular_logits
        return fusion_logits + direct_mix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")

MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "production" / "multimodal_model_3class_distilled_student.pth",
]
MODEL_DATASET_CSV = PROJECT_ROOT / "dataset" / "multimodal_image_dataset_3class_roi.csv"
TEMPERATURE_3CLASS_PATH = PROJECT_ROOT / "models" / "production" / "temperature_scaling_3class.json"
CLASS_NAMES_DEFAULT = ["high_malignancy", "low_malignancy", "medium_risk"]
CLASS_NAMES_REPORT_PATH = PROJECT_ROOT / "reports" / "final" / "multiclass_3class_eval_report_distilled_student.json"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "0").strip().lower() in {"1", "true", "yes", "on"}

MODEL: Multimodal3ClassModel | None = None
ENCODERS: dict[str, Any] | None = None
SCALER: StandardScaler | None = None
TEMPERATURE: float = 1.0
CLASS_NAMES_BY_INDEX: list[str] = CLASS_NAMES_DEFAULT.copy()
STARTUP_ERROR: str | None = None
MODEL_CHECKPOINT_NAME: str = "unknown"
NUMERIC_DEFAULTS: dict[str, float] = {}
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017").strip()
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ungthuphoi_demo").strip() or "ungthuphoi_demo"
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "prediction_history").strip() or "prediction_history"
MONGO_CLIENT: AsyncIOMotorClient | None = None
MONGO_COLLECTION: AsyncIOMotorCollection | None = None
MONGO_STARTUP_ERROR: str | None = None
YOLO_MODEL_PATH_RAW = os.getenv(
    "YOLO_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "localization" / "yolo_nodule_kaggle_m640_e80_fixray2_best.pt"),
).strip()
YOLO_MODEL_PATH = Path(YOLO_MODEL_PATH_RAW)
if not YOLO_MODEL_PATH.is_absolute():
    YOLO_MODEL_PATH = PROJECT_ROOT / YOLO_MODEL_PATH
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.12"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
YOLO_MODEL: Any | None = None
YOLO_STARTUP_ERROR: str | None = None


app = FastAPI(title="UngThuPhoi Inference API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_class_names() -> list[str]:
    if CLASS_NAMES_REPORT_PATH.exists():
        try:
            data = json.loads(CLASS_NAMES_REPORT_PATH.read_text(encoding="utf-8"))
            names = data.get("class_names_by_idx")
            if isinstance(names, list) and len(names) == 3:
                return [str(x).strip() for x in names]
        except Exception:
            pass
    return CLASS_NAMES_DEFAULT.copy()


def load_temperature() -> float:
    if not TEMPERATURE_3CLASS_PATH.exists():
        return 1.0
    try:
        data = json.loads(TEMPERATURE_3CLASS_PATH.read_text(encoding="utf-8"))
        value = float(data.get("temperature", 1.0))
        return value if value > 0 and np.isfinite(value) else 1.0
    except Exception:
        return 1.0


def load_tabular_preprocessors() -> tuple[dict[str, Any], StandardScaler, dict[str, float]]:
    if not MODEL_DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {MODEL_DATASET_CSV}")

    df = pd.read_csv(MODEL_DATASET_CSV)
    train_df = (
        df[df["split"].astype(str).str.strip() == "train"].copy()
        if "split" in df.columns
        else df.iloc[0:0].copy()
    )
    fit_df = train_df if not train_df.empty else df
    categorical_cols = ["sex", "smoking_status", "family_history"]
    numeric_cols = ["age", "tumor_size", "symptom_score"]
    numeric_defaults: dict[str, float] = {}
    for col in numeric_cols:
        values = pd.to_numeric(fit_df[col], errors="coerce")
        median = float(values.median())
        numeric_defaults[col] = median if np.isfinite(median) else 0.0

    encoders: dict[str, Any] = {}
    for col in categorical_cols:
        enc = LabelEncoder()
        enc.fit(fit_df[col].astype(str).str.strip().values)
        encoders[col] = enc

    features = []
    for _, row in fit_df.iterrows():
        row_feat: list[float] = []
        for col in categorical_cols:
            val = str(row[col]).strip()
            try:
                row_feat.append(float(encoders[col].transform([val])[0]))
            except Exception:
                row_feat.append(0.0)
        for col in numeric_cols:
            try:
                row_feat.append(float(row[col]))
            except Exception:
                row_feat.append(0.0)
        features.append(row_feat)

    scaler = StandardScaler()
    scaler.fit(np.array(features, dtype=np.float32))
    return encoders, scaler, numeric_defaults


def load_model() -> Multimodal3ClassModel:
    global MODEL_CHECKPOINT_NAME
    model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("Production checkpoint not found: models/production/multimodal_model_3class_distilled_student.pth")

    MODEL_CHECKPOINT_NAME = model_path.name
    model = Multimodal3ClassModel(num_classes=3)
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_yolo_model() -> Any | None:
    global YOLO_STARTUP_ERROR
    if not YOLO_MODEL_PATH.exists():
        YOLO_STARTUP_ERROR = f"YOLO checkpoint not found: {YOLO_MODEL_PATH}"
        return None

    # Keep third-party cache writes inside the project. This avoids Windows user-dir
    # permission problems and mirrors the stable training environment.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_config"))
    (PROJECT_ROOT / ".ultralytics_config").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".matplotlib_config").mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO

        model = YOLO(str(YOLO_MODEL_PATH))
        YOLO_STARTUP_ERROR = None
        return model
    except Exception as exc:
        YOLO_STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    if image.mode != "L":
        image = image.convert("L")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transform(image)


def validate_uploaded_lung_ct(image: Image.Image) -> tuple[bool, list[str]]:
    """Fast, conservative domain check for uploaded lung CT-like images."""
    issues: list[str] = []
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 96 or height < 96:
        issues.append("Ảnh quá nhỏ cho mô hình dự đoán.")
        return False, issues

    ratio = width / max(1, height)
    if ratio < 0.45 or ratio > 2.2:
        issues.append("Tỷ lệ ảnh không giống lát cắt CT phổi điển hình.")

    rgb.thumbnail((256, 256), Image.Resampling.LANCZOS)
    arr = np.asarray(rgb).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        issues.append("Không đọc được dữ liệu màu/gray của ảnh.")
        return False, issues

    gray_dev = np.mean((np.abs(arr[:, :, 0] - arr[:, :, 1]) + np.abs(arr[:, :, 1] - arr[:, :, 2]) + np.abs(arr[:, :, 0] - arr[:, :, 2])) / 3.0)
    if gray_dev > 25:
        issues.append("Ảnh không giống ảnh grayscale/CT-like.")

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    body = lum > 12
    body_count = int(body.sum())
    if body_count < int(lum.size * 0.08):
        issues.append("Không thấy đủ vùng cơ thể trong ảnh CT.")
        return False, issues

    # Lung window slices usually contain two broad low-density air regions.
    air = body & (lum < 85)
    h, w = air.shape
    mid = w // 2
    left_body = int(body[:, :mid].sum())
    right_body = int(body[:, mid:].sum())
    left_air_ratio = int(air[:, :mid].sum()) / max(1, left_body)
    right_air_ratio = int(air[:, mid:].sum()) / max(1, right_body)
    if left_air_ratio < 0.06 or right_air_ratio < 0.06:
        issues.append("Không nhận diện được đủ hai vùng phổi trái/phải.")

    return len(issues) == 0, issues


def coerce_numeric_feature(value: str, field_name: str) -> tuple[float, bool, str]:
    raw = str(value or "").strip()
    if raw == "":
        if field_name == "tumor_size" and SCALER is not None:
            # The deployed checkpoint has a fixed 6-feature tabular input. For a
            # genuinely missing optional tumor_size, encode the standardized
            # value as neutral (0.0) by using the scaler mean in raw space. This
            # keeps inference possible without presenting the value as a real
            # patient measurement.
            return float(SCALER.mean_[4]), True, "missing_neutral"
        return float(NUMERIC_DEFAULTS.get(field_name, 0.0)), True, "missing"
    try:
        parsed = float(raw)
        if np.isfinite(parsed):
            return parsed, False, "provided"
    except Exception:
        pass
    return float(NUMERIC_DEFAULTS.get(field_name, 0.0)), True, "invalid"


def preprocess_tabular(age: str, sex: str, smoking_status: str, tumor_size: str, family_history: str, symptom_score: str) -> tuple[torch.Tensor, dict[str, Any]]:
    global ENCODERS, SCALER, NUMERIC_DEFAULTS
    if ENCODERS is None or SCALER is None:
        ENCODERS, SCALER, NUMERIC_DEFAULTS = load_tabular_preprocessors()

    # IMPORTANT: same order as training_utils -> categorical first, then numeric.
    features: list[float] = []
    for col, val in [("sex", sex), ("smoking_status", smoking_status), ("family_history", family_history)]:
        value = str(val).strip()
        enc = ENCODERS.get(col)
        if enc is not None and value in enc.classes_:
            features.append(float(enc.transform([value])[0]))
        else:
            features.append(0.0)

    imputed: dict[str, float] = {}
    missing_encoded: dict[str, Any] = {}
    missing_fields: list[str] = []
    invalid_fields: list[str] = []
    for col, val in [("age", age), ("tumor_size", tumor_size), ("symptom_score", symptom_score)]:
        parsed, was_imputed, reason = coerce_numeric_feature(val, col)
        features.append(parsed)
        if reason == "missing_neutral":
            missing_encoded[col] = {
                "raw_reference_value": parsed,
                "scaled_value": 0.0,
                "strategy": "neutral_scaler_mean",
            }
        elif was_imputed:
            imputed[col] = parsed
        if reason in {"missing", "missing_neutral"}:
            missing_fields.append(col)
        elif reason == "invalid":
            invalid_fields.append(col)

    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    scaled = SCALER.transform(arr)[0]
    tumor_missing = "tumor_size" in missing_fields
    return torch.tensor(scaled, dtype=torch.float32), {
        "features": {
            "age": features[3],
            "tumor_size": None if tumor_missing else features[4],
            "symptom_score": features[5],
        },
        "model_features": {
            "sex": features[0],
            "smoking_status": features[1],
            "family_history": features[2],
            "age": features[3],
            "tumor_size": features[4],
            "symptom_score": features[5],
        },
        "imputed": imputed,
        "missing_encoded": missing_encoded,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "imputation_strategy": {
            "type": "fixed_vector_missing_encoding",
            "reason": "The deployed checkpoint expects a fixed 6-feature tabular vector. Optional missing tumor_size is encoded as the scaler-neutral value for inference only and is not treated as a patient measurement.",
        },
    }


def image_to_data_url(image: Image.Image, max_side: int = 900) -> str:
    preview = image.convert("RGB")
    if max(preview.size) > max_side:
        preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    preview.save(buf, format="JPEG", quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_file_to_web_png(path: Path, max_side: int = 520) -> bytes:
    """Return a browser-safe 8-bit RGB PNG for CT thumbnails/previews."""
    with Image.open(path) as image:
        source = image.copy()

    if source.mode not in {"RGB", "RGBA", "L"}:
        arr = np.asarray(source)
        if arr.ndim == 2:
            arr = arr.astype(np.float32)
            lo = float(np.percentile(arr, 1))
            hi = float(np.percentile(arr, 99))
            if hi <= lo:
                lo = float(np.min(arr))
                hi = float(np.max(arr))
            if hi > lo:
                arr = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            source = Image.fromarray(arr.astype(np.uint8), mode="L")
        else:
            source = source.convert("RGB")

    if source.mode == "L":
        web_image = ImageOps.autocontrast(source).convert("RGB")
    else:
        web_image = source.convert("RGB")
    if max(web_image.size) > max_side:
        web_image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    web_image.save(buf, format="PNG")
    return buf.getvalue()


def draw_yolo_boxes(image: Image.Image, boxes: list[dict[str, Any]]) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        conf = float(box.get("confidence", 0.0))
        label = f"not nghi ngo {conf * 100:.1f}%"
        color = (255, 68, 68)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        text_w = text_bbox[2] - text_bbox[0]
        y_text = max(0, y1 - text_h - 6)
        draw.rectangle((x1, y_text, x1 + text_w + 8, y_text + text_h + 6), fill=color)
        draw.text((x1 + 4, y_text + 3), label, fill=(255, 255, 255), font=font)
    return annotated


def run_yolo_localization(image: Image.Image, source: str) -> dict[str, Any]:
    if YOLO_MODEL is None:
        return {
            "available": False,
            "error": YOLO_STARTUP_ERROR or "YOLO model is not loaded",
            "boxes": [],
            "box_count": 0,
            "annotated_image": None,
            "checkpoint": str(YOLO_MODEL_PATH.relative_to(PROJECT_ROOT)) if YOLO_MODEL_PATH.exists() else str(YOLO_MODEL_PATH),
        }

    source_image = image.convert("RGB")
    try:
        results = YOLO_MODEL.predict(
            source=np.array(source_image),
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONF_THRESHOLD,
            device="cpu",
            verbose=False,
        )
        result = results[0] if results else None
        boxes: list[dict[str, Any]] = []
        if result is not None and result.boxes is not None:
            for idx, det in enumerate(result.boxes):
                xyxy = det.xyxy[0].detach().cpu().numpy().tolist()
                conf = float(det.conf[0].detach().cpu().item()) if det.conf is not None else 0.0
                cls_id = int(det.cls[0].detach().cpu().item()) if det.cls is not None else 0
                boxes.append(
                    {
                        "id": idx,
                        "class_id": cls_id,
                        "label": "nodule",
                        "confidence": conf,
                        "xyxy": [float(v) for v in xyxy],
                    }
                )

        annotated = draw_yolo_boxes(source_image, boxes)
        return {
            "available": True,
            "error": None,
            "source": source,
            "checkpoint": str(YOLO_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "conf_threshold": YOLO_CONF_THRESHOLD,
            "image_size": YOLO_IMAGE_SIZE,
            "boxes": boxes,
            "box_count": len(boxes),
            "annotated_image": image_to_data_url(annotated),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "source": source,
            "checkpoint": str(YOLO_MODEL_PATH.relative_to(PROJECT_ROOT)) if YOLO_MODEL_PATH.exists() else str(YOLO_MODEL_PATH),
            "boxes": [],
            "box_count": 0,
            "annotated_image": None,
        }


def resolve_allowed_sample_image(path: str) -> Path:
    requested = Path(str(path))
    if requested.exists():
        candidate = requested.resolve()
    else:
        candidate = (PROJECT_ROOT / path.lstrip("/\\")).resolve()

    allowed_dir = (PROJECT_ROOT / "data").resolve()
    try:
        candidate.relative_to(allowed_dir)
    except Exception:
        raise HTTPException(status_code=403, detail="Access to requested file is not allowed")

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return candidate


def load_predictor_assets() -> None:
    global MODEL, ENCODERS, SCALER, TEMPERATURE, CLASS_NAMES_BY_INDEX, YOLO_MODEL, NUMERIC_DEFAULTS
    CLASS_NAMES_BY_INDEX = load_class_names()
    TEMPERATURE = load_temperature()
    ENCODERS, SCALER, NUMERIC_DEFAULTS = load_tabular_preprocessors()
    MODEL = load_model()
    YOLO_MODEL = load_yolo_model()


async def init_mongo() -> None:
    global MONGO_CLIENT, MONGO_COLLECTION, MONGO_STARTUP_ERROR
    try:
        client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=2500)
        await client.admin.command("ping")
        collection = client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]
        await collection.create_index("timestamp")
        await collection.create_index("patient_id")
        await collection.create_index("patient_name")
        MONGO_CLIENT = client
        MONGO_COLLECTION = collection
        MONGO_STARTUP_ERROR = None
    except Exception as exc:
        MONGO_CLIENT = None
        MONGO_COLLECTION = None
        MONGO_STARTUP_ERROR = f"{type(exc).__name__}: {exc}"


def ensure_mongo_collection() -> AsyncIOMotorCollection:
    if MONGO_COLLECTION is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "MongoDB is not available",
                "issues": [MONGO_STARTUP_ERROR or "MongoDB connection not initialized"],
            },
        )
    return MONGO_COLLECTION


def serialize_prediction_doc(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    raw_id = out.pop("_id", None)
    out["id"] = str(raw_id) if raw_id is not None else ""
    ts = out.get("timestamp")
    if isinstance(ts, datetime):
        out["timestamp"] = ts.astimezone(timezone.utc).isoformat()
    return out


def build_record_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    suffix = uuid4().hex[:6].upper()
    return f"REC-{now}-{suffix}"


@app.on_event("startup")
async def _startup() -> None:
    global STARTUP_ERROR, MODEL
    try:
        load_predictor_assets()
        STARTUP_ERROR = None
    except Exception as exc:
        MODEL = None
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
    await init_mongo()


@app.get("/")
def root() -> dict[str, Any]:
    index_file = FRONTEND_DIST_DIR / "index.html"
    if SERVE_FRONTEND and index_file.exists():
        return FileResponse(index_file)
    return {
        "service": "UngThuPhoi Inference API",
        "status": "ok" if STARTUP_ERROR is None else "degraded",
        "health_endpoint": "/health",
        "docs_endpoint": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    status = "ok" if STARTUP_ERROR is None and MONGO_STARTUP_ERROR is None else "degraded"
    return {
        "status": status,
        "model_loaded": MODEL is not None,
        "checkpoint": MODEL_CHECKPOINT_NAME,
        "temperature": TEMPERATURE,
        "class_names_by_index": CLASS_NAMES_BY_INDEX,
        "dataset_csv": str(MODEL_DATASET_CSV.relative_to(PROJECT_ROOT)),
        "startup_error": STARTUP_ERROR,
        "mongo_connected": MONGO_COLLECTION is not None,
        "mongo_db": MONGO_DB_NAME,
        "mongo_collection": MONGO_COLLECTION_NAME,
        "mongo_startup_error": MONGO_STARTUP_ERROR,
        "yolo_localization_loaded": YOLO_MODEL is not None,
        "yolo_checkpoint": str(YOLO_MODEL_PATH.relative_to(PROJECT_ROOT)) if YOLO_MODEL_PATH.exists() else str(YOLO_MODEL_PATH),
        "yolo_conf_threshold": YOLO_CONF_THRESHOLD,
        "yolo_startup_error": YOLO_STARTUP_ERROR,
    }


@app.get("/sample-image")
def sample_image(
    path: str = Query(..., description="Absolute or project-relative image path"),
    view: str = Query(default="preview", description="Cache-busting/display mode label used by the frontend"),
):
    candidate = resolve_allowed_sample_image(path)
    media_type = mimetypes.guess_type(str(candidate))[0] or "image/png"
    # Return bytes directly instead of FileResponse. On the Windows demo setup the
    # detached uvicorn process could terminate silently while streaming sample
    # files, which surfaced in the UI as "Failed to fetch".
    return Response(
        content=candidate.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/predict")
async def predict(
    image: UploadFile | None = File(default=None),
    sample_path: str | None = Form(default=None),
    localization_sample_path: str | None = Form(default=None),
    patient_name: str = Form(default=""),
    patient_id: str = Form(default=""),
    age: str = Form(...),
    sex: str = Form(...),
    smoking_status: str = Form(...),
    tumor_size: str = Form(default=""),
    family_history: str = Form(...),
    symptom_score: str = Form(...),
):
    if STARTUP_ERROR is not None:
        raise HTTPException(status_code=503, detail={"message": "Model service unavailable", "issues": [STARTUP_ERROR]})
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    pil_image: Image.Image
    localization_image: Image.Image | None = None
    localization_source = "classification_input"
    resolved_sample_path: str | None = None
    resolved_localization_sample_path: str | None = None
    upload_filename: str | None = None

    if image is not None:
        upload_filename = image.filename
        if not image.content_type or not str(image.content_type).startswith("image/"):
            raise HTTPException(status_code=400, detail={"message": "Invalid image content type"})
        try:
            image_bytes = await image.read()
            uploaded_image = Image.open(BytesIO(image_bytes))
            is_lung_ct, validation_issues = validate_uploaded_lung_ct(uploaded_image)
            if not is_lung_ct:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Đây không phải ảnh CT phổi phù hợp cho mô hình. Vui lòng tải đúng ảnh CT phổi.",
                        "issues": validation_issues,
                    },
                )
            pil_image = uploaded_image.convert("L")
            localization_image = uploaded_image.convert("RGB")
            localization_source = "upload"
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail={"message": "Invalid uploaded image"})
    elif sample_path:
        sample_img = resolve_allowed_sample_image(sample_path)
        resolved_sample_path = str(sample_img.relative_to(PROJECT_ROOT)).replace("\\", "/")
        try:
            pil_image = Image.open(sample_img).convert("L")
        except Exception:
            raise HTTPException(status_code=400, detail={"message": "Invalid sample image"})

        yolo_sample_path = localization_sample_path or sample_path
        try:
            yolo_sample_img = resolve_allowed_sample_image(yolo_sample_path)
            resolved_localization_sample_path = str(yolo_sample_img.relative_to(PROJECT_ROOT)).replace("\\", "/")
            localization_image = Image.open(yolo_sample_img).convert("RGB")
            localization_source = "sample_preview" if localization_sample_path else "sample_classification_input"
        except HTTPException:
            localization_image = pil_image.convert("RGB")
            localization_source = "sample_classification_input"
        except Exception:
            localization_image = pil_image.convert("RGB")
            localization_source = "sample_classification_input"
    else:
        raise HTTPException(status_code=400, detail={"message": "Provide either image or sample_path"})

    localization_result = run_yolo_localization(localization_image or pil_image.convert("RGB"), localization_source)

    image_tensor = preprocess_image(pil_image)
    tabular_tensor, tabular_meta = preprocess_tabular(age, sex, smoking_status, tumor_size, family_history, symptom_score)

    with torch.no_grad():
        logits = MODEL(image_tensor.unsqueeze(0).to(DEVICE), tabular_tensor.unsqueeze(0).to(DEVICE))[0]
        probs = torch.softmax(logits / TEMPERATURE, dim=0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES_BY_INDEX[pred_idx] if pred_idx < len(CLASS_NAMES_BY_INDEX) else "unknown"

    prob_map = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES_BY_INDEX)}
    low_p = float(prob_map.get("low_malignancy", 0.0))
    med_p = float(prob_map.get("medium_risk", 0.0))
    high_p = float(prob_map.get("high_malignancy", 0.0))

    img_buf = BytesIO()
    pil_image.save(img_buf, format="PNG")
    img_hash = hashlib.sha1(img_buf.getvalue()).hexdigest()[:16]

    debug_payload = {
        "sample_path_requested": sample_path,
        "sample_path_resolved": resolved_sample_path,
        "localization_sample_path_requested": localization_sample_path,
        "localization_sample_path_resolved": resolved_localization_sample_path,
        "upload_filename": upload_filename,
        "image_sha1_16": img_hash,
        "raw_logits": [float(x) for x in logits.cpu().numpy().tolist()],
        "softmax_probabilities": prob_map,
        "predicted_class_index": pred_idx,
        "predicted_label": pred_label,
        "display_label": pred_label,
        "class_names_by_index": CLASS_NAMES_BY_INDEX,
    }
    print("[PREDICT DEBUG]", json.dumps(debug_payload, ensure_ascii=True))

    warnings: list[str] = []
    if pred_label == "medium_risk":
        warnings.append("Medium-risk case: interpret with caution.")
    if "tumor_size" in tabular_meta.get("missing_fields", []):
        warnings.append(
            "Kích thước u chưa được nhập. Checkpoint production hiện tại vẫn cần vector tabular 6 chiều, "
            "nên backend mã hóa trường này ở mức trung tính của scaler để chạy inference; đây không phải kích thước u thật của bệnh nhân."
        )
    elif "tumor_size" in tabular_meta.get("imputed", {}):
        warnings.append(
            "Kích thước u nhập vào không hợp lệ. Backend đã dùng giá trị thay thế để tránh lỗi inference; "
            "hãy kiểm tra lại thông tin đầu vào nếu dùng kết quả cho báo cáo."
        )
    if localization_result.get("available") and int(localization_result.get("box_count") or 0) == 0:
        warnings.append("YOLO không phát hiện vùng nghi ngờ rõ ràng theo ngưỡng hiện tại; mô hình phân loại vẫn được chạy.")
    localization_no_box_high_risk = (
        localization_result.get("available")
        and int(localization_result.get("box_count") or 0) == 0
        and pred_label == "high_malignancy"
    )
    if localization_no_box_high_risk:
        warnings.append(
            "Cảnh báo bất nhất: mô hình phân loại nghiêng về nguy cơ cao nhưng YOLO không phát hiện vùng nốt rõ ràng. "
            "Cần xem lại ảnh/ROI và không diễn giải như bằng chứng định vị tổn thương."
        )

    normalized_patient_name = patient_name.strip() or "Chưa cung c?p"
    normalized_patient_id = patient_id.strip() or build_record_id()
    source = "sample" if resolved_sample_path else "upload"
    image_path_or_name = resolved_sample_path or upload_filename or "unknown_image"
    now_utc = datetime.now(timezone.utc)

    history_doc = {
        "patient_name": normalized_patient_name,
        "patient_id": normalized_patient_id,
        "age": age,
        "sex": sex,
        "smoking_status": smoking_status,
        "tumor_size": tumor_size,
        "tumor_size_effective": tabular_meta["features"]["tumor_size"],
        "tumor_size_imputed": "tumor_size" in tabular_meta.get("imputed", {}),
        "tumor_size_missing": "tumor_size" in tabular_meta.get("missing_fields", []),
        "tumor_size_missing_encoded": "tumor_size" in tabular_meta.get("missing_encoded", {}),
        "family_history": family_history,
        "symptom_score": symptom_score,
        "image_path": image_path_or_name,
        "source": source,
        "predicted_label": pred_label,
        "predicted_class_index": pred_idx,
        "probabilities": {
            "low_malignancy": low_p,
            "medium_risk": med_p,
            "high_malignancy": high_p,
        },
        "confidence": float(np.max(probs)),
        "temperature": TEMPERATURE,
        "checkpoint": MODEL_CHECKPOINT_NAME,
        "sample_path": resolved_sample_path,
        "upload_filename": upload_filename,
        "debug": debug_payload,
        "tabular_preprocessing": tabular_meta,
        "localization": {
            "source": localization_source,
            "sample_path": resolved_localization_sample_path,
            # Keep the YOLO overlay in history so the frontend can show the exact
            # image that was localized during this prediction after a page reload.
            "result": localization_result,
        },
        "timestamp": now_utc,
    }

    history_id: str | None = None
    try:
        collection = ensure_mongo_collection()
        insert_result = await collection.insert_one(history_doc)
        history_id = str(insert_result.inserted_id)
    except HTTPException:
        warnings.append("History save unavailable.")
    except Exception as exc:
        warnings.append("History save failed.")
        print(f"[HISTORY SAVE ERROR] {type(exc).__name__}: {exc}")

    return {
        "risk_level": pred_label,
        "predicted_class_index": pred_idx,
        "predicted_label": pred_label,
        "calibrated_probability": float(np.max(probs)),
        "confidence": float(np.max(probs)),
        "probabilities": {
            "low_malignancy": low_p,
            "medium_risk": med_p,
            "high_malignancy": high_p,
        },
        "conclusion": f"Model estimates risk group: {pred_label}.",
        "warning": " ".join(warnings) if warnings else None,
        "warnings": warnings,
        "temperature": TEMPERATURE,
        "checkpoint": MODEL_CHECKPOINT_NAME,
        "history_id": history_id,
        "input_summary": {
            "patient_name": normalized_patient_name,
            "patient_id": normalized_patient_id,
            "age": age,
            "sex": sex,
            "smoking_status": smoking_status,
            "tumor_size": tumor_size,
            "tumor_size_effective": (
                None if tabular_meta["features"]["tumor_size"] is None else str(tabular_meta["features"]["tumor_size"])
            ),
            "tumor_size_imputed": "tumor_size" in tabular_meta.get("imputed", {}),
            "tumor_size_missing": "tumor_size" in tabular_meta.get("missing_fields", []),
            "tumor_size_missing_encoded": "tumor_size" in tabular_meta.get("missing_encoded", {}),
            "family_history": family_history,
            "symptom_score": symptom_score,
            "sample_path": resolved_sample_path,
            "localization_sample_path": resolved_localization_sample_path,
            "upload_filename": upload_filename,
        },
        "debug": debug_payload,
        "tabular_preprocessing": tabular_meta,
        "localization": localization_result,
        "localization_classification_consistency": {
            "no_box_high_risk": bool(localization_no_box_high_risk),
            "message": (
                "Classifier high_malignancy nhưng YOLO box_count=0; đây là tín hiệu cần kiểm tra thủ công, không phải bằng chứng định vị."
                if localization_no_box_high_risk
                else None
            ),
        },
    }


@app.get("/predictions")
async def list_predictions(
    limit: int = Query(default=50, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
):
    collection = ensure_mongo_collection()
    cursor = collection.find({}, sort=[("timestamp", -1)]).skip(skip).limit(limit)
    rows = await cursor.to_list(length=limit)
    return {
        "items": [serialize_prediction_doc(row) for row in rows],
        "count": len(rows),
        "skip": skip,
        "limit": limit,
    }


@app.get("/predictions/{prediction_id}")
async def get_prediction_detail(prediction_id: str):
    collection = ensure_mongo_collection()
    try:
        oid = ObjectId(prediction_id)
    except (InvalidId, ValueError):
        raise HTTPException(status_code=400, detail={"message": "Invalid prediction id"})

    row = await collection.find_one({"_id": oid})
    if not row:
        raise HTTPException(status_code=404, detail={"message": "Prediction history not found"})
    return serialize_prediction_doc(row)


@app.delete("/predictions/{prediction_id}")
async def delete_prediction(prediction_id: str):
    collection = ensure_mongo_collection()
    try:
        oid = ObjectId(prediction_id)
    except (InvalidId, ValueError):
        raise HTTPException(status_code=400, detail={"message": "Invalid prediction id"})

    result = await collection.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail={"message": "Prediction history not found"})
    return {"deleted": True, "id": prediction_id}


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str):
    if not SERVE_FRONTEND:
        raise HTTPException(status_code=404, detail="Not Found")

    index_file = FRONTEND_DIST_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found")

    protected_prefixes = ("health", "predict", "sample-image", "docs", "openapi.json", "redoc")
    if full_path.startswith(protected_prefixes):
        raise HTTPException(status_code=404, detail="Not Found")

    requested = (FRONTEND_DIST_DIR / full_path).resolve()
    try:
        requested.relative_to(FRONTEND_DIST_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=403, detail="Forbidden path")

    if requested.is_file():
        return FileResponse(requested)
    return FileResponse(index_file)
