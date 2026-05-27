"""Streamlit Web Demo for Multimodal CT Malignancy Prediction Model."""

import sys
from pathlib import Path

# Add scripts directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchvision import transforms
import json

TABULAR_CSV_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "multimodal_image_dataset_3class_roi.csv",
    PROJECT_ROOT / "archive" / "legacy" / "datasets" / "multimodal_image_dataset_improved.csv",
    PROJECT_ROOT / "archive" / "legacy" / "datasets" / "multimodal_image_dataset.csv",
]
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "production" / "multimodal_model_3class_distilled_student.pth",
    PROJECT_ROOT / "archive_unused" / "models" / "production_legacy" / "multimodal_model_improved_gating.pth",
    PROJECT_ROOT / "archive" / "legacy" / "checkpoints" / "multimodal_model_research_best.pth",
    PROJECT_ROOT / "archive" / "legacy" / "checkpoints" / "multimodal_model_frozen_cnn.pth",
    PROJECT_ROOT / "archive" / "legacy" / "checkpoints" / "multimodal_model.pth",
]
TEMPERATURE_CANDIDATES = [
    PROJECT_ROOT / "models" / "production" / "temperature_scaling_3class.json",
    PROJECT_ROOT / "archive_unused" / "models" / "production_legacy" / "temperature_scaling.json",
]


def first_existing_path(candidates: list[Path]) -> Path | None:
    return next((p for p in candidates if p.exists()), None)

# Page config
st.set_page_config(
    page_title="Dự đoán ác tính CT",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL CLASSES
# ============================================================================

class ImageCNNBranch(nn.Module):
    """CNN branch for extracting image features."""

    def __init__(self):
        super(ImageCNNBranch, self).__init__()
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
        x = torch.relu(self.fc(x))

        return x


class TabularMLPBranch(nn.Module):
    """MLP branch for processing tabular features."""

    def __init__(self, input_size: int = 6, h1: int = 128, h2: int = 64, out_dim: int = 32):
        super(TabularMLPBranch, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc3(x))

        return x


class MultimodalModel(nn.Module):
    """Multimodal model combining image and tabular branches."""

    def __init__(
        self,
        num_classes: int = 2,
        tabular_input_size: int = 6,
        tab_h1: int = 128,
        tab_h2: int = 64,
        tab_out: int = 32,
        fusion_h1: int = 160,
        fusion_h2: int = 80,
        dropout: float = 0.5,
    ):
        super(MultimodalModel, self).__init__()
        self.image_branch = ImageCNNBranch()
        self.tabular_branch = TabularMLPBranch(
            input_size=tabular_input_size,
            h1=tab_h1,
            h2=tab_h2,
            out_dim=tab_out,
        )

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

    def forward(self, image: torch.Tensor, tabular: torch.Tensor, return_aux: bool = False):
        image_features = self.image_branch(image)
        tabular_features = self.tabular_branch(tabular)
        image_logits = self.image_head(image_features)
        tabular_logits = self.tabular_head(tabular_features)

        image_probs = torch.softmax(image_logits, dim=1)
        boundary_strength = (1.0 - torch.abs(image_probs[:, :1] - 0.5) * 2.0).clamp(0.0, 1.0)

        gate = self.tabular_gate(tabular_features)
        uncertainty_suppression = 1.0 - 0.24 * boundary_strength
        gated_image = image_features * gate * uncertainty_suppression

        combined = torch.cat([gated_image, tabular_features], dim=1)
        fusion_logits = self.fusion(combined)

        blend = self.blend_gate(tabular_features)
        tabular_boost = 0.36 * boundary_strength
        image_weight = (blend * (1.0 - tabular_boost)).clamp(0.05, 0.95)
        direct_mix = image_weight * image_logits + (1.0 - image_weight) * tabular_logits
        output = fusion_logits + direct_mix

        if return_aux:
            return output, image_logits, tabular_logits

        return output


# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load the multimodal model."""
    device = torch.device("cpu")
    model = MultimodalModel(num_classes=2, tabular_input_size=6, tab_h1=128, tab_h2=64, tab_out=32, fusion_h1=160, fusion_h2=80, dropout=0.5)
    model_path = first_existing_path(MODEL_CANDIDATES)
    if model_path is None:
        raise FileNotFoundError("Khong tim thay checkpoint binary tuong thich de demo.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def get_research_notice() -> str:
    return (
        "CẢNH BÁO: Đây chỉ là công cụ phục vụ nghiên cứu/học tập, không dùng để chẩn đoán lâm sàng. "
        "Mô hình hiện ổn hơn ở các ca low/high rõ rệt và vẫn còn hạn chế đáng kể với medium-risk."
    )


@st.cache_data
def get_random_test_samples(seed: int, n_samples: int = 10):
    """Return a random set of test samples with metadata and suggested tabular values."""
    csv_path = first_existing_path(TABULAR_CSV_CANDIDATES)
    if csv_path is None:
        return []
    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    if test_df.empty:
        return []

    sample_df = test_df.sample(n=min(n_samples, len(test_df)), random_state=seed).reset_index(drop=True)
    return sample_df.to_dict(orient="records")


def load_sample_image(image_path: str):
    """Load a sample CT image from disk."""
    try:
        if Path(image_path).exists():
            return Image.open(image_path).convert('L')
    except Exception:
        return None
    return None


@st.cache_data
def load_tabular_preprocessors():
    """Fit label encoders and scaler exactly like training utils."""
    csv_path = first_existing_path(TABULAR_CSV_CANDIDATES)
    if csv_path is None:
        raise FileNotFoundError("Khong tim thay CSV de tao tabular preprocessors.")
    df = pd.read_csv(csv_path)
    categorical_cols = ["sex", "smoking_status", "family_history"]
    numeric_cols = ["age", "tumor_size", "symptom_score"]

    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        encoder.fit(df[col].astype(str).str.strip().values)
        encoders[col] = encoder

    all_features = []
    for _, row in df.iterrows():
        features = []
        for col in categorical_cols:
            value = str(row[col]).strip()
            try:
                features.append(float(encoders[col].transform([value])[0]))
            except ValueError:
                features.append(0.0)

        for col in numeric_cols:
            try:
                features.append(float(row[col]))
            except (ValueError, TypeError):
                features.append(0.0)

        all_features.append(features)

    scaler = StandardScaler()
    scaler.fit(np.array(all_features, dtype=np.float32))
    return encoders, scaler


@st.cache_resource
def load_temperature():
    """Load fitted temperature from disk if available, default to 1.0."""
    path = first_existing_path(TEMPERATURE_CANDIDATES)
    if path is None:
        return 1.0
    if path.exists():
        try:
            data = json.loads(path.read_text())
            t = float(data.get("temperature", 1.0))
            if t <= 0 or not np.isfinite(t):
                return 1.0
            return t
        except Exception:
            return 1.0
    return 1.0


@st.cache_resource
def load_training_image_validation_stats():
    """Estimate image-domain bounds from training images for out-of-distribution rejection."""
    csv_path = first_existing_path(TABULAR_CSV_CANDIDATES)
    if csv_path is None:
        return None

    try:
        df = pd.read_csv(csv_path)
        train_df = df[df["split"].astype(str).str.strip() == "train"].copy()
    except Exception:
        return None

    if train_df.empty or "image_path" not in train_df.columns:
        return None

    stats = {
        "height": [],
        "width": [],
        "contrast": [],
        "dark_fraction": [],
        "mid_fraction": [],
        "grad_mean": [],
        "mean_intensity": [],
        "std_intensity": [],
    }

    for _, row in train_df.iterrows():
        image_path = str(row.get("image_path", "")).strip()
        if not image_path or not Path(image_path).exists():
            continue

        try:
            image = Image.open(image_path).convert("L")
            arr = np.array(image, dtype=np.float32)
            if arr.ndim != 2 or arr.size == 0:
                continue

            h, w = arr.shape
            p05 = float(np.percentile(arr, 5))
            p95 = float(np.percentile(arr, 95))
            gy = np.abs(np.diff(arr, axis=0)).mean() if h > 1 else 0.0
            gx = np.abs(np.diff(arr, axis=1)).mean() if w > 1 else 0.0
            grad_mean = float((gx + gy) / 2.0)

            stats["height"].append(float(h))
            stats["width"].append(float(w))
            stats["contrast"].append(float(p95 - p05))
            stats["dark_fraction"].append(float((arr < 20).mean()))
            stats["mid_fraction"].append(float(((arr >= 20) & (arr <= 230)).mean()))
            stats["grad_mean"].append(grad_mean)
            stats["mean_intensity"].append(float(arr.mean()))
            stats["std_intensity"].append(float(arr.std()))
        except Exception:
            continue

    if not stats["contrast"]:
        return None

    bounds = {}
    for key, values in stats.items():
        arr = np.array(values, dtype=np.float32)
        bounds[key] = {
            "low": float(np.percentile(arr, 2.5)),
            "high": float(np.percentile(arr, 97.5)),
            "median": float(np.median(arr)),
        }

    return bounds


@st.cache_resource
def load_training_tabular_validation_specs():
    """Load tabular value constraints from the training split."""
    csv_path = first_existing_path(TABULAR_CSV_CANDIDATES)
    if csv_path is None:
        return None

    try:
        df = pd.read_csv(csv_path)
        train_df = df[df["split"].astype(str).str.strip() == "train"].copy()
    except Exception:
        return None

    if train_df.empty:
        return None

    specs = {
        "age": {
            "low": float(np.percentile(pd.to_numeric(train_df["age"], errors="coerce").dropna().astype(float), 2.5)),
            "high": float(np.percentile(pd.to_numeric(train_df["age"], errors="coerce").dropna().astype(float), 97.5)),
        },
        "tumor_size": {
            "low": float(np.percentile(pd.to_numeric(train_df["tumor_size"], errors="coerce").dropna().astype(float), 2.5)),
            "high": float(np.percentile(pd.to_numeric(train_df["tumor_size"], errors="coerce").dropna().astype(float), 97.5)),
        },
        "symptom_score": {
            "low": float(np.percentile(pd.to_numeric(train_df["symptom_score"], errors="coerce").dropna().astype(float), 2.5)),
            "high": float(np.percentile(pd.to_numeric(train_df["symptom_score"], errors="coerce").dropna().astype(float), 97.5)),
        },
        "allowed_sex": sorted({str(v).strip() for v in train_df["sex"].dropna().astype(str).values}),
        "allowed_smoking_status": sorted({str(v).strip() for v in train_df["smoking_status"].dropna().astype(str).values}),
        "allowed_family_history": sorted({str(v).strip() for v in train_df["family_history"].dropna().astype(str).values}),
    }

    return specs


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    if image.mode != 'L':
        image = image.convert('L')
    
    return transform(image)


def assess_lung_ct_image(image: Image.Image):
    """Heuristic check for whether an input image looks like a lung CT slice.

    This is a soft validation for demo safety messaging only, not a medical validator.
    """
    issues: list[str] = []
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)

    if arr.ndim != 2:
        issues.append("Anh khong o dang grayscale 2D.")
        return False, issues

    h, w = arr.shape
    if min(h, w) < 128:
        issues.append("Do phan giai thap, de mat thong tin cau truc pho.")

    p05 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    contrast = p95 - p05
    if contrast < 25:
        issues.append("Do tuong phan qua thap, co the khong phai lat CT pho hop le.")

    dark_fraction = float((arr < 20).mean())
    mid_fraction = float(((arr >= 20) & (arr <= 230)).mean())
    if dark_fraction < 0.01:
        issues.append("Phan nen toi rat it, hinh co the khong giong CT nguc.")
    if mid_fraction < 0.35:
        issues.append("Phan bo muc xam bat thuong, co the la anh ngoai mien CT phoi.")

    # Very low gradient images are often blank/flat screenshots.
    gy = np.abs(np.diff(arr, axis=0)).mean() if h > 1 else 0.0
    gx = np.abs(np.diff(arr, axis=1)).mean() if w > 1 else 0.0
    grad_mean = float((gx + gy) / 2.0)
    if grad_mean < 2.0:
        issues.append("Anh qua phang/it chi tiet, co the khong phu hop de danh gia.")

    train_bounds = load_training_image_validation_stats()
    if train_bounds is not None:
        # Reject samples that are too far outside the observed training image domain.
        domain_checks = {
            "height": float(h),
            "width": float(w),
            "contrast": float(contrast),
            "dark_fraction": float(dark_fraction),
            "mid_fraction": float(mid_fraction),
            "grad_mean": float(grad_mean),
            "mean_intensity": float(arr.mean()),
            "std_intensity": float(arr.std()),
        }
        for name, value in domain_checks.items():
            bound = train_bounds.get(name)
            if bound is None:
                continue
            if value < bound["low"] or value > bound["high"]:
                issues.append(
                    f"Thong so {name}={value:.2f} ngoai mien huan luyen ({bound['low']:.2f} - {bound['high']:.2f})."
                )

    is_suitable = len(issues) == 0
    return is_suitable, issues


def preprocess_tabular(age, sex, smoking_status, tumor_size, family_history, symptom_score):
    """Preprocess tabular features."""
    encoders, scaler = load_tabular_preprocessors()
    # IMPORTANT: use the same column order that MultimodalDataset._encode_tabular
    # used during training: [age, sex, smoking_status, tumor_size, family_history, symptom_score]
    raw_features = []

    # age (numeric)
    try:
        raw_features.append(float(age))
    except (ValueError, TypeError):
        raw_features.append(0.0)

    # sex (categorical -> encoded)
    value_str = str(sex).strip()
    encoder = encoders.get("sex")
    if encoder is not None and value_str in encoder.classes_:
        raw_features.append(float(encoder.transform([value_str])[0]))
    else:
        raw_features.append(0.0)

    # smoking_status (categorical -> encoded)
    value_str = str(smoking_status).strip()
    encoder = encoders.get("smoking_status")
    if encoder is not None and value_str in encoder.classes_:
        raw_features.append(float(encoder.transform([value_str])[0]))
    else:
        raw_features.append(0.0)

    # tumor_size (numeric)
    try:
        raw_features.append(float(tumor_size))
    except (ValueError, TypeError):
        raw_features.append(0.0)

    # family_history (categorical -> encoded)
    value_str = str(family_history).strip()
    encoder = encoders.get("family_history")
    if encoder is not None and value_str in encoder.classes_:
        raw_features.append(float(encoder.transform([value_str])[0]))
    else:
        raw_features.append(0.0)

    # symptom_score (numeric)
    try:
        raw_features.append(float(symptom_score))
    except (ValueError, TypeError):
        raw_features.append(0.0)

    # Convert to array with shape (1, -1) and apply scaler (scaler was fit on
    # categorical-then-numeric order during training, so we keep scaler usage
    # consistent with that fit but pass the same ordering used in training transforms)
    features_array = np.array(raw_features, dtype=np.float32).reshape(1, -1)
    try:
        scaled = scaler.transform(features_array)[0]
    except Exception:
        # If scaler fails for any reason, fall back to identity scaling
        scaled = features_array[0]

    return torch.tensor(scaled, dtype=torch.float32)


def validate_tabular_inputs(age, sex, smoking_status, tumor_size, family_history, symptom_score):
    """Validate required tabular fields and reject values outside the training domain."""
    issues = []

    required_fields = {
        "age": age,
        "sex": sex,
        "smoking_status": smoking_status,
        "tumor_size": tumor_size,
        "family_history": family_history,
        "symptom_score": symptom_score,
    }
    for field_name, value in required_fields.items():
        if value is None or str(value).strip() == "":
            issues.append(f"Truong bat buoc '{field_name}' khong duoc de trong.")

    if issues:
        return False, issues

    specs = load_training_tabular_validation_specs()
    if specs is None:
        return True, issues

    def _check_numeric(field_name, value, low_key="low", high_key="high"):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            issues.append(f"Truong '{field_name}' khong hop le.")
            return

        bound = specs.get(field_name)
        if bound is None:
            return
        if numeric_value < bound[low_key] or numeric_value > bound[high_key]:
            issues.append(
                f"Truong '{field_name}'={numeric_value:.2f} ngoai mien huan luyen ({bound[low_key]:.2f} - {bound[high_key]:.2f})."
            )

    _check_numeric("age", age)
    _check_numeric("tumor_size", tumor_size)
    _check_numeric("symptom_score", symptom_score)

    sex_value = str(sex).strip()
    if sex_value not in specs.get("allowed_sex", []):
        issues.append(f"Gioi tinh '{sex_value}' khong nam trong tap huan luyen.")

    smoking_value = str(smoking_status).strip()
    if smoking_value not in specs.get("allowed_smoking_status", []):
        issues.append(f"Tinh trang hut thuoc '{smoking_value}' khong nam trong tap huan luyen.")

    family_value = str(family_history).strip()
    if family_value not in specs.get("allowed_family_history", []):
        issues.append(f"Tien su gia dinh '{family_value}' khong nam trong tap huan luyen.")

    return len(issues) == 0, issues


def predict_debug(model, device, image_tensor, tabular_tensor):
    """Predict and return intermediate tabular vectors and per-branch contributions.

    Returns a dict with keys: raw_tabular_vector, scaled_tabular_vector,
    image_features, tabular_features, probs_fused, probs_image_only, probs_tabular_only, logits_fused
    """
    with torch.no_grad():
        img = image_tensor.unsqueeze(0).to(device)
        tab = tabular_tensor.unsqueeze(0).to(device)

        # Get branch features
        image_features = model.image_branch(img)
        tabular_features = model.tabular_branch(tab)

        # Fused logits
        gate = model.tabular_gate(tabular_features)
        gated_image = image_features * gate
        combined = torch.cat([gated_image, tabular_features], dim=1)
        logits_fused = model.fusion(combined)

        # Apply temperature scaling if available before softmax
        temperature = load_temperature()
        scaled_logits_fused = logits_fused / temperature
        probs_fused = torch.softmax(scaled_logits_fused, dim=1)[0].cpu().numpy()

        # Approximate image-only contribution: zero tabular features
        zeros_tab = torch.zeros_like(tabular_features)
        logits_image_only = model.fusion(torch.cat([image_features, zeros_tab], dim=1))
        probs_image_only = torch.softmax(logits_image_only / load_temperature(), dim=1)[0].cpu().numpy()

        # Approximate tabular-only contribution: zero image features
        zeros_img = torch.zeros_like(image_features)
        logits_tabular_only = model.fusion(torch.cat([zeros_img, tabular_features], dim=1))
        probs_tabular_only = torch.softmax(logits_tabular_only / load_temperature(), dim=1)[0].cpu().numpy()

    return {
        "image_features": image_features.cpu().numpy()[0],
        "tabular_features": tabular_features.cpu().numpy()[0],
        "logits_fused": logits_fused.cpu().numpy()[0],
        "probs_fused": probs_fused,
        "probs_image_only": probs_image_only,
        "probs_tabular_only": probs_tabular_only,
    }


def predict(model, device, image_tensor, tabular_tensor):
    """Make prediction using the model."""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        tabular_tensor = tabular_tensor.unsqueeze(0).to(device)
        
        logits = model(image_tensor, tabular_tensor)
        # Apply temperature scaling if available
        temperature = load_temperature()
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)[0]

        pred_class = torch.argmax(scaled_logits, dim=1).item()
        pred_prob = probs[pred_class].item()
        low_prob = probs[0].item()
        high_prob = probs[1].item()
        
    return pred_class, pred_prob, low_prob, high_prob


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("# Hệ thống đánh giá nguy cơ CT")
    st.markdown("### Mô hình học sâu đa phương thức (ảnh CT + dữ liệu lâm sàng)")
    st.markdown("---")
    st.warning(get_research_notice())
    st.error(
        "Hệ thống này chỉ dùng cho mục đích nghiên cứu. Kết quả ở nhóm medium-risk có thể lệch và không ổn định, "
        "vì vậy không được dùng làm căn cứ chẩn đoán hay quyết định điều trị."
    )

    if "sample_seed" not in st.session_state:
        st.session_state.sample_seed = 42
    
    # Load model
    with st.spinner("Dang tai mo hinh..."):
        model, device = load_model()
    
    st.success("Đã tải mô hình multimodal_improved_gating thành công!")
    
    # Create two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    # ========================================================================
    # LEFT COLUMN: IMAGE INPUT
    # ========================================================================
    with col1:
        st.subheader("Ảnh CT")
        
        image_input_method = st.radio(
            "Chọn cách nhập ảnh:",
            ["Tải ảnh lên", "Dùng ảnh mẫu"],
            key="image_method"
        )
        
        image = None
        
        if image_input_method == "Tải ảnh lên":
            uploaded_file = st.file_uploader(
                "Tải lên ảnh CT xám",
                type=["jpg", "jpeg", "png", "tiff", "tif"],
                help="Tải ảnh CT định dạng xám"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                if image.mode != 'L':
                    image = image.convert('L')
        
        else:  # Dung anh mau
            if st.button("Lấy 10 mẫu ngẫu nhiên khác", use_container_width=True):
                st.session_state.sample_seed = int(np.random.randint(0, 1_000_000))

            sample_cases = get_random_test_samples(st.session_state.sample_seed, n_samples=10)
            if sample_cases:
                selected_sample_index = st.selectbox(
                    "Chọn 1 trong 10 mẫu test",
                    options=list(range(len(sample_cases))),
                    format_func=lambda idx: f"Mẫu {idx + 1}: {Path(sample_cases[idx]['image_path']).name}",
                    key="sample_case_index",
                )
                selected_sample = sample_cases[selected_sample_index]
                image = load_sample_image(str(selected_sample["image_path"]))

                if image is not None:
                    st.info("Đang dùng 1 mẫu ngẫu nhiên trong 10 mẫu test")
                    st.caption(
                        f"Nhap goi y tu mau nay: age={selected_sample['age']}, sex={selected_sample['sex']}, "
                        f"smoking={selected_sample['smoking_status']}, tumor_size={selected_sample['tumor_size']} mm, "
                        f"family_history={selected_sample['family_history']}, symptom_score={selected_sample['symptom_score']}"
                    )
                    st.session_state.demo_defaults = {
                        "age": int(selected_sample["age"]),
                        "sex": str(selected_sample["sex"]),
                        "smoking_status": str(selected_sample["smoking_status"]),
                        "tumor_size": float(selected_sample["tumor_size"]),
                        "family_history": str(selected_sample["family_history"]),
                        "symptom_score": int(selected_sample["symptom_score"]),
                    }
                else:
                    st.warning("Không tải được ảnh mẫu")
            else:
                st.warning("Không tìm thấy mẫu test để hiển thị")
        
        if image is not None:
            st.image(image, use_column_width=True, caption="Ảnh CT đầu vào")
            st.success("Đã tải ảnh")

            suitable, image_issues = assess_lung_ct_image(image)
            if not suitable:
                st.warning(
                    "Cảnh báo đầu vào: ảnh có dấu hiệu không phải lát CT phổi điển hình. "
                    "Kết quả dự đoán có thể không đáng tin cậy và không phù hợp cho diễn giải."
                )
                with st.expander("Chi tiết cảnh báo đầu vào"):
                    for issue in image_issues:
                        st.write(f"- {issue}")
            else:
                st.caption("Ảnh đầu vào đạt tiêu chí cơ bản cho demo CT phổi.")
        else:
            st.info("Vui lòng tải ảnh lên hoặc dùng ảnh mẫu để tiếp tục")
    
    # ========================================================================
    # RIGHT COLUMN: CLINICAL DATA INPUT
    # ========================================================================
    with col2:
        st.subheader("Thông tin lâm sàng")

        demo_defaults = st.session_state.get("demo_defaults", {
            "age": 60,
            "sex": "female",
            "smoking_status": "never",
            "tumor_size": 20.0,
            "family_history": "no",
            "symptom_score": 5,
        })
        
        with st.form("clinical_form"):
            age = st.slider(
                "Tuổi",
                min_value=20,
                max_value=90,
                value=int(demo_defaults["age"]),
                step=1,
                help="Tuổi bệnh nhân"
            )
            
            sex = st.selectbox(
                "Giới tính",
                ["female", "male"],
                index=0 if demo_defaults["sex"] == "female" else 1,
                help="Giới tính bệnh nhân"
            )
            
            smoking_status = st.selectbox(
                "Tình trạng hút thuốc",
                ["never", "former", "current"],
                index=["never", "former", "current"].index(str(demo_defaults["smoking_status"])) if str(demo_defaults["smoking_status"]) in ["never", "former", "current"] else 0,
                help="Tiền sử hút thuốc"
            )
            
            tumor_size = st.slider(
                "Kích thước khối u (mm)",
                min_value=2.0,
                max_value=40.0,
                value=float(demo_defaults["tumor_size"]),
                step=0.1,
                help="Kích thước khối u tính theo mm"
            )
            
            family_history = st.selectbox(
                "Tiền sử gia đình có ung thư",
                ["no", "yes"],
                index=0 if demo_defaults["family_history"] == "no" else 1,
                help="Gia đình có tiền sử ung thư hay không"
            )
            
            symptom_score = st.slider(
                "Điểm triệu chứng (1-10)",
                min_value=1,
                max_value=10,
                value=int(demo_defaults["symptom_score"]),
                step=1,
                help="Mức độ triệu chứng (1=nhẹ, 10=nặng)"
            )
            
            submit_button = st.form_submit_button(
                "Đánh giá nguy cơ",
                use_container_width=True
            )
    
    # ========================================================================
    # PREDICTION
    # ========================================================================
    if submit_button:
        if image is None:
            st.error("Vui lòng tải ảnh trước")
        else:
            with st.spinner("Đang xử lý và dự đoán..."):
                try:
                    # Preprocess inputs
                    suitable, image_issues = assess_lung_ct_image(image)
                    if not suitable:
                        st.error("Ảnh đầu vào bị từ chối vì không hợp lệ hoặc nằm ngoài miền dữ liệu huấn luyện.")
                        with st.expander("Chi tiết lý do từ chối"):
                            for issue in image_issues:
                                st.write(f"- {issue}")
                        st.stop()

                    tabular_ok, tabular_issues = validate_tabular_inputs(
                        age, sex, smoking_status, tumor_size, family_history, symptom_score
                    )
                    if not tabular_ok:
                        st.error("Dữ liệu lâm sàng bị từ chối vì không hợp lệ hoặc nằm ngoài miền dữ liệu huấn luyện.")
                        with st.expander("Chi tiết lý do từ chối dữ liệu lâm sàng"):
                            for issue in tabular_issues:
                                st.write(f"- {issue}")
                        st.stop()

                    image_tensor = preprocess_image(image)
                    tabular_tensor = preprocess_tabular(
                        age, sex, smoking_status, tumor_size, family_history, symptom_score
                    )
                    
                    # Make prediction
                    pred_class, pred_prob, low_prob, high_prob = predict(
                        model, device, image_tensor, tabular_tensor
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Kết quả dự đoán")
                    st.caption(
                        "Kết quả này chỉ nên dùng để tham khảo nghiên cứu. "
                        "Nếu hồ sơ nằm ở vùng nguy cơ trung gian, dự đoán có thể không ổn định."
                    )
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        if pred_class == 0:
                            st.markdown("## Ước lượng nguy cơ ác tính thấp")
                            st.success(f"Độ tin cậy: {low_prob*100:.2f}%")
                        else:
                            st.markdown("## Ước lượng nguy cơ ác tính cao")
                            st.error(f"Độ tin cậy: {high_prob*100:.2f}%")
                    
                    with result_col2:
                        # Probability bars
                        st.markdown("### Phân bố xác suất")
                        
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            st.metric("Thấp", f"{low_prob*100:.1f}%")
                        with col_prob2:
                            st.metric("Cao", f"{high_prob*100:.1f}%")
                    
                    # Detailed metrics
                    st.markdown("---")
                    st.subheader("Phân tích chi tiết")
                    
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
                    with metrics_col1:
                        st.metric("Tuổi", f"{age}")
                    
                    with metrics_col2:
                        st.metric("Kích thước khối u", f"{tumor_size} mm")
                    
                    with metrics_col3:
                        st.metric("Điểm triệu chứng", f"{symptom_score}/10")
                    
                    with metrics_col4:
                        st.metric("Hút thuốc", smoking_status.capitalize())
                    
                    # Summary
                    st.markdown("---")
                    st.subheader("Tổng kết nghiên cứu")
                    
                    summary_text = f"""
                    **Hồ sơ bệnh nhân:**
                    - Tuoi: {age}
                    - Gioi tinh: {sex.capitalize()}
                    - Hut thuoc: {smoking_status.capitalize()}
                    - Kich thuoc khoi u: {tumor_size} mm
                    - Tien su gia dinh: {family_history.capitalize()}
                    - Diem trieu chung: {symptom_score}/10
                    
                    **Uoc luong cua mo hinh:**
                    - Nhan uoc luong: {'Nguy co cao' if pred_class == 1 else 'Nguy co thap'}
                    - Do tin cay: {pred_prob*100:.2f}%
                    - Xac suat ac tinh thap: {low_prob*100:.2f}%
                    - Xac suat ac tinh cao: {high_prob*100:.2f}%
                    
                    **Thong tin mo hinh:**
                    - Loai mo hinh: Hoc sau da phuong thuc
                    - Dau vao: Anh CT + dac trung lam sang
                    - File dung de demo: multimodal_model_improved_gating.pth
                    - Luu y: ket qua khong thay the danh gia lam sang
                    """
                    
                    st.info(summary_text)
                
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {str(e)}")
                    st.error("Vui long kiem tra du lieu dau vao va thu lai")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    ### Gioi thieu mo hinh
    - **Kien truc:** Hoc sau da phuong thuc (CNN cho anh + MLP cho du lieu bang)
    - **Du lieu huan luyen:** Tap LIDC-IDRI
    - **Muc tieu:** Ho tro nghien cuu va tham khao tren anh CT ket hop thong tin lam sang
    - **Luu y:** Mo hinh chi phu hop cho muc dich nghien cuu; nhom medium-risk co han che va can duoc dien giai than trong
    
    **Lưu ý:** Đây là demo phục vụ học tập/nghiên cứu, không dùng cho chẩn đoán lâm sàng.
    """)


if __name__ == "__main__":
    main()
