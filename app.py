"""Streamlit deployment for AI-assisted skin-lesion analysis.

Pipeline: RGB image -> U-Net segmentation -> padded masked crop -> DenseNet201
classification. The app also generates Grad-CAM and non-diagnostic visual
descriptors. It is a research demo, not a medical device.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


APP_DIR = Path(__file__).resolve().parent
SEGMENTATION_MODEL = APP_DIR / "unet_efficientnetv2s_final.keras"
CLASSIFICATION_MODEL = APP_DIR / "densenet201_best.h5"
LOGO_PATH = APP_DIR / "assets" / "cutivanta-logo.png"

BRAND_NAME = "CutiVanta Lens"
MODEL_VERSION = "U-Net EfficientNetV2S + DenseNet201 / 1.0"
APPLICATION_VERSION = "1.1.0"
RESEARCH_CAUTION = (
    "This prototype was developed using HAM10000-style dermoscopic images. Performance may be lower "
    "for ordinary smartphone photographs, poor-quality images, uncommon conditions, multiple lesions, "
    "images containing artefacts or populations insufficiently represented in the training data. "
    "A high model score does not confirm a diagnosis."
)

SEG_SIZE = (256, 256)
CLS_SIZE = (224, 224)
MASK_THRESHOLD = 0.50
CLASS_CODES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
CLASS_NAMES = {
    "AKIEC": "Actinic keratosis / intraepithelial carcinoma",
    "BCC": "Basal cell carcinoma",
    "BKL": "Benign keratosis-like lesion",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "VASC": "Vascular lesion",
}
DISEASE_GUIDE = {
    "AKIEC": {
        "about": "This training label combines actinic keratosis with intraepithelial keratinocyte carcinoma. Actinic keratosis is a UV-damaged precancerous growth; an image model cannot determine whether atypical cells are confined to the surface.",
        "care": "A dermatologist first confirms the diagnosis and checks for invasive cancer. For confirmed actinic keratosis, options depend on number, site, thickness, previous skin cancer, and general health. Clinician-selected options include cryosurgery, curettage, photodynamic therapy, or prescription field treatments such as 5-fluorouracil, imiquimod, diclofenac, or tirbanibulin. Do not start these medicines from an AI result.",
        "urgency": "Book a dermatologist review, especially if the area is thick, tender, enlarging, ulcerated, or bleeding.",
        "source": "American Academy of Dermatology - Actinic keratosis: diagnosis and treatment",
        "url": "https://www.aad.org/public/diseases/skin-cancer/actinic-keratosis-treatment",
    },
    "BCC": {
        "about": "Basal cell carcinoma is a skin cancer that usually grows slowly and rarely spreads to distant organs, but untreated disease can grow deeper and damage nearby tissue.",
        "care": "Diagnosis generally requires dermatologist assessment and often biopsy. Treatment is selected from pathology, clinical size, anatomical site, depth, recurrence risk, age, and health. Surgery with margin assessment or Mohs surgery is the main treatment for many confirmed BCCs. Curettage/electrodesiccation, radiation, cryosurgery, photodynamic therapy, or topical medicines are reserved for selected situations; nonsurgical approaches generally have lower cure rates than surgery.",
        "urgency": "Arrange prompt dermatology assessment rather than attempting home treatment.",
        "source": "American Academy of Dermatology - Basal cell carcinoma guideline",
        "url": "https://www.aad.org/member/clinical-quality/guidelines/bcc",
    },
    "BKL": {
        "about": "BKL is a broad dataset label for benign keratosis-like lesions; it can include appearances such as seborrheic keratosis and related benign keratin growths. The model does not identify an exact subtype.",
        "care": "A typical confirmed seborrheic keratosis is harmless and usually needs no treatment. A dermatologist may remove one that is irritated, catches on clothing, is cosmetically unwanted, or resembles skin cancer. Clinician options include cryosurgery, curettage, shave removal, or electrosurgery. Diagnostic uncertainty may require removal and microscopic examination.",
        "urgency": "Seek review if it is new, rapidly changing, bleeding, painful, or unlike your other spots.",
        "source": "American Academy of Dermatology - Seborrheic keratoses: diagnosis and treatment",
        "url": "https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-treatment",
    },
    "DF": {
        "about": "Dermatofibroma is usually a harmless fibrous skin nodule. A photograph can resemble other growths, so recent enlargement, ulceration, or unusual asymmetry needs clinical review.",
        "care": "A clinically confirmed dermatofibroma usually needs reassurance only. If it causes persistent symptoms or diagnostic concern, complete surgical removal may be considered, but it leaves a scar and incomplete removal can recur. Atypical lesions may need biopsy or diagnostic excision.",
        "urgency": "Routine review is reasonable; seek earlier review for recent growth, ulceration, pain, or marked change.",
        "source": "DermNet - Dermatofibroma",
        "url": "https://dermnetnz.org/topics/dermatofibroma",
    },
    "MEL": {
        "about": "Melanoma is a potentially serious skin cancer because it can invade and spread. An image classification is not a diagnosis; biopsy and pathology are required to confirm melanoma and determine its characteristics.",
        "care": "A qualified dermatologist may perform clinical and dermoscopic assessment and, when indicated, recommend biopsy or other investigations. If melanoma is confirmed, management depends on pathology, stage, Breslow thickness, ulceration, lymph-node findings, spread, and overall health—not the percentage of this photograph covered by the mask.",
        "urgency": "Arrange prompt dermatologist assessment; do not wait for another AI image result if the lesion is changing, bleeding, or otherwise concerning.",
        "source": "American Academy of Dermatology - Melanoma diagnosis and treatment",
        "url": "https://www.aad.org/public/diseases/skin-cancer/types/common/melanoma/diagnose-treat",
    },
    "NV": {
        "about": "A melanocytic nevus is a mole. Most moles are benign and require no treatment, but a photograph cannot reliably distinguish every atypical mole from melanoma.",
        "care": "A stable, clinically benign mole usually needs no treatment. A dermatologist may remove a mole that is suspicious or persistently bothersome using surgical excision or shave removal, with microscopic examination when appropriate. Never cut, burn, or use an online mole-removal product at home because this can scar the skin and delay cancer diagnosis.",
        "urgency": "Arrange review for a new or changing mole, asymmetry, irregular border, changing colour, symptoms, or an appearance different from your other moles.",
        "source": "American Academy of Dermatology - Moles: diagnosis and treatment",
        "url": "https://www.aad.org/public/diseases/a-z/moles-treatment",
    },
    "VASC": {
        "about": "VASC is a broad image label covering vascular-appearing lesions rather than one disease. Examples can include angiomas, angiokeratomas, or pyogenic granuloma, which have different behavior and treatment.",
        "care": "Treatment cannot be chosen until the vascular subtype is confirmed. Some angiomas and angiokeratomas need no treatment. A lesion that bleeds, grows rapidly, hurts, or is diagnostically uncertain may need dermoscopy, biopsy, cautery/curettage, excision, cryotherapy, or vascular laser selected by a clinician. Pyogenic granuloma can bleed significantly and often needs clinician-directed treatment.",
        "urgency": "Seek timely review for rapid growth, recurrent bleeding, ulceration, pain, or uncertain diagnosis.",
        "source": "DermNet - Vascular proliferations and abnormalities of blood vessels",
        "url": "https://dermnetnz.org/topics/vascular-proliferations-and-abnormalities-of-blood-vessels",
    },
}


st.set_page_config(
    page_title=BRAND_NAME,
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#17211f; --muted:#64716e; --teal:#087f72; --aqua:#35c6ae; --violet:#7668ed; --paper:#f4f8f7; }
        .stApp {
            background:
              radial-gradient(circle at 82% 0%, rgba(118,104,237,.13), transparent 29rem),
              radial-gradient(circle at 3% 34%, rgba(53,198,174,.14), transparent 31rem),
              linear-gradient(180deg,#fbfdfc 0%,#f2f8f6 100%);
            color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif;
        }
        .block-container { max-width:1240px; padding-top:2rem; padding-bottom:4rem; }
        h1,h2,h3 { color:var(--ink) !important; letter-spacing:-.035em; }
        [data-testid="stSidebar"] { display:none; }
        [data-testid="stFileUploader"] { background:rgba(255,255,255,.86); border:1.5px dashed rgba(8,127,114,.48); border-radius:22px; padding:16px; box-shadow:0 16px 45px rgba(26,75,66,.06); }
        [data-testid="stMetric"] { background:rgba(255,255,255,.9); border:1px solid rgba(8,127,114,.12); border-radius:18px; padding:14px 16px; box-shadow:0 10px 30px rgba(28,70,59,.05); }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        .hero { background:linear-gradient(125deg,#092f2c 0%,#087f72 58%,#6559d8 135%); color:white; border-radius:32px; padding:42px 44px; margin:2px 0 28px; box-shadow:0 24px 70px rgba(9,47,44,.18); position:relative; overflow:hidden; }
        .hero:before,.hero:after { content:''; position:absolute; border-radius:50%; border:1px solid rgba(255,255,255,.15); }
        .hero:before { width:310px; height:310px; right:-75px; top:-135px; }
        .hero:after { width:205px; height:205px; right:35px; top:-82px; }
        .brand { display:inline-flex; gap:12px; align-items:center; font-size:.8rem; letter-spacing:.16em; text-transform:uppercase; font-weight:750; opacity:.95; }
        .brand-logo { width:48px; height:48px; object-fit:contain; padding:5px; border-radius:14px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18); }
        .hero h1 { color:white !important; font-size:clamp(2.35rem,5vw,4.45rem); margin:.55rem 0 .45rem; max-width:840px; line-height:1.02; }
        .hero p { max-width:670px; opacity:.86; font-size:1.04rem; line-height:1.65; margin:0; }
        .upload-title { text-align:center; font-size:1.28rem; font-weight:750; margin:4px 0 3px; }
        .upload-sub { text-align:center; color:var(--muted); margin:0 0 16px; }
        .panel-label { font-size:.72rem; letter-spacing:.13em; text-transform:uppercase; font-weight:800; color:var(--teal); margin-bottom:7px; }
        .result-card { background:linear-gradient(145deg,#fff,#ecf8f5); border:1px solid rgba(8,127,114,.16); border-radius:24px; padding:24px 25px; box-shadow:0 14px 38px rgba(24,83,68,.08); }
        .result-code { display:inline-block; color:#fff; background:linear-gradient(90deg,#087f72,#6559d8); border-radius:999px; padding:5px 11px; font-size:.75rem; font-weight:800; }
        .result-name { font-family:'Manrope'; font-size:1.48rem; font-weight:800; line-height:1.2; margin:12px 0 6px; }
        .soft-note { color:#60716c; font-size:.88rem; line-height:1.55; }
        .status-good,.status-warn { border-radius:14px; padding:12px 14px; font-size:.9rem; }
        .status-good { background:#e5f5ef; color:#14614f; border:1px solid #bce3d5; }
        .status-warn { background:#fff4e6; color:#85551f; border:1px solid #f2d6ad; }
        div[data-testid="stImage"] img { border-radius:18px; }
        .disclaimer { background:#fff8ee; border-left:4px solid #e69b4c; border-radius:10px; padding:13px 15px; color:#654c31; font-size:.86rem; }
        .descriptor { background:#fff; border:1px solid #dfeae7; border-radius:16px; padding:13px 15px; margin:9px 0; }
        .descriptor-line { height:7px; border-radius:99px; background:#e8efed; margin-top:8px; overflow:hidden; }
        .descriptor-fill { height:100%; background:#20ad98; border-radius:99px; }
        .section-rule { height:1px; background:linear-gradient(90deg,transparent,#b8d2cb,transparent); margin:34px 0; }
        .copyright { text-align:center; color:var(--muted); font-size:.78rem; margin:28px 0 4px; }
        div[data-baseweb="tab-list"] { gap:6px; background:#e7f0ed; padding:5px; border-radius:15px; }
        button[data-baseweb="tab"] { flex:1; border-radius:11px; min-height:42px; font-weight:750; }
        button[data-baseweb="tab"][aria-selected="true"] { background:white; color:var(--teal); box-shadow:0 3px 12px rgba(20,65,56,.12); }
        .stButton>button, .stDownloadButton>button { width:100%; border-radius:14px; min-height:48px; font-weight:750; border:0; background:linear-gradient(90deg,#087f72,#6559d8); color:#fff; box-shadow:0 10px 25px rgba(8,127,114,.18); }
        .stButton>button:hover, .stDownloadButton>button:hover { color:#fff; border:0; transform:translateY(-1px); }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[tf.keras.Model, tf.keras.Model]:
    missing = [str(p.name) for p in (SEGMENTATION_MODEL, CLASSIFICATION_MODEL) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model file(s): {', '.join(missing)}")
    seg_model = tf.keras.models.load_model(SEGMENTATION_MODEL, compile=False)
    cls_model = tf.keras.models.load_model(CLASSIFICATION_MODEL, compile=False)
    if tuple(seg_model.input_shape[1:]) != (256, 256, 3):
        raise ValueError(f"Unexpected segmentation input shape: {seg_model.input_shape}")
    if tuple(cls_model.input_shape[1:]) != (224, 224, 3) or cls_model.output_shape[-1] != 7:
        raise ValueError(f"Unexpected classification model shape: {cls_model.input_shape} -> {cls_model.output_shape}")
    return seg_model, cls_model


def read_image(uploaded_file: Any) -> tuple[np.ndarray, dict[str, Any]]:
    image = Image.open(uploaded_file)
    original_mode = image.mode
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    metadata = {
        "original_mode": original_mode,
        "aspect_ratio": width / max(height, 1),
    }
    if max(image.size) > 2400:
        image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
    return np.asarray(image), metadata


def largest_component(binary_mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if count <= 1:
        return binary_mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def segment_lesion(model: tf.keras.Model, image_rgb: np.ndarray) -> dict[str, Any]:
    resized = cv2.resize(image_rgb, SEG_SIZE, interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32)[None, ...] / 255.0
    probability = np.squeeze(model.predict(tensor, verbose=0)).astype(np.float32)
    binary = (probability >= MASK_THRESHOLD).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum_candidate_area = binary.size * 0.005
    candidate_count = int(sum(area >= minimum_candidate_area for area in stats[1:, cv2.CC_STAT_AREA]))
    binary = largest_component(binary)

    height, width = image_rgb.shape[:2]
    mask = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    probability_full = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    coverage = float(mask.mean())
    if np.any(mask):
        foreground_certainty = float(probability_full[mask == 1].mean())
        background_certainty = float((1.0 - probability_full[mask == 0]).mean()) if np.any(mask == 0) else foreground_certainty
        certainty = (foreground_certainty + background_certainty) / 2.0
    else:
        certainty = float((1.0 - probability_full).mean())

    return {
        "probability": probability_full,
        "mask": mask,
        "coverage": coverage,
        "certainty": certainty,
        "foreground_probability": foreground_certainty if np.any(mask) else 0.0,
        "candidate_count": candidate_count,
    }


def padded_masked_crop(image_rgb: np.ndarray, mask: np.ndarray, padding: float = 0.12) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    points = cv2.findNonZero(mask.astype(np.uint8))
    height, width = image_rgb.shape[:2]
    if points is None:
        return image_rgb.copy(), (0, 0, width, height)
    x, y, w, h = cv2.boundingRect(points)
    pad = max(8, int(max(w, h) * padding))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
    masked = image_rgb * mask[..., None]
    return masked[y0:y1, x0:x1], (x0, y0, x1, y1)


def classify(model: tf.keras.Model, crop_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Matches the training notebook's ImageDataGenerator(rescale=1./255).
    resized = cv2.resize(crop_rgb, CLS_SIZE, interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32)[None, ...] / 255.0
    probabilities = np.asarray(model.predict(tensor, verbose=0)[0], dtype=np.float32)
    probabilities = probabilities / max(float(probabilities.sum()), 1e-8)
    return probabilities, tensor


def gradcam(model: tf.keras.Model, tensor: np.ndarray, class_index: int) -> np.ndarray:
    # Keras 3 exposes symbolic KerasTensor objects here (not tf.Tensor), so
    # select by rank rather than concrete tensor type.
    spatial_layer = next(
        layer for layer in reversed(model.layers)
        if getattr(getattr(layer, "output", None), "shape", None) is not None
        and len(layer.output.shape) == 4
    )
    gradient_model = tf.keras.Model(model.inputs, [spatial_layer.output, model.outputs[0]])
    with tf.GradientTape() as tape:
        features, predictions = gradient_model([tensor], training=False)
        score = predictions[:, class_index]
    gradients = tape.gradient(score, features)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients were unavailable.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * features, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    return (heatmap / (maximum + tf.keras.backend.epsilon())).numpy()


def heatmap_overlay(crop_rgb: np.ndarray, heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heat = cv2.resize(heatmap, (crop_rgb.shape[1], crop_rgb.shape[0]))
    colored = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(crop_rgb, 0.58, colored, 0.42, 0)
    return overlay, heat


def mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    opacity: float = 0.48,
    boundary_thickness: int | None = None,
) -> np.ndarray:
    overlay = image_rgb.copy()
    tint = np.zeros_like(image_rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = 240, 103, 86
    opacity = float(np.clip(opacity, 0.0, 1.0))
    selected = cv2.addWeighted(image_rgb, 1.0 - opacity, tint, opacity, 0)
    overlay[mask == 1] = selected[mask == 1]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inner = boundary_thickness or max(2, image_rgb.shape[1] // 280)
    cv2.drawContours(overlay, contours, -1, (25, 28, 28), inner + max(2, inner))
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), inner)
    return overlay


def quality_checks(
    image_rgb: np.ndarray,
    segmentation: dict[str, Any],
    image_metadata: dict[str, Any],
) -> tuple[list[str], float]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    coverage = float(segmentation["coverage"])
    mask = segmentation["mask"].astype(bool)
    warnings: list[str] = []
    if focus < 35:
        warnings.append("Blur or poor focus detected")
    dark_fraction = float(np.mean(gray <= 15))
    bright_fraction = float(np.mean(gray >= 245))
    if brightness < 42 or dark_fraction > 0.35:
        warnings.append("Underexposure detected")
    if brightness > 218 or bright_fraction > 0.25:
        warnings.append("Overexposure detected")
    if coverage < 0.01:
        warnings.append("Extremely small lesion or no stable lesion region localized")
    elif coverage > 0.90:
        warnings.append("Mask covers almost the entire image")
    if np.any(mask):
        ys, xs = np.where(mask)
        center_x, center_y = float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])
        if abs(center_x - 0.5) > 0.23 or abs(center_y - 0.5) > 0.23:
            warnings.append("Lesion is not centred")
        edge_y = max(1, int(mask.shape[0] * 0.015))
        edge_x = max(1, int(mask.shape[1] * 0.015))
        if mask[:edge_y].any() or mask[-edge_y:].any() or mask[:, :edge_x].any() or mask[:, -edge_x:].any():
            warnings.append("Lesion touches an image border")
    if int(segmentation.get("candidate_count", 0)) > 1:
        warnings.append("Multiple candidate lesions detected")

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    hair_fraction = float(np.mean(blackhat > 22))
    if hair_fraction > 0.055:
        warnings.append("Possible hair obstruction")

    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=min(gray.shape) * 0.42, maxLineGap=12)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    blue_ink = ((hsv[..., 0] > 85) & (hsv[..., 0] < 140) & (hsv[..., 1] > 80) & (hsv[..., 2] < 210))
    if (lines is not None and len(lines) >= 3) or float(blue_ink.mean()) > 0.008:
        warnings.append("Possible ruler or ink-marker artefact")
    aspect_ratio = float(image_metadata.get("aspect_ratio", 1.0))
    if aspect_ratio > 3.0 or aspect_ratio < 1 / 3:
        warnings.append("Invalid aspect ratio")
    if image_metadata.get("original_mode") not in {"RGB", "RGBA"}:
        warnings.append(f"Unsupported colour mode ({image_metadata.get('original_mode', 'unknown')}); converted to RGB")
    saturation = float(hsv[..., 1].mean())
    if saturation < 12 or (coverage < 0.01 and focus > 80):
        warnings.append("Image may have a non-dermoscopic appearance")
    return warnings, focus


def probability_map_with_colorbar(probability: np.ndarray) -> np.ndarray:
    """Render a Turbo probability map with an embedded 0–1 colour bar."""
    values = np.uint8(np.clip(probability, 0, 1) * 255)
    colored = cv2.cvtColor(cv2.applyColorMap(values, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    bar_width = max(34, colored.shape[1] // 10)
    gradient = np.linspace(255, 0, colored.shape[0], dtype=np.uint8)[:, None]
    bar = cv2.cvtColor(cv2.applyColorMap(np.repeat(gradient, bar_width, axis=1), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    labelled = np.full((colored.shape[0], bar_width + 42, 3), 255, dtype=np.uint8)
    labelled[:, :bar_width] = bar
    for value, y in ((1.0, 14), (0.5, colored.shape[0] // 2), (0.0, colored.shape[0] - 7)):
        cv2.putText(labelled, f"{value:.1f}", (bar_width + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 33, 31), 1, cv2.LINE_AA)
    return np.concatenate([colored, labelled], axis=1)


def zoom_to_mask(image: np.ndarray, mask: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.01 or not np.any(mask):
        return image
    height, width = image.shape[:2]
    ys, xs = np.where(mask > 0)
    center_x, center_y = int(xs.mean()), int(ys.mean())
    crop_w, crop_h = max(1, int(width / zoom)), max(1, int(height / zoom))
    x0 = min(max(0, center_x - crop_w // 2), width - crop_w)
    y0 = min(max(0, center_y - crop_h // 2), height - crop_h)
    return image[y0:y0 + crop_h, x0:x0 + crop_w]


def array_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.uint8(np.clip(image_rgb, 0, 255))).save(buffer, format="PNG")
    return buffer.getvalue()


def visual_descriptors(crop_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return {"border_irregularity": 0.0, "asymmetry": 0.0, "color_variation": 0.0}
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = float(cv2.arcLength(contour, True))
    irregularity = max(0.0, min(1.0, (perimeter * perimeter / (4 * np.pi * area) - 1.0) / 3.0))
    overlap_h = np.logical_and(mask, np.fliplr(mask)).sum() / max(np.logical_or(mask, np.fliplr(mask)).sum(), 1)
    overlap_v = np.logical_and(mask, np.flipud(mask)).sum() / max(np.logical_or(mask, np.flipud(mask)).sum(), 1)
    asymmetry = float(1.0 - (overlap_h + overlap_v) / 2.0)
    pixels = crop_rgb[mask.astype(bool)]
    color_variation = float(np.mean(np.std(pixels, axis=0)) / 128.0) if len(pixels) else 0.0
    return {
        "border_irregularity": irregularity,
        "asymmetry": min(1.0, asymmetry),
        "color_variation": min(1.0, color_variation),
    }


def lesion_location(box: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> str:
    """Describe the detected box location using image-relative thirds."""
    height, width = image_shape[:2]
    center_x = ((box[0] + box[2]) / 2) / max(width, 1)
    center_y = ((box[1] + box[3]) / 2) / max(height, 1)
    horizontal = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
    vertical = "upper" if center_y < 1 / 3 else "lower" if center_y > 2 / 3 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "near the center of the uploaded image"
    return f"in the {vertical}-{horizontal} area of the uploaded image"


def array_to_jpeg_buffer(image_rgb: np.ndarray, quality: int = 90) -> BytesIO:
    buffer = BytesIO()
    Image.fromarray(np.uint8(np.clip(image_rgb, 0, 255))).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return buffer


def mask_extent(mask: np.ndarray) -> dict[str, float | int]:
    """Return image-relative mask measurements without implying clinical size."""
    height, width = mask.shape[:2]
    affected_pixels = int(np.count_nonzero(mask))
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return {"pixels": 0, "total": int(mask.size), "width_fraction": 0.0, "height_fraction": 0.0}
    return {
        "pixels": affected_pixels,
        "total": int(mask.size),
        "width_fraction": float((xs.max() - xs.min() + 1) / max(width, 1)),
        "height_fraction": float((ys.max() - ys.min() + 1) / max(height, 1)),
    }


def coverage_description(coverage: float) -> str:
    if coverage < 0.05:
        return "a small fraction"
    if coverage < 0.20:
        return "a limited fraction"
    if coverage < 0.40:
        return "a substantial fraction"
    return "a large fraction"


def color_palette(image_rgb: np.ndarray, mask: np.ndarray, colors_count: int = 4) -> tuple[np.ndarray, list[float]]:
    """Build a compact dominant-colour strip for visual inspection."""
    pixels = image_rgb[mask.astype(bool)]
    if len(pixels) < colors_count:
        pixels = image_rgb.reshape(-1, 3)
    if len(pixels) > 12000:
        indices = np.linspace(0, len(pixels) - 1, 12000, dtype=int)
        pixels = pixels[indices]
    samples = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.3)
    _compactness, labels, centers = cv2.kmeans(
        samples, colors_count, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=colors_count)
    order = np.argsort(counts)[::-1]
    centers, counts = centers[order], counts[order]
    fractions = counts / max(counts.sum(), 1)
    palette = np.zeros((100, 700, 3), dtype=np.uint8)
    start = 0
    for index, (center, fraction) in enumerate(zip(centers, fractions)):
        end = 700 if index == len(centers) - 1 else start + int(round(700 * fraction))
        palette[:, start:end] = np.uint8(np.clip(center, 0, 255))
        start = end
    return palette, [float(value) for value in fractions]


def build_pdf_report(
    image_rgb: np.ndarray,
    segmented_overlay: np.ndarray,
    predicted_code: str,
    classification_confidence: float,
    mean_foreground_probability: float,
    coverage: float,
    location_text: str,
    warnings: list[str],
    report_id: str,
    prediction_uncertainty: float,
) -> bytes:
    """Create an in-memory, two-image patient-friendly research report."""
    output = BytesIO()
    guide = DISEASE_GUIDE[predicted_code]
    doc = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=42, leftMargin=42,
        topMargin=30, bottomMargin=52, title=f"{BRAND_NAME} Research Report",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PrismTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=19, leading=23, textColor=colors.HexColor("#087F72"),
        alignment=TA_CENTER, spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "PrismSub", parent=styles["Normal"], fontSize=9.5, leading=14,
        textColor=colors.HexColor("#64716E"), alignment=TA_CENTER, spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "PrismHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=colors.HexColor("#17211F"), spaceBefore=8, spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "PrismBody", parent=styles["BodyText"], fontSize=9.5, leading=14,
        textColor=colors.HexColor("#34413E"), spaceAfter=6,
    )
    caution_style = ParagraphStyle(
        "PrismCaution", parent=body_style, fontName="Helvetica-Bold", fontSize=9,
        leading=13, textColor=colors.HexColor("#7B4A15"), alignment=TA_CENTER,
    )

    def footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7E4E0"))
        canvas.line(42, 43, A4[0] - 42, 43)
        canvas.setFillColor(colors.HexColor("#7B4A15"))
        canvas.setFont("Helvetica-Bold", 7.2)
        canvas.drawCentredString(A4[0] / 2, 29, "RESEARCH USE ONLY - professional confirmation is required")
        canvas.setFont("Helvetica", 6.8)
        canvas.drawCentredString(A4[0] / 2, 18, "© 2026 Siba Sundar Das. All rights reserved.")
        canvas.restoreState()

    generated_at = datetime.now().astimezone().strftime("%d %b %Y, %I:%M:%S %p %Z")
    metadata_table = Table([
        ["Report ID", report_id, "Generated date and time", generated_at],
        ["Model version", MODEL_VERSION, "Application version", APPLICATION_VERSION],
        ["Use label", "RESEARCH USE ONLY", "", ""],
    ], colWidths=[0.9 * inch, 2.55 * inch, 1.15 * inch, 2.15 * inch])
    metadata_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F8F6")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#34413E")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7E4E0")),
        ("SPAN", (1, 2), (3, 2)),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    original = PDFImage(array_to_jpeg_buffer(image_rgb), width=2.9 * inch, height=2.0 * inch, kind="proportional")
    segmented = PDFImage(array_to_jpeg_buffer(segmented_overlay), width=2.9 * inch, height=2.0 * inch, kind="proportional")
    image_table = Table(
        [[original, segmented], [Paragraph("<b>Original image</b>", subtitle_style), Paragraph("<b>Model-predicted lesion boundary</b>", subtitle_style)]],
        colWidths=[3.3 * inch, 3.3 * inch], hAlign="CENTER",
    )
    image_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, 0), 0.5, colors.HexColor("#D7E4E0")),
        ("INNERGRID", (0, 0), (-1, 0), 0.5, colors.HexColor("#D7E4E0")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F3F8F6")),
        ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
    ]))

    result_table = Table([
        ["Model-predicted category", CLASS_NAMES[predicted_code]],
        ["Calibrated score", f"{classification_confidence:.1%}"],
        ["Image area covered by predicted mask", f"{coverage:.1%}"],
        ["Mean foreground mask probability", f"{mean_foreground_probability:.1%}"],
        ["Prediction uncertainty", f"{prediction_uncertainty:.1%}"],
    ], colWidths=[2.0 * inch, 4.75 * inch])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F6F2")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#24322F")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CFE0DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    quality_text = "No basic image-quality warnings were triggered." if not warnings else "Quality notes: " + "; ".join(warnings) + "."
    story = [
        Paragraph(f"{BRAND_NAME} — AI-Assisted Skin Lesion Research Report", title_style),
        Paragraph("Research-use-only output requiring professional confirmation", subtitle_style),
        metadata_table,
        Spacer(1, 6),
        image_table,
        Spacer(1, 5),
        Paragraph("Analysis summary", heading_style),
        result_table,
        Spacer(1, 4),
        Paragraph("Where the model found the lesion", heading_style),
        Paragraph(
            f"The segmentation model localized a connected region <b>{location_text}</b>, covering approximately <b>{coverage:.1%}</b> of the image. The colored overlay marks this model-selected region; it does not establish the true clinical boundary.",
            body_style,
        ),
        Paragraph("About the selected category", heading_style),
        Paragraph(
            f"The classifier's highest-scoring category was <b>{CLASS_NAMES[predicted_code]} ({predicted_code})</b> with a model probability of <b>{classification_confidence:.1%}</b>. This category requires clinical confirmation and may be incorrect.",
            body_style,
        ),
        Paragraph("Disease-specific explanation", heading_style),
        Paragraph(guide["about"], body_style),
        Paragraph("Treatment / cure context", heading_style),
        Paragraph(guide["care"], body_style),
        Paragraph(f"<b>Recommended review:</b> {guide['urgency']}", body_style),
        Paragraph(
            f"The highlighted mask occupies {coverage:.1%} of this photograph. This is a 2D image-coverage measurement, not disease stage, depth, body-surface involvement, or a treatment-dose calculation.",
            body_style,
        ),
        Paragraph(f"<b>Research source:</b> {guide['source']}<br/><font size='7'>{guide['url']}</font>", body_style),
        Paragraph(quality_text, body_style),
        Spacer(1, 5),
        Paragraph(
            RESEARCH_CAUTION,
            caution_style,
        ),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def render_ui_footer() -> None:
    st.markdown(f'<div class="disclaimer"><b>Caution:</b> {RESEARCH_CAUTION}</div>', unsafe_allow_html=True)
    st.markdown('<div class="copyright">© 2026 Siba Sundar Das. All rights reserved.</div>', unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <section class="hero">
          <div class="brand"><img class="brand-logo" src="data:image/png;base64,{logo_data}" alt="{BRAND_NAME} logo">{BRAND_NAME}</div>
          <h1>Look closer. Understand better.</h1>
          <p>A focused visual workspace for lesion localization, model evidence, image measurements, and carefully sourced condition information.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="upload-title">Upload an image for analysis</div><div class="upload-sub">Choose one clear, close-up image with the lesion centered and fully visible.</div>', unsafe_allow_html=True)
    _upload_left, upload_center, _upload_right = st.columns([1, 1.65, 1])
    with upload_center:
        uploaded = st.file_uploader(
            "Skin lesion image",
            type=["jpg", "jpeg", "png", "webp"],
            help="For best results: center one lesion, use even lighting, and avoid blur.",
            label_visibility="collapsed",
        )
        analyze = st.button("Upload & analyze image", type="primary", disabled=uploaded is None)
    st.markdown(f'<div class="disclaimer"><b>Research-use-only:</b> {BRAND_NAME} provides an AI-generated educational result, not a medical diagnosis; professional dermatologist review is required.</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.session_state.pop("cutivanta_analysis", None)
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.info("**One lesion**\n\nKeep the lesion centered and fully visible.")
        b.info("**Even light**\n\nAvoid glare, shadows, rulers, or pen markings.")
        c.info("**Sharp image**\n\nUse a close, focused photo with natural skin color.")
        render_ui_footer()
        return
    upload_key = hashlib.sha256(uploaded.getvalue()).hexdigest()
    cached_result = st.session_state.get("cutivanta_analysis")
    if not analyze and (not cached_result or cached_result.get("upload_key") != upload_key):
        st.info("Your image is ready. Select **Upload & analyze image** to begin.")
        render_ui_footer()
        return

    if analyze:
        try:
            image_rgb, image_metadata = read_image(uploaded)
            upload_fingerprint = upload_key[:10]
            with st.spinner(f"{BRAND_NAME} is examining the image..."):
                seg_model, cls_model = load_models()
                segmentation = segment_lesion(seg_model, image_rgb)
                warnings, focus = quality_checks(image_rgb, segmentation, image_metadata)
                crop_rgb, box = padded_masked_crop(image_rgb, segmentation["mask"])
                probabilities, cls_tensor = classify(cls_model, crop_rgb)
                predicted_index = int(np.argmax(probabilities))
                predicted_code = CLASS_CODES[predicted_index]
                prediction_confidence = float(probabilities[predicted_index])
                sorted_probs = np.sort(probabilities)
                margin = float(sorted_probs[-1] - sorted_probs[-2])
                entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-8)) / np.log(len(probabilities)))
                cam = gradcam(cls_model, cls_tensor, predicted_index)
                gradcam_view, heat = heatmap_overlay(crop_rgb, cam)
                segmented_view = mask_overlay(image_rgb, segmentation["mask"])
                location_text = lesion_location(box, image_rgb.shape)
            st.session_state["cutivanta_analysis"] = {
                "upload_key": upload_key,
                "upload_fingerprint": upload_fingerprint,
                "image_rgb": image_rgb,
                "image_metadata": image_metadata,
                "segmentation": segmentation,
                "warnings": warnings,
                "focus": focus,
                "crop_rgb": crop_rgb,
                "box": box,
                "probabilities": probabilities,
                "predicted_index": predicted_index,
                "predicted_code": predicted_code,
                "prediction_confidence": prediction_confidence,
                "margin": margin,
                "entropy": entropy,
                "gradcam_view": gradcam_view,
                "heat": heat,
                "segmented_view": segmented_view,
                "location_text": location_text,
            }
        except Exception as exc:
            st.error("The analysis could not be completed.")
            st.exception(exc)
            st.caption("Check that both model files remain beside app.py and that TensorFlow/Keras versions match requirements.txt.")
            render_ui_footer()
            return
    else:
        result = cached_result
        upload_fingerprint = result["upload_fingerprint"]
        image_rgb = result["image_rgb"]
        image_metadata = result.get("image_metadata", {"original_mode": "RGB", "aspect_ratio": image_rgb.shape[1] / image_rgb.shape[0]})
        segmentation = result["segmentation"]
        warnings = result["warnings"]
        focus = result["focus"]
        crop_rgb = result["crop_rgb"]
        box = result["box"]
        probabilities = result["probabilities"]
        predicted_index = result["predicted_index"]
        predicted_code = result["predicted_code"]
        prediction_confidence = result["prediction_confidence"]
        margin = result["margin"]
        entropy = result["entropy"]
        gradcam_view = result["gradcam_view"]
        heat = result["heat"]
        segmented_view = result["segmented_view"]
        location_text = result["location_text"]

    if warnings:
        st.markdown(f'<div class="status-warn"><b>Image quality note:</b> {" / ".join(warnings)}. The result may be less reliable.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-good"><b>Image check passed.</b> No basic focus, exposure, or localization warnings were triggered.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("## Your image analysis")
    original_col, segment_col, result_col = st.columns([1, 1, 1.05], gap="large")
    with original_col:
        st.markdown('<div class="panel-label">Original image</div>', unsafe_allow_html=True)
        st.image(image_rgb, use_container_width=True)
    with segment_col:
        st.markdown('<div class="panel-label">Model-predicted lesion boundary</div>', unsafe_allow_html=True)
        st.image(segmented_view, use_container_width=True)
        st.caption(f"Localized {location_text}.")
    with result_col:
        st.markdown('<div class="panel-label">Model-predicted category</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-card"><div class="panel-label">Model-predicted category</div><div class="result-name">{CLASS_NAMES[predicted_code]}</div><div class="soft-note">Calibrated score: <b>{prediction_confidence:.1%}</b><br>This research output requires professional confirmation.</div></div>',
            unsafe_allow_html=True,
        )
        st.progress(prediction_confidence)
        st.caption("Model probability is not diagnostic certainty.")

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Image area covered by predicted mask", f"{segmentation['coverage']:.1%}")
    metric2.metric("Mean foreground mask probability", f"{segmentation['foreground_probability']:.1%}", help="Mean model probability within the retained mask; not clinical confidence.")
    metric3.metric("Top predicted-class score", f"{prediction_confidence:.1%}")
    metric4.metric("Prediction uncertainty", f"{entropy:.1%}", help="Normalized entropy across the seven model classes; higher means less decisive.")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("## Model insight")
    crop_mask = cv2.resize(
        segmentation["mask"][box[1]:box[3], box[0]:box[2]],
        (heat.shape[1], heat.shape[0]), interpolation=cv2.INTER_NEAREST,
    )
    hot = heat >= np.quantile(heat, 0.70)
    attention_overlap = float(hot[crop_mask.astype(bool)].sum() / max(hot.sum(), 1))
    descriptors = visual_descriptors(crop_rgb, crop_mask)
    gradcam_tab, segmentation_tab, shape_tab, colour_tab = st.tabs([
        "Grad-CAM", "Segmentation", "Shape", "Colour",
    ])

    with gradcam_tab:
        left, right = st.columns([1.08, 1], gap="large")
        with left:
            st.image(gradcam_view, use_container_width=True)
        with right:
            st.markdown("### Classification evidence")
            st.metric("High-attention pixels inside lesion", f"{attention_overlap:.1%}")
            st.write(
                "Warm colours show the regions that most increased the selected class score. "
                "A high overlap means the classifier concentrated more of its strongest activation "
                "inside the segmented region; it does not prove the category is correct."
            )
            st.caption("Grad-CAM explains model attention, not biological malignancy.")

    with segmentation_tab:
        st.markdown("### Segmentation visualisation")
        controls = st.columns(4)
        with controls[0]:
            overlay_opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.48, 0.05)
        with controls[1]:
            boundary_thickness = st.slider("Boundary thickness", 1, 8, 3)
        with controls[2]:
            zoom_level = st.slider("Zoom", 1.0, 3.0, 1.0, 0.25)
        with controls[3]:
            side_by_side = st.toggle("Side-by-side comparison", value=True)
        interactive_overlay = mask_overlay(
            image_rgb, segmentation["mask"], overlay_opacity, boundary_thickness,
        )
        probability_view = probability_map_with_colorbar(segmentation["probability"])
        zoomed_original = zoom_to_mask(image_rgb, segmentation["mask"], zoom_level)
        zoomed_overlay = zoom_to_mask(interactive_overlay, segmentation["mask"], zoom_level)
        st.caption(
            f"Mask threshold: **{MASK_THRESHOLD:.2f}** · Overlay opacity: **{overlay_opacity:.0%}** · "
            f"Boundary: **{boundary_thickness}px white inner contour with dark outer contour**"
        )
        left, middle, right = st.columns(3, gap="large")
        with left:
            st.markdown("### Probability map")
            st.image(probability_view, use_container_width=True)
            st.caption("The colour bar maps pixel-level lesion probability from 0.0 to 1.0.")
        with middle:
            st.markdown("### Binary mask")
            st.image(segmentation["mask"] * 255, clamp=True, use_container_width=True)
            st.caption(f"Threshold {MASK_THRESHOLD:.2f}; the largest connected region was retained.")
        with right:
            st.markdown("### Boundary overlay")
            st.image(zoomed_overlay, use_container_width=True)
            st.metric("Mean foreground probability", f"{segmentation['foreground_probability']:.1%}")
        if side_by_side:
            st.markdown("#### Original and overlay comparison")
            comparison_left, comparison_right = st.columns(2, gap="small")
            comparison_left.image(zoomed_original, caption="Original image", use_container_width=True)
            comparison_right.image(zoomed_overlay, caption="Model-predicted lesion boundary", use_container_width=True)
        download_mask, download_overlay = st.columns(2)
        download_mask.download_button(
            "Download mask",
            data=array_to_png_bytes(segmentation["mask"] * 255),
            file_name=f"{upload_fingerprint}_predicted_mask.png",
            mime="image/png",
            on_click="ignore",
        )
        download_overlay.download_button(
            "Download overlay",
            data=array_to_png_bytes(interactive_overlay),
            file_name=f"{upload_fingerprint}_boundary_overlay.png",
            mime="image/png",
            on_click="ignore",
        )

    with shape_tab:
        left, right = st.columns([1.05, 1], gap="large")
        extent = mask_extent(segmentation["mask"])
        with left:
            st.image(segmented_view, use_container_width=True)
            st.caption("The dark outer and white inner contours show the model-selected boundary.")
        with right:
            st.markdown("### Geometry summary")
            st.metric("Mask coverage", f"{segmentation['coverage']:.1%}")
            st.caption(
                f"Selected {extent['pixels']:,} of {extent['total']:,} image pixels; "
                f"mask span is {extent['width_fraction']:.1%} of image width and "
                f"{extent['height_fraction']:.1%} of image height."
            )
            for label, value in (
                ("Relative asymmetry", descriptors["asymmetry"]),
                ("Relative border variation", descriptors["border_irregularity"]),
            ):
                st.markdown(f'<div class="descriptor"><b>{label}</b><span style="float:right">{value:.0%}</span><div class="descriptor-line"><div class="descriptor-fill" style="width:{value * 100:.1f}%"></div></div></div>', unsafe_allow_html=True)
            st.caption("These are image descriptors, not clinical ABCDE scores.")

    with colour_tab:
        left, right = st.columns([1.05, 1], gap="large")
        palette, colour_fractions = color_palette(crop_rgb, crop_mask)
        lesion_pixels = crop_rgb[crop_mask.astype(bool)]
        channel_spread = float(np.mean(np.std(lesion_pixels, axis=0))) if len(lesion_pixels) else 0.0
        with left:
            st.image(crop_rgb, use_container_width=True, caption="Masked classifier input")
            st.image(palette, use_container_width=True, caption="Dominant lesion-region colour palette")
        with right:
            st.markdown("### Colour and texture summary")
            st.metric("Relative colour variation", f"{descriptors['color_variation']:.0%}")
            st.metric("RGB channel spread", f"{channel_spread:.1f} / 255")
            st.write(
                "Palette share: " + ", ".join(f"colour {index + 1}: {share:.0%}" for index, share in enumerate(colour_fractions))
            )
            st.caption("Lighting, camera processing, skin tone, hair, and masking can all change these values.")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("## Condition explanation")
    extent = mask_extent(segmentation["mask"])
    guide = DISEASE_GUIDE[predicted_code]
    explanation_tab, treatment_tab, research_tab = st.tabs([
        "What this result means", "Treatment / cure context", "Research sources",
    ])
    with explanation_tab:
        st.markdown(f"### {CLASS_NAMES[predicted_code]}")
        st.write(guide["about"])
        st.markdown("#### Area selected in this image")
        st.write(
            f"The segmentation mask selected **{extent['pixels']:,} of {extent['total']:,} pixels** "
            f"(**{segmentation['coverage']:.1%}** of the uploaded image), {location_text}. "
            f"That is {coverage_description(float(segmentation['coverage']))} of this photograph."
        )
        st.warning(
            "This percentage changes with zoom, crop, and camera distance. It does not measure "
            "true physical diameter, depth, body-surface area, cancer stage, or spread."
        )
    with treatment_tab:
        st.markdown("### Evidence-based management context")
        st.write(guide["care"])
        st.markdown(f"**When to seek review:** {guide['urgency']}")
        st.error(
            "No cure or prescription can be selected from this image or mask percentage. "
            "Treatment requires clinical examination and, when indicated, biopsy/pathology."
        )
    with research_tab:
        st.markdown("### Primary clinical reference")
        st.markdown(f"[{guide['source']}]({guide['url']})")
        st.write(
            "The application summary is deliberately conservative: it describes established "
            "management options but does not choose a medication, procedure, dose, or surgical margin."
        )

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("## Your report")
    report_id = f"CVL-{datetime.now().astimezone():%Y%m%d}-{upload_fingerprint.upper()}"
    with st.spinner("Preparing your illustrated PDF report..."):
        pdf_report = build_pdf_report(
            image_rgb=image_rgb,
            segmented_overlay=segmented_view,
            predicted_code=predicted_code,
            classification_confidence=prediction_confidence,
            mean_foreground_probability=float(segmentation["foreground_probability"]),
            coverage=float(segmentation["coverage"]),
            location_text=location_text,
            warnings=warnings,
            report_id=report_id,
            prediction_uncertainty=entropy,
        )
    st.download_button(
        "Download illustrated PDF report",
        data=pdf_report,
        file_name=f"cutivanta_lens_{upload_fingerprint}.pdf",
        mime="application/pdf",
        on_click="ignore",
    )
    render_ui_footer()


if __name__ == "__main__":
    main()
