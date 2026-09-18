from __future__ import annotations

import base64
import io
import json
import os
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from lightgbm import LGBMClassifier
from pydantic import BaseModel, Field
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"
ARTIFACT_DIR = BASE / "artifacts"
MODEL_REGISTRY_PATH = MODEL_DIR / "model_registry.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = json.loads((BASE / "feature_names.json").read_text(encoding="utf-8"))
RESEARCH = json.loads((ARTIFACT_DIR / "research.json").read_text(encoding="utf-8"))

QHAT_FILE = BASE / "q_hat_absolute.csv"
if not QHAT_FILE.exists():
    raise FileNotFoundError(f"Missing conformal calibration file: {QHAT_FILE}")

_qhat_df = pd.read_csv(QHAT_FILE)
if "q_hat" not in _qhat_df.columns or _qhat_df.empty:
    raise ValueError("q_hat_absolute.csv must contain a non-empty 'q_hat' column.")
BASELINE_QHAT = float(_qhat_df["q_hat"].iloc[0])

LOW = 0.30
HIGH = 0.60

# Thesis Table 3.7 thresholds.
THRESH = {
    "auc_min": 0.70,
    "ece_max": 0.08,
    "coverage_min": 0.90,
    "psi_max": 0.25,
    "shap_max": 0.15,
}

NOMINAL_COVERAGE = 0.90

# ============================================================
# MODEL LOADING
# ============================================================

ORIGINAL_MODEL_PATH = MODEL_DIR / "lightgbm_model.joblib"


def model_version(path: Path) -> str:
    if path.name.startswith("production_model_"):
        return path.stem.replace("production_model_", "")
    return path.stem


def is_original_model(path: Path) -> bool:
    return path.resolve() == ORIGINAL_MODEL_PATH.resolve()


def _read_model_registry() -> Dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        return {
            "active_model": None,
            "previous_model": None,
            "models": [],
        }

    try:
        data = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Registry must contain a JSON object.")
        data.setdefault("active_model", None)
        data.setdefault("previous_model", None)
        data.setdefault("models", [])
        return data
    except Exception as exc:
        print(f"[model-registry] Could not read registry: {exc}")
        return {
            "active_model": None,
            "previous_model": None,
            "models": [],
        }


def _write_model_registry(data: Dict[str, Any]) -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MODEL_REGISTRY_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(MODEL_REGISTRY_PATH)


def _discover_production_models() -> list[Path]:
    return sorted(
        MODEL_DIR.glob("production_model_*.joblib"),
        key=lambda p: p.stat().st_mtime,
    )


def _ensure_registry() -> Dict[str, Any]:
    registry = _read_model_registry()
    production_models = _discover_production_models()

    # Remove registry entries for files that no longer exist.
    valid_names = {p.name for p in production_models}
    cleaned_models = []
    for item in registry.get("models", []):
        if isinstance(item, dict):
            filename = item.get("filename")
            if filename in valid_names or filename == ORIGINAL_MODEL_PATH.name:
                cleaned_models.append(item)

    registry["models"] = cleaned_models

    known = {item.get("filename") for item in cleaned_models}

    # Always keep the baseline model represented.
    if ORIGINAL_MODEL_PATH.exists() and ORIGINAL_MODEL_PATH.name not in known:
        registry["models"].insert(
            0,
            {
                "filename": ORIGINAL_MODEL_PATH.name,
                "model_version": model_version(ORIGINAL_MODEL_PATH),
                "type": "baseline",
                "status": "archived",
                "roc_auc": None,
                "created_at": None,
            },
        )

    # Backward-compatible migration: if no active model is registered,
    # use the newest existing production model, otherwise baseline.
    active_name = registry.get("active_model")
    active_path = MODEL_DIR / active_name if active_name else None

    if active_path is None or not active_path.exists():
        if production_models:
            active_path = production_models[-1]
            registry["active_model"] = active_path.name
        elif ORIGINAL_MODEL_PATH.exists():
            registry["active_model"] = ORIGINAL_MODEL_PATH.name
        else:
            registry["active_model"] = None

    # Mark statuses consistently.
    active_name = registry.get("active_model")
    for item in registry["models"]:
        filename = item.get("filename")
        if filename == active_name:
            item["status"] = "active"
        elif filename != ORIGINAL_MODEL_PATH.name:
            item["status"] = "archived"

    _write_model_registry(registry)
    return registry


def get_active_model_path() -> Path:
    registry = _ensure_registry()
    active_name = registry.get("active_model")

    if active_name:
        path = MODEL_DIR / active_name
        if path.exists():
            return path

    if ORIGINAL_MODEL_PATH.exists():
        return ORIGINAL_MODEL_PATH

    raise FileNotFoundError(
        "No active production model or lightgbm_model.joblib was found."
    )


def get_latest_production_model() -> Tuple[Any, Path]:
    path = get_active_model_path()
    return joblib.load(path), path


if not ORIGINAL_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing original model: {ORIGINAL_MODEL_PATH}")

BASELINE_MODEL = joblib.load(ORIGINAL_MODEL_PATH)
ACTIVE_MODEL, ACTIVE_MODEL_PATH = get_latest_production_model()



def model_version(path: Path) -> str:
    if path.name.startswith("production_model_"):
        return path.stem.replace("production_model_", "")
    return path.stem


def is_original_model(path: Path) -> bool:
    return path.resolve() == ORIGINAL_MODEL_PATH.resolve()


# ============================================================
# REFERENCE / RETRAINING DATA
# ============================================================

# IMPORTANT:
# - The monitoring CSV can be a FULL dataset such as test_2019.csv.
# - PSI and SHAP Drift still need a REFERENCE dataset.
# - The preferred reference is:
#       1) REFERENCE_DATA_PATH (full training/reference CSV), if set
#       2) backend/reference_data.csv
#       3) backend/X_train_small.csv
#       4) cached/downloaded X_train_small.csv from the original research Space
#
# This does NOT limit the monitoring/test dataset to the small sample.
# It only provides the baseline distribution needed for drift calculations.

REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "").strip()
REFERENCE_DATA_URL = os.getenv(
    "REFERENCE_DATA_URL",
    "https://huggingface.co/spaces/siboshan/research/resolve/main/X_train_small.csv?download=true",
).strip()

REFERENCE_CANDIDATES = [
    Path(REFERENCE_DATA_PATH) if REFERENCE_DATA_PATH else None,
    BASE / "reference_data.csv",
    BASE / "X_train_small.csv",
    ARTIFACT_DIR / "reference_data.csv",
]

REFERENCE_CACHE = ARTIFACT_DIR / "_reference_cache.csv"

TRAIN_X_FILE = BASE / "X_train_small.csv"
TRAIN_Y_FILE = BASE / "y_train_small.csv"
CAL_X_FILE = BASE / "X_cal_small.csv"
CAL_Y_FILE = BASE / "y_cal_small.csv"

RETRAINING_FILES = [
    TRAIN_X_FILE,
    TRAIN_Y_FILE,
    CAL_X_FILE,
    CAL_Y_FILE,
]

MODEL_LOCK = threading.RLock()


def _existing_reference_path() -> Optional[Path]:
    for path in REFERENCE_CANDIDATES:
        if path is not None and path.exists():
            return path
    if REFERENCE_CACHE.exists():
        return REFERENCE_CACHE
    return None


def _download_reference_if_needed() -> Optional[Path]:
    """
    Last-resort compatibility fallback for the public demo.

    The production recommendation is to package reference_data.csv with the
    backend or set REFERENCE_DATA_PATH to the full training/reference dataset.
    The fallback keeps the demo functional when the local small reference file
    was not copied from the original research project.
    """
    if not REFERENCE_DATA_URL:
        return None

    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        if REFERENCE_CACHE.exists() and REFERENCE_CACHE.stat().st_size > 1000:
            return REFERENCE_CACHE

        with urllib.request.urlopen(REFERENCE_DATA_URL, timeout=30) as response:
            data = response.read()

        if len(data) < 1000:
            raise ValueError("Downloaded reference file is unexpectedly small.")

        REFERENCE_CACHE.write_bytes(data)
        return REFERENCE_CACHE
    except Exception as exc:
        print(f"[reference] Remote reference download failed: {exc}")
        return None


def get_reference_frame() -> Tuple[pd.DataFrame, str]:
    """
    Return the feature-only reference frame and its source.

    A full training dataset can be used here; any extra columns such as
    default, issue_d or year are ignored by validate_feature_frame().
    """
    path = _existing_reference_path()
    source = str(path) if path is not None else ""

    if path is None:
        path = _download_reference_if_needed()
        source = str(path) if path is not None else ""

    if path is None:
        raise ValueError(
            "No PSI/SHAP reference dataset is available. "
            "Add backend/reference_data.csv (preferred), set REFERENCE_DATA_PATH, "
            "or install X_train_small.csv from the research artifacts."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read reference dataset: {path.name}") from exc

    X = validate_feature_frame(df)
    if len(X) < 20:
        raise ValueError("The PSI/SHAP reference dataset must contain at least 20 rows.")

    return X, source


def load_retraining_data() -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load the original research/retraining samples when they are available.

    These files are NOT used as the monitoring/test dataset. They are only
    used when the adaptive retraining workflow needs to build a candidate
    model using the original Gradio training strategy.
    """
    result: Dict[str, Optional[pd.DataFrame]] = {
        "X_train": None,
        "y_train": None,
        "X_cal": None,
        "y_cal": None,
    }

    if all(p.exists() for p in RETRAINING_FILES):
        result["X_train"] = pd.read_csv(TRAIN_X_FILE)
        result["y_train"] = pd.read_csv(TRAIN_Y_FILE).squeeze("columns")
        result["X_cal"] = pd.read_csv(CAL_X_FILE)
        result["y_cal"] = pd.read_csv(CAL_Y_FILE).squeeze("columns")

    return result


RETRAIN_DATA = load_retraining_data()

# ============================================================
# VALIDATION / DATA HELPERS
# ============================================================

def validate_feature_frame(df: pd.DataFrame, allow_extra: bool = True) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing},
        )

    X = df[FEATURES].copy()

    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    invalid = X.columns[X.isna().any()].tolist()
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={"invalid_numeric_columns": invalid},
        )

    return X


def frame_from_dict(data: Dict[str, Any]) -> pd.DataFrame:
    return validate_feature_frame(pd.DataFrame([data]))


def probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    X = validate_feature_frame(X)
    probs = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.clip(probs, 0.0, 1.0)


def risk_category(p: float) -> str:
    if p < LOW:
        return "Low Risk"
    if p < HIGH:
        return "Medium Risk"
    return "High Risk"


def safe_metric(value: float) -> float:
    return round(float(value), 6)


# ============================================================
# TRUSTWORTHINESS METRICS
# ============================================================

def calculate_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    value = 0.0

    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            accuracy = float(np.mean(y_true[mask]))
            confidence = float(np.mean(y_prob[mask]))
            value += float(np.mean(mask)) * abs(accuracy - confidence)

    return float(value)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Prediction-distribution PSI used by the original research demo:
    expected = reference model prediction distribution on the training
    reference sample; actual = prediction distribution on the uploaded
    monitoring observations.

    This keeps the operational calculation consistent with the original
    Gradio implementation while using the thesis threshold of 0.25.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) < 20 or len(actual) < 20:
        raise ValueError("PSI requires at least 20 reference and 20 monitoring values.")

    percentiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.unique(np.percentile(expected, percentiles))

    if len(breakpoints) < 3:
        return 0.0

    # np.histogram requires strictly increasing bin edges.
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts.astype(float) / len(expected)
    actual_pct = actual_counts.astype(float) / len(actual)

    eps = 1e-6
    return float(
        np.sum(
            (actual_pct - expected_pct)
            * np.log((actual_pct + eps) / (expected_pct + eps))
        )
    )


def calculate_conformal_coverage(
    y_true: np.ndarray,
    probs: np.ndarray,
    q_hat: float,
) -> float:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    lower = np.clip(probs - q_hat, 0.0, 1.0)
    upper = np.clip(probs + q_hat, 0.0, 1.0)

    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def _shap_matrix(model: Any, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    # SHAP versions can return either a matrix or a list for classifiers.
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    values = np.asarray(values)

    # Some SHAP/model combinations can produce an extra output dimension.
    if values.ndim == 3:
        values = values[:, :, 1]

    return values


def calculate_shap_reference(
    model: Any,
    X_reference: pd.DataFrame,
    max_rows: int = 500,
) -> np.ndarray:
    sample = X_reference.sample(
        n=min(max_rows, len(X_reference)),
        random_state=4233,
    )
    values = _shap_matrix(model, sample)
    return np.abs(values).mean(axis=0)


def calculate_shap_drift(
    model: Any,
    X_new: pd.DataFrame,
    baseline_shap: np.ndarray,
    max_rows: int = 500,
) -> float:
    if len(X_new) == 0:
        raise ValueError("SHAP Drift requires at least one monitoring row.")

    sample = X_new.sample(
        n=min(max_rows, len(X_new)),
        random_state=4233,
    )

    current_values = _shap_matrix(model, sample)
    current_shap = np.abs(current_values).mean(axis=0)

    if len(current_shap) != len(baseline_shap):
        raise ValueError(
            "SHAP feature count does not match the stored/reference feature schema."
        )

    drift = np.mean(np.abs(current_shap - baseline_shap))
    return float(drift)


# ============================================================
# REFERENCE SHAP CACHE
# ============================================================

BASELINE_SHAP_REFERENCE: Optional[np.ndarray] = None
BASELINE_SHAP_SOURCE: Optional[str] = None


def get_baseline_shap_reference() -> np.ndarray:
    global BASELINE_SHAP_REFERENCE, BASELINE_SHAP_SOURCE

    if BASELINE_SHAP_REFERENCE is None:
        X_reference, source = get_reference_frame()

        BASELINE_SHAP_REFERENCE = calculate_shap_reference(
            BASELINE_MODEL,
            X_reference,
        )
        BASELINE_SHAP_SOURCE = source

    return BASELINE_SHAP_REFERENCE


# ============================================================
# MODEL EVALUATION
# ============================================================

def validate_labels(y: pd.Series) -> np.ndarray:
    values = pd.to_numeric(y, errors="coerce")

    if values.isna().any():
        raise HTTPException(
            status_code=422,
            detail="'default' must contain only numeric 0/1 labels.",
        )

    values = values.astype(int)

    if not set(values.unique()).issubset({0, 1}):
        raise HTTPException(
            status_code=422,
            detail="'default' must contain only 0 and 1.",
        )

    if len(values) < 2 or len(values.unique()) < 2:
        raise HTTPException(
            status_code=422,
            detail="'default' must contain both 0 and 1 labels.",
        )

    return values.to_numpy()


def evaluate_predictive_metrics(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    include_probability_vector: bool = False,
) -> Dict[str, Any]:
    probs = probability(model, X)

    metrics = {
        "roc_auc": safe_metric(roc_auc_score(y, probs)),
        "brier_score": safe_metric(brier_score_loss(y, probs)),
        "log_loss": safe_metric(log_loss(y, probs, labels=[0, 1])),
        "ece": safe_metric(calculate_ece(y, probs)),
        "cp_coverage": safe_metric(
            calculate_conformal_coverage(y, probs, BASELINE_QHAT)
        ),
    }

    if include_probability_vector:
        metrics["_probs"] = probs

    return metrics


def evaluate_full_trustworthiness(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    calculate_drift: bool = True,
) -> Dict[str, Any]:
    metrics = evaluate_predictive_metrics(
        model,
        X,
        y,
        include_probability_vector=True,
    )

    probs = metrics.pop("_probs")

    result: Dict[str, Any] = {
        "roc_auc": metrics["roc_auc"],
        "brier_score": metrics["brier_score"],
        "log_loss": metrics["log_loss"],
        "ece": metrics["ece"],
        "cp_coverage": metrics["cp_coverage"],
        "psi": None,
        "shap_drift": None,
        "psi_reference": "training prediction distribution",
        "shap_reference": "original LightGBM model + training reference sample",
    }

    if calculate_drift:
        X_reference, reference_source = get_reference_frame()

        # PSI is calculated against the reference model prediction
        # distribution, while the uploaded monitoring/test dataset can be
        # any size (including the full 2019 dataset).
        baseline_probs = probability(model, X_reference)
        result["psi"] = safe_metric(calculate_psi(baseline_probs, probs))
        result["psi_reference_source"] = reference_source

        baseline_shap = get_baseline_shap_reference()
        result["shap_drift"] = safe_metric(
            calculate_shap_drift(model, X, baseline_shap)
        )
        result["shap_reference_source"] = BASELINE_SHAP_SOURCE

    return result


# ============================================================
# THRESHOLD DECISION
# ============================================================

def threshold_flags(metrics: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "roc_auc": float(metrics["roc_auc"]) < THRESH["auc_min"],
        "ece": float(metrics["ece"]) > THRESH["ece_max"],
        "cp_coverage": float(metrics["cp_coverage"]) < THRESH["coverage_min"],
        "psi": (
            metrics["psi"] is not None
            and float(metrics["psi"]) > THRESH["psi_max"]
        ),
        "shap_drift": (
            metrics["shap_drift"] is not None
            and float(metrics["shap_drift"]) > THRESH["shap_max"]
        ),
    }


def should_retrain(flags: Dict[str, bool]) -> bool:
    # Five threshold-based indicators in the thesis.
    return any(
        flags[key]
        for key in (
            "roc_auc",
            "ece",
            "cp_coverage",
            "psi",
            "shap_drift",
        )
    )


def monitoring_status(flags: Dict[str, bool]) -> str:
    return "RETRAIN" if should_retrain(flags) else "HEALTHY"


# ============================================================
# RETRAINING
# ============================================================

def require_retraining_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = RETRAIN_DATA["X_train"]
    y_train = RETRAIN_DATA["y_train"]
    X_cal = RETRAIN_DATA["X_cal"]
    y_cal = RETRAIN_DATA["y_cal"]

    if any(x is None for x in (X_train, y_train, X_cal, y_cal)):
        missing = [
            str(path.name)
            for path in RETRAINING_FILES
            if not path.exists()
        ]
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Retraining data files are not installed.",
                "missing_files": missing,
            },
        )

    X_train = validate_feature_frame(X_train)
    X_cal = validate_feature_frame(X_cal)

    y_train = pd.to_numeric(y_train, errors="coerce").astype(int)
    y_cal = pd.to_numeric(y_cal, errors="coerce").astype(int)

    if not set(y_train.unique()).issubset({0, 1}) or not set(y_cal.unique()).issubset({0, 1}):
        raise HTTPException(
            status_code=422,
            detail="Training/calibration labels must contain only 0 and 1.",
        )

    return X_train, y_train, X_cal, y_cal


def train_candidate_model(
    X_new: pd.DataFrame,
    y_new: np.ndarray,
) -> Any:
    X_train, y_train, X_cal, y_cal = require_retraining_data()

    X_new = validate_feature_frame(X_new)
    y_new_series = pd.Series(y_new, name="default")

    # Preserve the original Gradio retraining strategy:
    # original training sample + calibration sample + new labelled data.
    X_retrain = pd.concat(
        [X_train, X_cal, X_new],
        ignore_index=True,
    )
    y_retrain = pd.concat(
        [pd.Series(y_train), pd.Series(y_cal), y_new_series],
        ignore_index=True,
    )

    candidate = LGBMClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        random_state=42,
        verbosity=-1,
    )

    candidate.fit(X_retrain, y_retrain)
    return candidate


def save_production_model(model: Any) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = MODEL_DIR / f"production_model_{timestamp}.joblib"

    # Avoid overwriting an existing timestamped model.
    counter = 1
    while path.exists():
        path = MODEL_DIR / f"production_model_{timestamp}_{counter}.joblib"
        counter += 1

    joblib.dump(model, path)
    return path


def register_model(
    path: Path,
    *,
    model_type: str = "production",
    roc_auc: Optional[float] = None,
) -> None:
    registry = _ensure_registry()
    models = [
        item for item in registry.get("models", [])
        if item.get("filename") != path.name
    ]

    models.append(
        {
            "filename": path.name,
            "model_version": model_version(path),
            "type": model_type,
            "status": "active" if registry.get("active_model") == path.name else "archived",
            "roc_auc": roc_auc,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    registry["models"] = models
    _write_model_registry(registry)


def deploy_candidate(
    candidate: Any,
    candidate_path: Path,
    *,
    roc_auc: Optional[float] = None,
) -> None:
    global ACTIVE_MODEL, ACTIVE_MODEL_PATH

    registry = _ensure_registry()
    previous_name = registry.get("active_model")

    ACTIVE_MODEL = candidate
    ACTIVE_MODEL_PATH = candidate_path

    registry["previous_model"] = previous_name
    registry["active_model"] = candidate_path.name

    # Preserve every previous production model; never delete/overwrite it.
    register_model(
        candidate_path,
        model_type="production",
        roc_auc=roc_auc,
    )

    registry = _ensure_registry()
    for item in registry.get("models", []):
        filename = item.get("filename")
        if filename == candidate_path.name:
            item["status"] = "active"
        elif filename != ORIGINAL_MODEL_PATH.name:
            item["status"] = "archived"

    _write_model_registry(registry)


def retrain_and_compare(
    X_new: pd.DataFrame,
    y_new: np.ndarray,
) -> Dict[str, Any]:
    """
    Retrain and compare candidate/current model on a holdout split of the
    newly supplied labelled data.

    This avoids evaluating the candidate only on the same observations used
    to fit it. Deployment occurs only when candidate ROC-AUC is strictly
    higher than the current model ROC-AUC on the same holdout data.
    """
    if len(X_new) < 40:
        raise HTTPException(
            status_code=422,
            detail="At least 40 labelled monitoring rows are required for safe candidate evaluation.",
        )

    if len(np.unique(y_new)) < 2:
        raise HTTPException(
            status_code=422,
            detail="The monitoring labels must contain both 0 and 1.",
        )

    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X_new,
        y_new,
        test_size=0.25,
        random_state=42,
        stratify=y_new,
    )

    with MODEL_LOCK:
        current_model = ACTIVE_MODEL
        current_path = ACTIVE_MODEL_PATH

        candidate = train_candidate_model(X_fit, y_fit)

        current_probs = probability(current_model, X_eval)
        candidate_probs = probability(candidate, X_eval)

        current_auc = safe_metric(roc_auc_score(y_eval, current_probs))
        candidate_auc = safe_metric(roc_auc_score(y_eval, candidate_probs))

        accepted = candidate_auc > current_auc

        candidate_path: Optional[Path] = None

        if accepted:
            candidate_path = save_production_model(candidate)
            deploy_candidate(candidate, candidate_path, roc_auc=candidate_auc)

        return {
            "accepted": accepted,
            "current_model": current_path.name,
            "current_model_version": model_version(current_path),
            "current_holdout_roc_auc": current_auc,
            "candidate_holdout_roc_auc": candidate_auc,
            "roc_auc_change": safe_metric(candidate_auc - current_auc),
            "candidate_model": (
                candidate_path.name if candidate_path is not None else None
            ),
            "deployment_status": (
                "DEPLOYED"
                if accepted
                else "REJECTED_CURRENT_MODEL_RETAINED"
            ),
            "conformal_recalibration": {
                "status": "NOT_PERFORMED",
                "q_hat_used_for_monitoring": BASELINE_QHAT,
                "note": (
                    "The thesis reports that complete post-retraining conformal "
                    "recalibration was not performed because a suitable recent "
                    "calibration dataset was unavailable."
                ),
            },
        }


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Trustworthy AI Credit Risk API",
    version="3.0.0",
    description=(
        "Public API for the thesis demonstration: credit risk prediction, "
        "seven-indicator trustworthiness monitoring, adaptive retraining, "
        "candidate-vs-current model comparison, and verified research outputs."
    ),
)

origins = [
    x.strip()
    for x in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class PredictReq(BaseModel):
    features: Dict[str, Any] = Field(...)
    mode: str = "production"


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Trustworthy AI Credit Risk API",
        "status": "healthy",
        "docs": "/docs",
        "health": "/api/health",
        "reference": "/api/reference",
        "monitor": "/api/monitor",
    }


@app.get("/api/health")
def health():
    registry = _ensure_registry()
    return {
        "status": "healthy",
        "active_model": ACTIVE_MODEL_PATH.name,
        "active_model_version": model_version(ACTIVE_MODEL_PATH),
        "previous_model": registry.get("previous_model"),
        "baseline_model": ORIGINAL_MODEL_PATH.name,
        "feature_count": len(FEATURES),
        "baseline_qhat": BASELINE_QHAT,
        "nominal_coverage": NOMINAL_COVERAGE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/config")
def config():
    return {
        "features": FEATURES,
        "risk_thresholds": {
            "low_max_exclusive": LOW,
            "medium_max_exclusive": HIGH,
        },
        "monitoring_thresholds": THRESH,
        "nominal_coverage": NOMINAL_COVERAGE,
        "active_model": ACTIVE_MODEL_PATH.name,
        "active_model_version": model_version(ACTIVE_MODEL_PATH),
        "previous_model": _ensure_registry().get("previous_model"),
        "baseline_model": ORIGINAL_MODEL_PATH.name,
        "baseline_model_version": model_version(ORIGINAL_MODEL_PATH),
        "baseline_qhat": BASELINE_QHAT,
        "monitoring_indicators": [
            "ROC-AUC",
            "Brier Score",
            "LogLoss",
            "ECE",
            "Conformal Prediction Coverage",
            "PSI",
            "SHAP Drift",
        ],
    }


@app.get("/api/model")
def model_status():
    registry = _ensure_registry()
    return {
        "active_model": ACTIVE_MODEL_PATH.name,
        "active_model_version": model_version(ACTIVE_MODEL_PATH),
        "previous_model": registry.get("previous_model"),
        "baseline_model": ORIGINAL_MODEL_PATH.name,
        "baseline_model_version": model_version(ORIGINAL_MODEL_PATH),
        "is_baseline_active": is_original_model(ACTIVE_MODEL_PATH),
        "conformal_status": (
            "baseline_qhat_available"
            if is_original_model(ACTIVE_MODEL_PATH)
            else "post_retraining_recalibration_not_performed"
        ),
    }


# ============================================================
# MODEL HISTORY / ROLLBACK
# ============================================================

@app.get("/api/models")
def model_history():
    registry = _ensure_registry()

    items = []
    for item in registry.get("models", []):
        filename = item.get("filename")
        path = MODEL_DIR / filename if filename else None

        if path is None or not path.exists():
            continue

        items.append(
            {
                **item,
                "active": filename == registry.get("active_model"),
                "exists": True,
            }
        )

    return {
        "active_model": registry.get("active_model"),
        "previous_model": registry.get("previous_model"),
        "baseline_model": ORIGINAL_MODEL_PATH.name,
        "models": items,
        "rollback_available": len(
            [x for x in items
             if x.get("filename") not in {
                 registry.get("active_model"),
                 ORIGINAL_MODEL_PATH.name,
             }]
        ) > 0,
    }


class RollbackReq(BaseModel):
    model: str = Field(..., description="Exact model filename to activate.")


@app.post("/api/models/rollback")
def rollback_model(req: RollbackReq):
    global ACTIVE_MODEL, ACTIVE_MODEL_PATH

    target_name = Path(req.model).name

    if target_name == ORIGINAL_MODEL_PATH.name:
        target_path = ORIGINAL_MODEL_PATH
    else:
        target_path = MODEL_DIR / target_name

    if not target_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {target_name}",
        )

    if target_path.suffix != ".joblib":
        raise HTTPException(
            status_code=422,
            detail="Rollback target must be a .joblib model file.",
        )

    if target_path.resolve().parent != MODEL_DIR.resolve():
        raise HTTPException(status_code=422, detail="Invalid model path.")

    with MODEL_LOCK:
        previous_name = ACTIVE_MODEL_PATH.name
        try:
            restored_model = joblib.load(target_path)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load rollback model: {exc}",
            ) from exc

        ACTIVE_MODEL = restored_model
        ACTIVE_MODEL_PATH = target_path

        registry = _ensure_registry()
        registry["previous_model"] = previous_name
        registry["active_model"] = target_path.name

        for item in registry.get("models", []):
            filename = item.get("filename")
            if filename == target_path.name:
                item["status"] = "active"
            elif filename != ORIGINAL_MODEL_PATH.name:
                item["status"] = "archived"

        _write_model_registry(registry)

    return {
        "status": "ROLLED_BACK",
        "active_model": ACTIVE_MODEL_PATH.name,
        "active_model_version": model_version(ACTIVE_MODEL_PATH),
        "previous_model": previous_name,
        "baseline_model": ORIGINAL_MODEL_PATH.name,
        "conformal_status": (
            "baseline_qhat_available"
            if is_original_model(ACTIVE_MODEL_PATH)
            else "post_retraining_recalibration_not_performed"
        ),
        "message": (
            f"Active model changed from {previous_name} "
            f"to {ACTIVE_MODEL_PATH.name}. "
            "The model file was preserved."
        ),
    }


# ============================================================
# REFERENCE DATA STATUS
# ============================================================

@app.get("/api/reference")
def reference_status():
    try:
        X_reference, source = get_reference_frame()
        return {
            "available": True,
            "source": source,
            "rows": len(X_reference),
            "feature_count": len(X_reference.columns),
            "features": list(X_reference.columns),
            "note": (
                "This dataset is the PSI/SHAP reference only. The uploaded "
                "monitoring CSV can be a different, full dataset."
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "source": None,
            "rows": 0,
            "feature_count": 0,
            "error": str(exc),
        }


# ============================================================
# CREDIT RISK PREDICTION
# ============================================================

@app.post("/api/predict")
def predict(req: PredictReq):
    mode = req.mode.lower().strip()

    if mode not in {"production", "baseline"}:
        raise HTTPException(
            status_code=422,
            detail="mode must be 'production' or 'baseline'.",
        )

    X = frame_from_dict(req.features)

    with MODEL_LOCK:
        if mode == "baseline":
            active_model = BASELINE_MODEL
            active_path = ORIGINAL_MODEL_PATH
        else:
            active_model = ACTIVE_MODEL
            active_path = ACTIVE_MODEL_PATH

        p = float(probability(active_model, X)[0])

    interval = None
    note = None

    if mode == "baseline":
        interval = {
            "lower": round(max(0.0, p - BASELINE_QHAT), 6),
            "upper": round(min(1.0, p + BASELINE_QHAT), 6),
        }
        note = (
            f"Static CP interval using the thesis-calibrated q-hat "
            f"= {BASELINE_QHAT:.6f}. The interval describes the empirical "
            f"coverage behaviour of the conformal procedure; it is not a "
            f"probability statement about this individual borrower."
        )
    else:
        note = (
            "The active production model is used for the probability. "
            "No conformal interval is attached to a retrained production "
            "model because the thesis reports that complete post-retraining "
            "conformal recalibration was not performed."
        )

    return {
        "default_probability": round(p, 6),
        "risk_category": risk_category(p),
        "conformal_interval": interval,
        "conformal_note": note,
        "model": active_path.name,
        "model_version": model_version(active_path),
        "mode": mode,
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/api/predict/batch")
async def batch_predict(file: UploadFile = File(...)):
    raw = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, "Invalid CSV file.") from exc

    X = validate_feature_frame(df)

    with MODEL_LOCK:
        probs = probability(ACTIVE_MODEL, X)
        active_path = ACTIVE_MODEL_PATH

    output = df.copy()
    output["default_probability"] = np.round(probs, 6)
    output["risk_category"] = [risk_category(float(p)) for p in probs]

    high = int(np.sum(probs >= HIGH))
    medium = int(np.sum((probs >= LOW) & (probs < HIGH)))
    low = int(np.sum(probs < LOW))

    csv_bytes = output.to_csv(index=False).encode("utf-8")

    return {
        "rows": len(output),
        "summary": {
            "low": low,
            "medium": medium,
            "high": high,
        },
        "model": active_path.name,
        "model_version": model_version(active_path),
        "preview": (
            output.head(30)
            .replace({np.nan: None})
            .to_dict("records")
        ),
        "csv_base64": base64.b64encode(csv_bytes).decode("ascii"),
    }


# ============================================================
# VERIFIED RESEARCH DASHBOARD
# ============================================================

@app.get("/api/research")
def research():
    # Research dashboard values are intentionally read from the stored
    # verified artifact instead of being recomputed on every interaction.
    return RESEARCH


@app.get("/api/trustworthiness")
def trustworthiness():
    return {
        "verified": RESEARCH.get("verified_trustworthiness"),
        "thresholds": THRESH,
        "indicators": [
            {
                "name": "ROC-AUC",
                "threshold": THRESH["auc_min"],
                "direction": "concern_below",
            },
            {
                "name": "ECE",
                "threshold": THRESH["ece_max"],
                "direction": "concern_above",
            },
            {
                "name": "Conformal Prediction Coverage",
                "threshold": THRESH["coverage_min"],
                "direction": "concern_below",
            },
            {
                "name": "PSI",
                "threshold": THRESH["psi_max"],
                "direction": "concern_above",
            },
            {
                "name": "SHAP Drift",
                "threshold": THRESH["shap_max"],
                "direction": "concern_above",
            },
            {
                "name": "Brier Score",
                "threshold": None,
                "direction": "supporting_metric",
            },
            {
                "name": "LogLoss",
                "threshold": None,
                "direction": "supporting_metric",
            },
        ],
    }


# ============================================================
# LIVE TRUSTWORTHINESS MONITOR + ADAPTIVE RETRAINING
# ============================================================

@app.post("/api/monitor")
async def monitor(
    file: UploadFile = File(...),
    auto_retrain: bool = Form(True),
):
    raw = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, "Invalid CSV file.") from exc

    if "default" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="Monitoring CSV must contain a 'default' column.",
        )

    X = validate_feature_frame(df)
    y = validate_labels(df["default"])

    # Snapshot the active model for the whole monitoring transaction.
    with MODEL_LOCK:
        current_model = ACTIVE_MODEL
        current_path = ACTIVE_MODEL_PATH

    try:
        current_metrics = evaluate_full_trustworthiness(
            current_model,
            X,
            y,
            calculate_drift=True,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            500,
            f"Live trustworthiness calculation failed: {exc}",
        ) from exc

    flags = threshold_flags(current_metrics)
    decision = monitoring_status(flags)

    response: Dict[str, Any] = {
        "rows": len(df),
        "model": current_path.name,
        "model_version": model_version(current_path),
        "total_indicators": 7,
        "threshold_based_indicators": 5,
        "supporting_indicators": 2,
        "metrics": current_metrics,
        "threshold_flags": flags,
        "decision": decision,
        "auto_retrain_requested": auto_retrain,
        "retraining": None,
        "conformal_recalibration": {
            "status": (
                "AVAILABLE_FOR_BASELINE_MONITORING"
                if is_original_model(current_path)
                else "NOT_PERFORMED_AFTER_RETRAINING"
            ),
            "q_hat_used": BASELINE_QHAT,
            "note": (
                "The live monitoring coverage calculation uses the available "
                "thesis q-hat. It does not claim a newly recalibrated q-hat "
                "for a retrained production model."
            ),
        },
        "indicator_notes": {
            "psi": (
                "Live PSI compares the active-model prediction distribution "
                "on the configured reference dataset with the uploaded monitoring "
                "dataset. The monitoring dataset may be the full test_2019.csv."
            ),
            "shap_drift": (
                "Live SHAP Drift compares mean absolute SHAP importance on the "
                "uploaded monitoring dataset with the original-model reference dataset."
            ),
            "brier_score": "Supporting predictive-quality indicator.",
            "log_loss": "Supporting predictive-quality indicator.",
        },
    }

    if decision == "RETRAIN" and auto_retrain:
        try:
            retraining_result = retrain_and_compare(X, y)
        except HTTPException as exc:
            response["retraining"] = {
                "status": "FAILED",
                "error": exc.detail,
                "deployment_status": "CURRENT_MODEL_RETAINED",
            }
            response["retraining_error"] = str(exc.detail)
            return response
        except Exception as exc:
            response["retraining"] = {
                "status": "FAILED",
                "error": str(exc),
                "deployment_status": "CURRENT_MODEL_RETAINED",
            }
            response["retraining_error"] = str(exc)
            return response

        response["retraining"] = retraining_result

        # Evaluate the newly active model on the same uploaded monitoring
        # data so the UI can show before/after results. This is descriptive;
        # candidate acceptance itself was decided using the holdout split.
        if retraining_result["accepted"]:
            with MODEL_LOCK:
                new_model = ACTIVE_MODEL
                new_path = ACTIVE_MODEL_PATH

            try:
                # Keep the defence response fast: candidate acceptance has
                # already been decided on the held-out monitoring split.
                # Re-running SHAP/PSI here would repeat the most expensive
                # part of the transaction and can make a browser request
                # appear to fail even though retraining succeeded.
                after_metrics = evaluate_predictive_metrics(new_model, X, y)
            except Exception as exc:
                response["after_retraining"] = {
                    "model": new_path.name,
                    "model_version": model_version(new_path),
                    "metrics": None,
                    "threshold_flags": None,
                    "note": f"Model was deployed, but post-deployment predictive metrics could not be displayed: {exc}",
                }
            else:
                response["after_retraining"] = {
                    "model": new_path.name,
                    "model_version": model_version(new_path),
                    "metrics": after_metrics,
                    "threshold_flags": threshold_flags({
                        **after_metrics,
                        "psi": None,
                        "shap_drift": None,
                    }),
                    "drift_recalculation": "not repeated after deployment in this live transaction",
                }
        else:
            response["after_retraining"] = None

    elif decision == "RETRAIN" and not auto_retrain:
        response["retraining"] = {
            "status": "NOT_RUN",
            "reason": "A threshold-based degradation signal was detected, but auto_retrain=false.",
        }

    return response


# ============================================================
# MANUAL RETRAINING ENDPOINT
# ============================================================

@app.post("/api/retrain")
async def retrain_only(
    file: UploadFile = File(...),
):
    raw = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, "Invalid CSV file.") from exc

    if "default" not in df.columns:
        raise HTTPException(
            422,
            "Retraining CSV must contain a 'default' column.",
        )

    X = validate_feature_frame(df)
    y = validate_labels(df["default"])

    result = retrain_and_compare(X, y)

    if result["accepted"]:
        with MODEL_LOCK:
            new_model = ACTIVE_MODEL
            new_path = ACTIVE_MODEL_PATH

        try:
            after = evaluate_full_trustworthiness(
                new_model,
                X,
                y,
                calculate_drift=True,
            )
        except Exception as exc:
            raise HTTPException(
                500,
                f"Post-retraining evaluation failed: {exc}",
            ) from exc

        result["after_retraining_metrics"] = after
        result["after_retraining_model"] = new_path.name

    return result


# ============================================================
# APPLICATION STARTUP INFORMATION
# ============================================================

print("=" * 64)
print("TRUSTWORTHY AI CREDIT RISK API")
print("=" * 64)
print(f"Active model : {ACTIVE_MODEL_PATH.name}")
try:
    _startup_registry = _ensure_registry()
    print(f"Previous     : {_startup_registry.get('previous_model')}")
    print(f"Registry     : {MODEL_REGISTRY_PATH}")
except Exception as exc:
    print(f"Registry     : unavailable ({exc})")
print(f"Baseline     : {ORIGINAL_MODEL_PATH.name}")
print(f"Features     : {len(FEATURES)}")
print(f"Baseline qhat: {BASELINE_QHAT:.6f}")
print(f"Thresholds   : {THRESH}")
try:
    _ref_path = _existing_reference_path()
    print(f"Reference    : {str(_ref_path) if _ref_path else 'remote fallback / not cached'}")
except Exception:
    print("Reference    : unavailable at startup")
print("=" * 64)
