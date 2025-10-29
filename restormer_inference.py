"""Inferencia simplificada del modelo Restormer para la demo en Streamlit.

Descarga el modelo TorchScript desde Hugging Face la primera vez y lo reutiliza
posteriormente. Las dependencias fuertes (PyTorch) se cargan de forma perezosa
para no romper el flujo principal cuando no estan instaladas.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # PyTorch no está disponible.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

import cv2


class RestormerUnavailableError(RuntimeError):
    """Se lanza cuando Restormer no puede utilizarse en la instalacion actual."""


_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "real_denoising": {
        "url": "https://huggingface.co/spaces/swzamir/Restormer/resolve/main/real_denoising.pt",
        "filename": "real_denoising.pt",
        "label": "Restormer Real Denoising",
    },
}

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_CACHE: Dict[Tuple[str, str], "torch.jit.ScriptModule"] = {}


def restormer_is_available() -> bool:
    """Indica si PyTorch esta instalado y listo para usarse."""

    return torch is not None


def _safe_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")

    request = urllib.request.Request(url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(request) as response, open(temp_path, "wb") as out_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)

    shutil.move(temp_path, destination)


def _ensure_weights(task: str) -> Path:
    if task not in _MODEL_SPECS:
        raise ValueError(f"Tarea Restormer desconocida: {task}")

    spec = _MODEL_SPECS[task]
    target_path = _MODEL_DIR / spec["filename"]

    if not target_path.exists():
        _safe_download(spec["url"], target_path)

    return target_path


def load_restormer_model(task: str = "real_denoising", device: Optional[str] = None) -> "torch.jit.ScriptModule":
    """Carga el modelo TorchScript y lo deja cacheado en memoria."""

    if torch is None:
        raise RestormerUnavailableError(
            "PyTorch no esta instalado. Instala torch>=1.12 para usar el modelo Restormer."
        )

    model_path = _ensure_weights(task)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (task, resolved_device)

    if cache_key not in _MODEL_CACHE:
        model = torch.jit.load(str(model_path), map_location=resolved_device)
        model = model.to(resolved_device)
        model.eval()
        _MODEL_CACHE[cache_key] = model

    return _MODEL_CACHE[cache_key]


def apply_restormer(
    image_bgr: np.ndarray,
    task: str = "real_denoising",
    boost_amount: float = 0.0,
    boost_sigma: float = 1.0,
) -> np.ndarray:
    """Ejecuta Restormer sobre una imagen BGR (uint8) y permite realce posterior.

    Args:
        image_bgr: Imagen de entrada en formato BGR uint8.
        task: Nombre de la tarea (por ahora solo 'real_denoising').
        boost_amount: Intensidad del realce extra posterior (unsharp). 0 desactiva.
        boost_sigma: Sigma para el desenfoque gaussiano del realce.
    """

    if torch is None or F is None:
        raise RestormerUnavailableError(
            "PyTorch no esta disponible. Instala torch>=1.12 para habilitar Restormer."
        )

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Se esperaba una imagen BGR con tres canales.")

    rgb = image_bgr[:, :, ::-1].copy()
    tensor = torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_restormer_model(task, device=device)

    tensor = tensor.to(device)
    height, width = tensor.shape[2], tensor.shape[3]
    multiple = 8
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.inference_mode():
        restored = torch.clamp(model(padded), 0.0, 1.0)

    restored = restored[:, :, :height, :width]
    restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored = np.clip(restored * 255.0, 0.0, 255.0).astype(np.uint8)
    result_bgr = restored[:, :, ::-1]

    if boost_amount > 0:
        blur = cv2.GaussianBlur(result_bgr, ksize=(0, 0), sigmaX=boost_sigma)
        boosted = cv2.addWeighted(result_bgr, 1.0 + boost_amount, blur, -boost_amount, 0.0)
        result_bgr = np.clip(boosted, 0, 255).astype(np.uint8)

    return result_bgr


__all__ = [
    "apply_restormer",
    "load_restormer_model",
    "restormer_is_available",
    "RestormerUnavailableError",
]
