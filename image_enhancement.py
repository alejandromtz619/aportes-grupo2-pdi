"""Image Enhancement OpenCV

Mejoras implementadas:
- Se centraliza toda la configuración del pipeline en una data class para facilitar ajustes.
- Se añade soporte para procesar imágenes individuales o carpetas completas desde la CLI.
- Se mejora la calidad trabajando en el espacio de color LAB y manteniendo el color original.
- Se incorporan filtros bilaterales y máscara de realce (unsharp mask) configurables.
- Se expone una API reutilizable (enhance_image_array) para reusar el pipeline desde otros módulos.
- Se agrega registro estructurado y manejo explícito de errores para depuración.
"""

#--------------------------------
# Date : 18-06-2020
# Project : Image Enhancement OpenCV
# Category : Image Processing
# Company : weblineindia
# Department : AI/ML
#--------------------------------

import argparse  # Mejora: interfaz CLI moderna.
import logging  # Mejora: registro estructurado en vez de prints sueltos.
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from scipy import ndimage

# Mejora: constantes con Path para portabilidad multiplataforma.
DEFAULT_INPUT_DIR = Path("Dataset")
DEFAULT_OUTPUT_DIR = Path("Results")


@dataclass
class EnhancementConfig:
    """Mejora: configuración centralizada y documentada del pipeline."""

    clip_limit: float = 2.5  # Mejora: CLAHE más configurable.
    tile_grid_size: int = 8
    gamma: float = 1.4  # Mejora: control fino de gamma.
    median_kernel: int = 3
    use_bilateral: bool = True
    bilateral_d: int = 7
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    sharpen_amount: float = 1.1
    sharpen_sigma: float = 1.0
    process_in_color: bool = True  # Mejora: conservar color realzando luminancia.


def ensure_directory(path: Path) -> None:
    """Mejora: crea carpetas de salida de forma segura."""
    path.mkdir(parents=True, exist_ok=True)


def _apply_pipeline_to_luminance(
    luminance: np.ndarray, config: EnhancementConfig
) -> np.ndarray:
    """Mejora: pipeline modular reutilizable para el canal de luminancia."""
    # Mejora: filtrado mediano controlado para reducir ruido impulsivo.
    filtered = ndimage.median_filter(luminance, size=config.median_kernel)

    if config.use_bilateral:
        # Mejora: bilateral preserva bordes mientras suaviza texturas.
        filtered = cv2.bilateralFilter(
            filtered,
            d=config.bilateral_d,
            sigmaColor=config.bilateral_sigma_color,
            sigmaSpace=config.bilateral_sigma_space,
        )

    # Mejora: CLAHE adaptativo para contrastes locales balanceados.
    clahe = cv2.createCLAHE(
        clipLimit=config.clip_limit, tileGridSize=(config.tile_grid_size, config.tile_grid_size)
    )
    contrast = clahe.apply(filtered)

    # Mejora: gamma configurable para evitar quemar zonas oscuras/clares.
    normalized = np.clip(contrast / 255.0, 0.0, 1.0)
    gamma_corrected = np.power(normalized, 1.0 / config.gamma)
    gamma_corrected = np.uint8(np.clip(gamma_corrected * 255.0, 0, 255))

    if config.sharpen_amount > 0:
        # Mejora: máscara de realce para recuperar nitidez sin ruido excesivo.
        blurred = cv2.GaussianBlur(gamma_corrected, ksize=(0, 0), sigmaX=config.sharpen_sigma)
        sharpened = cv2.addWeighted(
            gamma_corrected, 1 + config.sharpen_amount, blurred, -config.sharpen_amount, 0
        )
        return np.uint8(np.clip(sharpened, 0, 255))

    return gamma_corrected


def enhance_image_array(image: np.ndarray, config: Optional[EnhancementConfig] = None) -> np.ndarray:
    """Mejora: API reutilizable para realzar imágenes en memoria."""
    if config is None:
        config = EnhancementConfig()

    if image is None or image.size == 0:
        raise ValueError("La imagen recibida está vacía.")

    if config.process_in_color:
        # Mejora: trabajar en LAB para preservar color mientras realzamos luminancia.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = _apply_pipeline_to_luminance(l_channel, config)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = _apply_pipeline_to_luminance(gray, config)

    return enhanced


def save_image(output_path: Path, image: np.ndarray) -> None:
    """Mejora: guardado robusto con creación previa de carpeta."""
    ensure_directory(output_path.parent)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise IOError(f"No se pudo guardar la imagen en {output_path}")


def iter_input_images(path: Path) -> Iterable[Path]:
    """Mejora: iterador seguro que acepta archivos individuales o carpetas."""
    if path.is_file():
        yield path
    elif path.is_dir():
        for file_path in sorted(path.iterdir()):
            if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                yield file_path
    else:
        raise FileNotFoundError(f"No se encontró la ruta {path}")


def process_images(
    input_path: Path,
    output_dir: Path,
    config: Optional[EnhancementConfig] = None,
) -> None:
    """Mejora: función de alto nivel para procesar múltiples imágenes."""
    if config is None:
        config = EnhancementConfig()

    ensure_directory(output_dir)

    for image_path in iter_input_images(input_path):
        try:
            logging.info("Procesando %s", image_path.name)
            image = cv2.imread(str(image_path))
            if image is None:
                logging.warning("No se pudo leer la imagen %s", image_path)
                continue

            enhanced = enhance_image_array(image, config)
            output_path = output_dir / image_path.name
            save_image(output_path, enhanced)
            logging.info("Guardado en %s", output_path)
        except Exception as exc:  # pylint: disable=broad-except
            # Mejora: manejo de excepciones que continúa con el lote.
            logging.error("Error procesando %s: %s", image_path, exc)


def parse_args() -> argparse.Namespace:
    """Mejora: CLI con argumentos descriptivos y ayuda."""
    parser = argparse.ArgumentParser(
        description="Pipeline de mejora de imágenes basado en OpenCV y SciPy."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Archivo o carpeta con imágenes a procesar (por defecto: Dataset/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta de salida donde se guardarán las imágenes mejoradas (por defecto: Results/).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Valor gamma para controlar el brillo global (por defecto configurado en EnhancementConfig).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Procesa en escala de grises (usa solo luminancia).",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Desactiva la máscara de realce si causa ruido.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de verbosidad del registro (por defecto: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    """Mejora: punto de entrada controlado con logging y CLI."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(message)s",
    )

    config = EnhancementConfig()
    if args.gamma:
        config.gamma = args.gamma  # Mejora: permitir ajustar gamma en la línea de comandos.
    if args.no_color:
        config.process_in_color = False
    if args.no_sharpen:
        config.sharpen_amount = 0.0

    logging.info("-------- IMAGE ENHANCEMENT TECHNIQUE --------")
    logging.info("---------------------------------------------")
    logging.info("-------- INICIALIZANDO PROCESAMIENTO --------")

    process_images(args.input, args.output, config)

    logging.info("-------- PROCESAMIENTO COMPLETADO -----------")


if __name__ == "__main__":
    main()

