"""Interfaz web Streamlit para el pipeline de mejora de imagenes.

Mejoras clave en este modulo:
- Interfaz grafica para cargar imagenes y aplicar el pipeline sin CLI.
- Controles interactivos para ajustar parametros criticos (gamma, contraste, enfoque).
- Previsualizacion en vivo del resultado junto con descarga directa del archivo mejorado.
- Uso del pipeline reutilizable definido en image_enhancement.py.
- Nueva pagina que resume visualmente los aportes del equipo.
"""

from typing import List

import cv2
import numpy as np
import streamlit as st

from image_enhancement import EnhancementConfig, enhance_image_array

# Configuracion inicial de la pagina Streamlit.
st.set_page_config(page_title="Image Enhancement", layout="wide")

# Resumen dinamico de aportes que se mostrara en la pagina adicional.
CONTRIBUTIONS: List[dict] = [
    {
        "titulo": "Pipeline modular",
        "categoria": "Procesamiento",
        "impacto": "Calidad de imagen consistente con menos artefactos.",
        "detalle": (
            "Se creo EnhancementConfig y una funcion reutilizable que opera sobre la luminancia, "
            "combina filtros medianos, bilaterales y mascara de realce."
        ),
        "score": 0.90,
        "emoji": ":brain:",
    },
    {
        "titulo": "Espacio de color LAB",
        "categoria": "Procesamiento",
        "impacto": "Colores mas naturales tras el realce.",
        "detalle": (
            "El pipeline trabaja sobre el canal L y reconvierte a BGR, preservando la cromatica original."
        ),
        "score": 0.75,
        "emoji": ":art:",
    },
    {
        "titulo": "CLI configurable",
        "categoria": "DevOps",
        "impacto": "Permite automatizar lotes y ajustar parametros desde la linea de comandos.",
        "detalle": "Se agregaron argumentos para gamma, modo color, nitidez y nivel de logging.",
        "score": 0.80,
        "emoji": ":desktop_computer:",
    },
    {
        "titulo": "Guardado robusto",
        "categoria": "DevOps",
        "impacto": "Evita fallas silenciosas al escribir archivos.",
        "detalle": "Se garantizo la creacion de carpetas y verificacion del guardado con excepciones claras.",
        "score": 0.70,
        "emoji": ":floppy_disk:",
    },
    {
        "titulo": "Interfaz Streamlit",
        "categoria": "Experiencia",
        "impacto": "Permite probar mejoras sin programar y comparar resultados en vivo.",
        "detalle": "Carga de imagen, sliders para configuracion y descarga inmediata del resultado.",
        "score": 0.95,
        "emoji": ":globe_showing_meridians:",
    },
    {
        "titulo": "Documentacion renovada",
        "categoria": "Experiencia",
        "impacto": "Onboarding rapido gracias a instrucciones actualizadas.",
        "detalle": "README describe requisitos, uso de CLI y lanzamiento de la interfaz web.",
        "score": 0.65,
        "emoji": ":books:",
    },
]


def _bytes_from_image(image: np.ndarray) -> bytes:
    """Convierte un arreglo BGR a bytes PNG descargables."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode(".png", rgb)
    if not success:
        raise RuntimeError("No se pudo generar la imagen de salida.")
    return buffer.tobytes()


def _render_enhancement_page() -> None:
    """Renderiza la pagina principal de mejora de imagenes."""
    st.title("Grupo 2 AI/ML Image Enhancement")
    st.write(
        "Carga una imagen, ajusta los parametros y descarga una version mejorada con OpenCV + SciPy. "
        "Mejora realizada por el grupo 2. Integrantes: Alejandro Martinez, Giovana Marachi, "
        "Yamili Acosta y Pedro Florenciano."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Parametros de mejora")
    clip_limit = st.sidebar.slider("Clip Limit (CLAHE)", 1.0, 5.0, 2.5, 0.1)
    tile_grid = st.sidebar.slider("Tamano de cuadricula CLAHE", 4, 16, 8, 1)
    gamma = st.sidebar.slider("Gamma", 0.8, 2.2, 1.4, 0.1)
    median_kernel = st.sidebar.slider("Kernel mediano", 1, 7, 3, 2)
    use_bilateral = st.sidebar.checkbox("Activar filtro bilateral", value=True)
    sharpen_amount = st.sidebar.slider("Intensidad de realce (unsharp)", 0.0, 2.0, 1.1, 0.1)
    sharpen_sigma = st.sidebar.slider("Sigma del desenfoque (unsharp)", 0.1, 3.0, 1.0, 0.1)
    process_in_color = st.sidebar.checkbox("Conservar color (espacio LAB)", value=True)

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )

    if uploaded_file is None:
        st.info("Sube una imagen para comenzar el proceso de mejora.")
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("No se pudo leer la imagen cargada. Intentalo nuevamente.")
        return

    config = EnhancementConfig(
        clip_limit=clip_limit,
        tile_grid_size=tile_grid,
        gamma=gamma,
        median_kernel=median_kernel,
        use_bilateral=use_bilateral,
        sharpen_amount=sharpen_amount,
        sharpen_sigma=sharpen_sigma,
        process_in_color=process_in_color,
    )

    enhanced_image = enhance_image_array(image, config)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.subheader("Mejorada")
        st.image(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    buffer = _bytes_from_image(enhanced_image)
    st.download_button(
        "Descargar imagen mejorada (PNG)",
        data=buffer,
        file_name="enhanced.png",
        mime="image/png",
    )


def _render_contributions_page() -> None:
    """Renderiza la pagina con los aportes del proyecto de forma dinamica."""
    st.title("Aportes del equipo")
    st.write(
        "Resumen interactivo de las mejoras incorporadas al proyecto. "
        "Filtra por categoria y revisa el impacto de cada iniciativa."
    )

    categorias = sorted({item["categoria"] for item in CONTRIBUTIONS})
    st.sidebar.markdown("---")
    st.sidebar.header("Explorar aportes")
    seleccion = st.sidebar.multiselect(
        "Filtrar por categoria",
        options=categorias,
        default=categorias,
    )

    st.sidebar.metric("Total de aportes", len(CONTRIBUTIONS))
    if seleccion:
        conteo_filtrado = sum(1 for item in CONTRIBUTIONS if item["categoria"] in seleccion)
        st.sidebar.metric("Aportes visibles", conteo_filtrado)
    else:
        st.sidebar.warning("Selecciona al menos una categoria para visualizar aportes.")
        return

    datos_filtrados = [item for item in CONTRIBUTIONS if item["categoria"] in seleccion]

    if not datos_filtrados:
        st.info("No hay aportes registrados para las categorias seleccionadas.")
        return

    st.caption("Impacto relativo estimado (0 a 1) representado como barra de progreso.")
    for idx in range(0, len(datos_filtrados), 2):
        cols = st.columns(2)
        for col, item in zip(cols, datos_filtrados[idx : idx + 2]):
            with col:
                st.markdown(f"### {item['emoji']} {item['titulo']}")
                st.write(item["detalle"])
                st.success(f"Impacto: {item['impacto']}")
                st.progress(min(max(item["score"], 0.0), 1.0))


# Navegacion: permite alternar entre el laboratorio y la pagina de aportes.
st.sidebar.title("Navegacion")
pagina = st.sidebar.radio(
    "Selecciona una seccion",
    options=("Laboratorio de mejora", "Aportes del proyecto"),
)

if pagina == "Laboratorio de mejora":
    _render_enhancement_page()
else:
    _render_contributions_page()
