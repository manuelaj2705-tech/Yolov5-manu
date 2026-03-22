from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="📷",
    layout="wide"
)

# ─── Estilos personalizados ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo general */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Título principal */
h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    text-align: center !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 0.3rem !important;
}

/* Subtítulos */
h2, h3 {
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    color: #c8caff !important;
    letter-spacing: -0.3px !important;
}

/* Párrafos y markdown */
p, .stMarkdown p {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 300 !important;
    color: #9da3b4 !important;
    text-align: center;
    font-size: 1rem !important;
    line-height: 1.7 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #1e2130 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #c8caff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #9da3b4 !important;
    font-family: 'DM Sans', sans-serif !important;
    text-align: left !important;
}

/* Sliders */
.stSlider > div > div > div {
    background: #6366f1 !important;
}

/* Botón de cámara */
[data-testid="stCameraInput"] button {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.4rem !important;
    transition: background 0.2s ease !important;
}
[data-testid="stCameraInput"] button:hover {
    background: #4f52d9 !important;
}

/* Label del camera input */
[data-testid="stCameraInput"] label {
    font-family: 'DM Sans', sans-serif !important;
    color: #9da3b4 !important;
    font-size: 0.95rem !important;
    text-align: center !important;
    display: block !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Info / error / caption */
.stAlert {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stCaption, caption {
    font-family: 'Space Mono', monospace !important;
    color: #555a72 !important;
    font-size: 0.78rem !important;
    text-align: center !important;
}

/* Divider */
hr {
    border-color: #1e2130 !important;
    margin: 2rem 0 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #6366f1 !important;
}

/* Centrar bloques principales */
.block-container {
    padding-top: 2.5rem !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* Bar chart */
[data-testid="stVegaLiteChart"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Modelo ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

st.title(" 📸 Detección de Objetos en Imágenes 📸")
st.markdown("Esta aplicación utiliza un modelo entrenado en Teachable Machine para detectar personas y objetos en imágenes capturadas con tu cámara, mostrando en tiempo real la probabilidad de cada detección.")

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if model:
    with st.sidebar:
        st.title("Parámetros")
        st.subheader("Configuración de detección")
        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

    picture = st.camera_input("Captura una imagen y descubre los objetos que podemos capturar", key="camera")

    if picture:
        bytes_data = picture.getvalue()
        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]  # RGB → BGR para que YOLO procese bien

        with st.spinner("Detectando objetos..."):
            try:
                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=int(max_det)
                )
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        result    = results[0]
        boxes     = result.boxes
        annotated = result.plot()              # devuelve BGR numpy array
        annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB sin cv2

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen con detecciones")
            st.image(annotated_rgb, use_container_width=True)
        with col2:
            st.subheader("Objetos detectados")
            if boxes is not None and len(boxes) > 0:
                label_names    = model.names
                category_count = {}
                category_conf  = {}
                for box in boxes:
                    cat  = int(box.cls.item())
                    conf = float(box.conf.item())
                    category_count[cat] = category_count.get(cat, 0) + 1
                    category_conf.setdefault(cat, []).append(conf)
                data = [
                    {
                        "Categoría":          label_names[cat],
                        "Cantidad":           count,
                        "Confianza promedio": f"{np.mean(category_conf[cat]):.2f}"
                    }
                    for cat, count in category_count.items()
                ]
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Categoría")["Cantidad"])
            else:
                st.info("No se detectaron objetos con los parámetros actuales.")
                st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
