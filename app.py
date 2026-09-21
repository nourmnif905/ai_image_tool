
import io
import numpy as np
import streamlit as st
import torch
from PIL import Image

from utils import (
    analyze_edges,
    analyze_sharpness,
    color_histogram,
    dominant_colors,
)

st.set_page_config(
    page_title="Générateur d'images IA",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #1a1a1a;
        --panel: #242424;
        --text: #f2f2f2;
        --muted: #a3a3a3;
        --accent: #4f8ef7;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--panel);
    }

    h1.app-title {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    p.app-subtitle {
        color: var(--muted);
        margin-bottom: 1.8rem;
    }

    .stButton > button {
        background-color: var(--accent);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.4rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #3d7ce0;
        color: white;
    }

    div[data-testid="stImage"] {
        background-color: var(--panel);
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipeline(model_id: str):
    """Charge et met en cache le pipeline Stable Diffusion."""
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    if device == "cpu":
        # Réduit l'empreinte mémoire sur CPU (plus lent mais fonctionnel)
        pipe.enable_attention_slicing()

    return pipe, device

st.sidebar.header("Paramètres")

model_id = st.sidebar.selectbox(
    "Modèle",
    [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1-base",
    ],
)

num_steps = st.sidebar.slider("Qualité (étapes)", 5, 50, 25)
guidance_scale = st.sidebar.slider("Fidélité au texte", 1.0, 15.0, 7.5)
seed = st.sidebar.number_input("Seed (0 = aléatoire)", min_value=0, value=0, step=1)


st.markdown('<h1 class="app-title">Générateur d\'images IA</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Décrivez une image, générez-la, puis explorez son analyse.</p>',
    unsafe_allow_html=True,
)

prompt = st.text_area(
    "Description de l'image",
    placeholder="ex: a futuristic city skyline at sunset, digital art, highly detailed",
    height=80,
)

generate_clicked = st.button("Générer l'image", type="primary")

if generate_clicked:
    if not prompt.strip():
        st.warning("Merci d'entrer une description avant de générer une image.")
    else:
        with st.spinner("Chargement du modèle (première exécution uniquement)..."):
            pipe, device = load_pipeline(model_id)

        generator = None
        if seed != 0:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        with st.spinner("Génération de l'image en cours..."):
            result = pipe(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            image = result.images[0]

        
        st.session_state["last_image"] = image
        st.session_state["last_prompt"] = prompt

if "last_image" in st.session_state:
    image: Image.Image = st.session_state["last_image"]

    st.markdown("---")
    col_img, col_download = st.columns([3, 1])

    with col_img:
        st.image(image, caption=st.session_state.get("last_prompt", ""), use_container_width=True)

    with col_download:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            "Télécharger l'image",
            data=buf.getvalue(),
            file_name="generated_image.png",
            mime="image/png",
        )

    st.markdown("## Analyse de l'image")
    tab_edges, tab_sharp, tab_hist, tab_colors = st.tabs(
        ["Contours", "Netteté", "Histogramme", "Couleurs dominantes"]
    )

    img_array = np.array(image.convert("RGB"))

    with tab_edges:
        edges = analyze_edges(img_array)
        st.image(edges, caption="Détection de contours (Canny)", use_container_width=True)

    with tab_sharp:
        sharpness = analyze_sharpness(img_array)
        st.metric("Score de netteté", f"{sharpness:.2f}")
        st.caption(
            "Un score plus élevé indique une image plus nette / plus détaillée."
        )

    with tab_hist:
        fig = color_histogram(img_array)
        st.pyplot(fig)

    with tab_colors:
        colors = dominant_colors(img_array, k=5)
        cols = st.columns(len(colors))
        for c, col in zip(colors, cols):
            hex_color = "#%02x%02x%02x" % tuple(c)
            col.markdown(
                f"<div style='background-color:{hex_color}; height:70px; "
                f"border-radius:6px;'></div>"
                f"<p style='text-align:center; color:#a3a3a3; margin-top:0.35rem;'>{hex_color}</p>",
                unsafe_allow_html=True,
            )
else:
    st.info("Entrez une description ci-dessus puis cliquez sur *Générer l'image* pour commencer.")
