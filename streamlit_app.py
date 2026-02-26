import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import google.generativeai as genai
from skimage import exposure 
import gc

# --- CONFIGURATION IA ---
@st.cache_resource
def load_gemini():
    try:
        genai.configure(api_key=st.secrets["GENINI_API_KEY"])
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

model = load_gemini()

# --- FONCTIONS TECHNIQUES ---

def match_style(source_np, reference_np):
    try:
        matched = exposure.match_histograms(source_np, reference_np, channel_axis=-1)
        return matched.astype(np.uint8)
    except: return source_np

def retouche_expert(img_np, intensite):
    """Retouche avec dosage variable (intensite de 0.0 à 1.0)."""
    # 1. Base : Éclat et Contraste (LAB)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2 + (intensite * 0.8)) # Le contraste augmente avec l'intensité
    l = clahe.apply(l)
    img_base = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    
    # 2. Lissage Peau (Bilateral Filter)
    # On pousse les paramètres de lissage en fonction de l'intensité
    sigma = int(15 + (intensite * 60)) 
    img_lisse = cv2.bilateralFilter(img_base, d=9, sigmaColor=sigma, sigmaSpace=sigma)
    
    # 3. MÉLANGE FINAL (Le "Dosage")
    # intensite = 0 -> Image originale | intensite = 1 -> Image max retouchée
    resultat = cv2.addWeighted(img_base, 1.0 - intensite, img_lisse, intensite, 0)
    return resultat

def analyse_groupe(images_pil, max_photos):
    if not model or not images_pil: return list(range(min(len(images_pil), max_photos)))
    prompt = f"Sélectionne les {max_photos} meilleurs indices de photos (variété de poses). Réponds juste les chiffres séparés par des virgules."
    try:
        response = model.generate_content([prompt] + images_pil[:15])
        return [int(s.strip()) for s in response.text.split(',') if s.strip().isdigit()]
    except: return list(range(min(len(images_pil), max_photos)))

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Evolution", layout="wide")

with st.sidebar:
    st.header("⚙️ Réglages Pro")
    nb_selection = st.slider("Nombre de photos à garder", 1, 50, 10)
    
    # NOUVELLE JAUGE DE RETOUCHE
    intensite_retouche = st.slider("Intensité de la retouche (Peau & Éclat)", 0, 100, 50, help="0% = Naturel | 100% = Magazine")
    intensite_float = intensite_retouche / 100.0
    
    st.divider()
    st.subheader("🎨 Style Transfer")
    ref_file = st.file_uploader("Photo de référence", type=['jpg', 'jpeg'])

st.title("🎨 Studio IA : Sélection & Retouche Dosable")

uploaded_files = st.file_uploader("📥 Déposez votre shooting", type=['jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Lancer le Traitement"):
        raw_data = []
        barre = st.progress(0)
        
        # Style Transfer
        ref_np = None
        if ref_file:
            ref_np = np.array(Image.open(ref_file).convert('RGB'))

        for i, f in enumerate(uploaded_files):
            img_np = np.array(Image.open(f).convert('RGB'))
            if ref_np is not None:
                img_np = match_style(img_np, ref_np)
            raw_data.append({"nom": f.name, "np": img_np, "pil": Image.fromarray(img_np)})
            barre.progress((i + 1) / len(uploaded_files))

        # Analyse Groupe
        indices = analyse_groupe([d["pil"] for d in raw_data], nb_selection)
        selection = [raw_data[i] for i in indices if i < len(raw_data)][:nb_selection]

        # ZIP & Affichage
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as xzip:
            st.subheader(f"✨ Sélection (Intensité : {intensite_retouche}%)")
            for idx, item in enumerate(selection):
                # Application de la retouche dosée
                retouche = retouche_expert(item["np"], intensite_float)
                
                # ZIP
                out_io = io.BytesIO()
                Image.fromarray(retouche).save(out_io, format='JPEG', quality=95)
                xzip.writestr(f"EDIT_{item['nom']}", out_io.getvalue())
                
                # UX Comparaison
                with st.expander(f"Photo : {item['nom']}"):
                    c1, c2 = st.columns(2)
                    c1.image(item["np"], caption="Avant (Style)")
                    c2.image(retouche, caption=f"Après (IA {intensite_retouche}%)")
                gc.collect()

        st.divider()
        st.download_button(f"📥 TÉLÉCHARGER LE PACK ({len(selection)} PHOTOS)", data=zip_buf.getvalue(), file_name="shooting_ia.zip")
