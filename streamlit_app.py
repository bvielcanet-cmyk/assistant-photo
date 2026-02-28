import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from google import genai
import gc

# --- CONFIGURATION IA ---
@st.cache_resource
def get_ai_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except: return None

client = get_ai_client()

# --- MOTEUR DE RENDU STUDIO MASTER ---
def retouche_studio_master(img_np, intensite, rotation_angle, client_ai=None):
    # 1. Pivotage initial si nécessaire
    if rotation_angle != 0:
        if rotation_angle == 90:
            img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            img_np = cv2.rotate(img_np, cv2.ROTATE_180)
        elif rotation_angle == 270:
            img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 2. Correction Expo IA
    correction_expo = 1.0
    if client_ai:
        try:
            thumb_ai = Image.fromarray(cv2.resize(img_np, (256, 256)))
            res = client_ai.models.generate_content(model="gemini-1.5-flash", contents=["Expo?", thumb_ai])
            if "DARK" in res.text.upper(): correction_expo = 1.15
        except: pass

    # 3. Traitement Lumière & Peau (Version optimisée)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.multiply(l, correction_expo).astype(np.uint8)
    l = cv2.createCLAHE(clipLimit=1.8).apply(l)
    img_studio = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

    # Séparation de fréquences simplifiée
    low = cv2.GaussianBlur(img_studio, (15, 15), 0)
    high = cv2.subtract(img_studio, low)
    low_smooth = cv2.bilateralFilter(low, d=7, sigmaColor=int(10+intensite*40), sigmaSpace=int(10+intensite*40))
    final = cv2.add(low_smooth, high)

    return cv2.addWeighted(img_np, 1.0 - intensite, final, intensite, 0)

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Master", layout="wide")

# Initialisation des états
if 'finalistes' not in st.session_state:
    st.session_state.finalistes = []
if 'rotations' not in st.session_state:
    st.session_state.rotations = {} # Dictionnaire {nom_photo: angle}

with st.sidebar:
    st.header("💎 Réglages")
    nb_final = st.slider("Photos à garder", 1, 50, 15)
    intensite_val = st.slider("Intensité Retouche", 0, 100, 50) / 100.0
    if st.button("🗑️ Réinitialiser tout"):
        st.session_state.finalistes = []
        st.session_state.rotations = {}
        st.rerun()

st.title("📸 Studio IA : Tri, Retouche & Pivot")

files = st.file_uploader("Glissez vos photos ici", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files and not st.session_state.finalistes:
    if st.button(f"⚡ Analyser {len(files)} photos"):
        with st.status("Analyse technique...", expanded=True):
            data_pool = []
            for f in files:
                img_pil = Image.open(f).convert('RGB')
                thumb = img_pil.copy(); thumb.thumbnail((500,500))
                score = cv2.Laplacian(cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
                data_pool.append({"nom": f.name, "score_t": score, "thumb": thumb, "file": f})
            
            st.session_state.finalistes = sorted(data_pool, key=lambda x: x['score_t'], reverse=True)[:nb_final]
            # Initialiser les rotations à 0
            for it in st.session_state.finalistes:
                st.session_state.rotations[it['nom']] = 0
            st.rerun()

# --- ZONE DE RÉSULTATS & PIVOTAGE ---
if st.session_state.finalistes:
    st.subheader("🖼️ Ajustez l'orientation et validez le pack")
    
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.finalistes):
        with cols[idx % 3]:
            # Appliquer la rotation sur la miniature d'aperçu
            angle = st.session_state.rotations.get(item['nom'], 0)
            img_display = item['thumb']
            if angle != 0:
                img_display = img_display.rotate(-angle, expand=True) # PIL utilise le sens anti-horaire
            
            st.image(img_display, use_container_width=True)
            
            # Bouton pour pivoter
            if st.button(f"🔄 Pivoter (+90°)", key=f"rot_{item['nom']}"):
                st.session_state.rotations[item['nom']] = (angle + 90) % 360
                st.rerun()

    st.divider()
    
    # BOUTON FINAL POUR GÉNÉRER LE ZIP AVEC RETOUCHE
    if st.button("📥 GÉNÉRER ET TÉLÉCHARGER LE PACK HD"):
        zip_buf = io.BytesIO()
        with st.spinner("Application de la retouche Studio Master sur la HD..."):
            with zipfile.ZipFile(zip_buf, "w") as xzip:
                for it in st.session_state.finalistes:
                    img_hd = np.array(Image.open(it['file']).convert('RGB'))
                    angle = st.session_state.rotations.get(it['nom'], 0)
                    
                    # Traitement Master (incluant la rotation)
                    res = retouche_studio_master(img_hd, intensite_val, angle, client)
                    
                    img_io = io.BytesIO()
                    Image.fromarray(res).save(img_io, format='JPEG', quality=92)
                    xzip.writestr(f"STUDIO_{it['nom']}", img_io.getvalue())
                    gc.collect()
            
            st.download_button(
                label="🔥 CLIQUER ICI POUR RÉCUPÉRER LE ZIP",
                data=zip_buf.getvalue(),
                file_name="mon_shooting_final.zip",
                mime="application/zip"
      )
