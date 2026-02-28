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
    # 1. Rotation
    if rotation_angle != 0:
        if rotation_angle == 90: img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180: img_np = cv2.rotate(img_np, cv2.ROTATE_180)
        elif rotation_angle == 270: img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 2. Traitement Lumière & Peau
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.8).apply(l)
    img_studio = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

    low = cv2.GaussianBlur(img_studio, (15, 15), 0)
    high = cv2.subtract(img_studio, low)
    low_smooth = cv2.bilateralFilter(low, d=7, sigmaColor=int(10+intensite*40), sigmaSpace=int(10+intensite*40))
    final = cv2.add(low_smooth, high)

    return cv2.addWeighted(img_np, 1.0 - intensite, final, intensite, 0)

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Master", layout="wide")

# Initialisation des états de session
if 'finalistes' not in st.session_state: st.session_state.finalistes = []
if 'rotations' not in st.session_state: st.session_state.rotations = {}
if 'rendus_prets' not in st.session_state: st.session_state.rendus_prets = False
if 'zip_buffer' not in st.session_state: st.session_state.zip_buffer = None

with st.sidebar:
    st.header("💎 Réglages")
    nb_final = st.slider("Photos à garder", 1, 50, 15)
    intensite_val = st.slider("Intensité Retouche", 0, 100, 50) / 100.0
    if st.button("🗑️ Réinitialiser tout"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

st.title("📸 Studio IA : Tri, Retouche & Export")

# --- ÉTAPE 1 : UPLOAD ET TRI ---
if not st.session_state.finalistes:
    files = st.file_uploader("Glissez vos photos ici", type=['jpg', 'jpeg'], accept_multiple_files=True)
    if files:
        if st.button(f"⚡ Analyser {len(files)} photos"):
            with st.status("Analyse technique et sélection des meilleures pépites..."):
                data_pool = []
                for f in files:
                    img_pil = Image.open(f).convert('RGB')
                    thumb = img_pil.copy(); thumb.thumbnail((500,500))
                    score = cv2.Laplacian(cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
                    data_pool.append({"nom": f.name, "score_t": score, "thumb": thumb, "file": f})
                st.session_state.finalistes = sorted(data_pool, key=lambda x: x['score_t'], reverse=True)[:nb_final]
                for it in st.session_state.finalistes: st.session_state.rotations[it['nom']] = 0
            st.rerun()

# --- ÉTAPE 2 : CONFIGURATION (ROTATION & APERÇU) ---
if st.session_state.finalistes and not st.session_state.rendus_prets:
    st.subheader("🖼️ Étape 2 : Ajustez l'orientation")
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.finalistes):
        with cols[idx % 3]:
            angle = st.session_state.rotations.get(item['nom'], 0)
            img_disp = item['thumb'].rotate(-angle, expand=True)
            st.image(img_disp, use_container_width=True)
            if st.button(f"🔄 Pivoter", key=f"rot_{item['nom']}"):
                st.session_state.rotations[item['nom']] = (angle + 90) % 360
                st.rerun()

    st.divider()
    if st.button("🚀 GÉNÉRER LES RETOUCHES HAUTE DÉFINITION"):
        st.session_state.rendus_prets = True
        st.rerun()

# --- ÉTAPE 3 : RÉSULTAT FINAL ET ZIP ---
if st.session_state.rendus_prets:
    st.subheader("✨ Étape 3 : Rendu Final & Téléchargement")
    
    # On prépare le ZIP en mémoire une seule fois
    if st.session_state.zip_buffer is None:
        zip_io = io.BytesIO()
        with st.status("Application du traitement Studio Master HD...") as status:
            with zipfile.ZipFile(zip_io, "w") as xzip:
                cols = st.columns(3)
                for idx, it in enumerate(st.session_state.finalistes):
                    # Traitement HD
                    img_hd = np.array(Image.open(it['file']).convert('RGB'))
                    angle = st.session_state.rotations.get(it['nom'], 0)
                    res = retouche_studio_master(img_hd, intensite_val, angle, client)
                    
                    # Sauvegarde pour le ZIP
                    img_io = io.BytesIO()
                    img_pil_final = Image.fromarray(res)
                    img_pil_final.save(img_io, format='JPEG', quality=92)
                    xzip.writestr(f"STUDIO_{it['nom']}", img_io.getvalue())
                    
                    # Affichage du résultat final (en vignette pour la page)
                    with cols[idx % 3]:
                        thumb_final = img_pil_final.copy()
                        thumb_final.thumbnail((400, 400))
                        st.image(thumb_final, caption="Rendu Final HD")
                    
                    gc.collect()
            st.session_state.zip_buffer = zip_io.getvalue()
            status.update(label="✅ Toutes les photos ont été retouchées !", state="complete")

    st.divider()
    # Le bouton de téléchargement est maintenant indépendant
    st.download_button(
        label="📥 TÉLÉCHARGER LE PACK COMPLET (ZIP)",
        data=st.session_state.zip_buffer,
        file_name="mon_shooting_premium.zip",
        mime="application/zip",
        use_container_width=True
    )
    
    if st.button("⬅️ Revenir aux réglages"):
        st.session_state.rendus_prets = False
        st.session_state.zip_buffer = None
        st.rerun()
