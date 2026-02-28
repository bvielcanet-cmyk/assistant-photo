import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from google import genai
from skimage import exposure 
import gc

# --- CONFIGURATION IA ---
@st.cache_resource
def get_ai_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except: return None

client = get_ai_client()

# --- MOTEUR DE RENDU ELITE 2026 ---
def retouche_elite_2026(img_np, intensite, rotation_angle, client_ai=None):
    # 1. Pivotage
    if rotation_angle != 0:
        if rotation_angle == 90: img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180: img_np = cv2.rotate(img_np, cv2.ROTATE_180)
        elif rotation_angle == 270: img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 2. Correction Expo IA
    correction_expo = 1.0
    if client_ai:
        try:
            thumb_ai = Image.fromarray(cv2.resize(img_np, (256, 256)))
            res = client_ai.models.generate_content(model="gemini-1.5-flash", contents=["Expo?", thumb_ai])
            if "DARK" in res.text.upper(): correction_expo = 1.15
            elif "BRIGHT" in res.text.upper(): correction_expo = 0.85
        except: pass

    # 3. COLOR GRADING & ECLAT (LAB)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.multiply(l, correction_expo).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Color Grading : réchauffer les tons chair (Canal A=Rouge, B=Jaune)
    a = cv2.add(a, int(3 * intensite))
    b = cv2.add(b, int(2 * intensite))
    img_studio = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    # 4. SEPARATION DE FREQUENCES (Peau parfaite)
    low = cv2.GaussianBlur(img_studio, (25, 25), 0)
    high = cv2.subtract(img_studio, low)
    sigma = int(10 + (intensite * 50))
    low_smooth = cv2.bilateralFilter(low, d=9, sigmaColor=sigma, sigmaSpace=sigma)
    
    # Recomposition avec Texture Boost (1.1)
    img_combined = cv2.addWeighted(low_smooth, 1.0, high, 1.1, 0)

    # 5. VIGNETTAGE PRO (Focus sujet)
    rows, cols = img_combined.shape[:2]
    k_x = cv2.getGaussianKernel(cols, cols/2)
    k_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = k_y * k_x.T
    mask = kernel / kernel.max()
    img_vignette = img_combined.astype(float)
    for i in range(3):
        img_vignette[:,:,i] = img_vignette[:,:,i] * (0.8 + 0.2 * mask)
    
    img_elite = np.clip(img_vignette, 0, 255).astype(np.uint8)
    return cv2.addWeighted(img_np, 1.0 - intensite, img_elite, intensite, 0)

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Elite 2026", layout="wide")

if 'finalistes' not in st.session_state: st.session_state.finalistes = []
if 'rotations' not in st.session_state: st.session_state.rotations = {}
if 'rendus_prets' not in st.session_state: st.session_state.rendus_prets = False
if 'zip_buffer' not in st.session_state: st.session_state.zip_buffer = None

with st.sidebar:
    st.header("💎 Options Élite")
    nb_final = st.slider("Photos suggérées", 1, 50, 15)
    intensite_val = st.slider("Intensité Retouche Élite", 0, 100, 50) / 100.0
    if st.button("🗑️ Réinitialiser tout"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

st.title("📸 Studio IA : Rendu Master Élite")

# --- ETAPE 1 : ANALYSE ---
if not st.session_state.finalistes:
    files = st.file_uploader("Déposez vos photos", type=['jpg', 'jpeg'], accept_multiple_files=True)
    if files:
        if st.button(f"🚀 Analyser {len(files)} photos"):
            with st.status("Analyse intelligente..."):
                data = []
                for f in files:
                    img_pil = Image.open(f).convert('RGB')
                    thumb = img_pil.copy(); thumb.thumbnail((500,500))
                    score = cv2.Laplacian(cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
                    data.append({"nom": f.name, "score_t": score, "thumb": thumb, "file": f})
                st.session_state.finalistes = sorted(data, key=lambda x: x['score_t'], reverse=True)[:nb_final]
                for it in st.session_state.finalistes: st.session_state.rotations[it['nom']] = 0
            st.rerun()

# --- ETAPE 2 : CONFIGURATION ---
if st.session_state.finalistes and not st.session_state.rendus_prets:
    st.subheader(f"🖼️ Sélection et Orientation ({len(st.session_state.finalistes)} photos)")
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.finalistes):
        with cols[idx % 3]:
            angle = st.session_state.rotations.get(item['nom'], 0)
            st.image(item['thumb'].rotate(-angle, expand=True), use_container_width=True)
            c1, c2 = st.columns(2)
            if c1.button(f"🔄 Pivoter", key=f"r_{item['nom']}"):
                st.session_state.rotations[item['nom']] = (angle + 90) % 360
                st.rerun()
            if c2.button(f"❌ Supprimer", key=f"d_{item['nom']}"):
                st.session_state.finalistes = [f for f in st.session_state.finalistes if f['nom'] != item['nom']]
                st.rerun()

    st.divider()
    if st.session_state.finalistes:
        if st.button("✨ GÉNÉRER LE RENDU ÉLITE HD"):
            st.session_state.rendus_prets = True
            st.rerun()

# --- ETAPE 3 : RÉSULTAT ---
if st.session_state.rendus_prets:
    st.subheader("✨ Rendu Élite Finalisé")
    if st.session_state.zip_buffer is None:
        zip_io = io.BytesIO()
        with st.status("Traitement Master Élite HD...") as status:
            with zipfile.ZipFile(zip_io, "w") as xzip:
                cols = st.columns(3)
                for idx, it in enumerate(st.session_state.finalistes):
                    img_hd = np.array(Image.open(it['file']).convert('RGB'))
                    res = retouche_elite_2026(img_hd, intensite_val, st.session_state.rotations[it['nom']], client)
                    
                    # Aperçu final
                    img_pil_final = Image.fromarray(res)
                    with cols[idx % 3]:
                        t_final = img_pil_final.copy(); t_final.thumbnail((400,400))
                        st.image(t_final, caption="Rendu Élite")
                    
                    # Enregistrement ZIP
                    img_io = io.BytesIO()
                    img_pil_final.save(img_io, format='JPEG', quality=92)
                    xzip.writestr(f"ELITE_{it['nom']}", img_io.getvalue())
                    gc.collect()
            st.session_state.zip_buffer = zip_io.getvalue()
            status.update(label="✅ Traitement terminé !", state="complete")

    st.divider()
    st.download_button("📥 TÉLÉCHARGER LE PACK ÉLITE (ZIP)", st.session_state.zip_buffer, "shooting_elite_2026.zip", "application/zip", use_container_width=True)
    if st.button("⬅️ Retourner à la sélection"):
        st.session_state.rendus_prets = False
        st.session_state.zip_buffer = None
        st.rerun()
