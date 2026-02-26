import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import google.generativeai as genai

# --- CONFIGURATION GEMINI ---
IA_ARTISTIQUE_ACTIVE = False
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    IA_ARTISTIQUE_ACTIVE = True
except:
    pass

# --- FONCTIONS TECHNIQUES ---
def analyser_technique(image_np):
    gris = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    return min((lap_var / 150) * 50, 50)

def analyser_artistique(image_pil):
    if not IA_ARTISTIQUE_ACTIVE: return 25
    prompt = "Note cette photo de mode sur 50 points (expression, pose, émotion). Réponds UNIQUEMENT par un nombre."
    try:
        response = model.generate_content([prompt, image_pil])
        return min(int(''.join(filter(str.isdigit, response.text))), 50)
    except: return 25

def retoucher_pro(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    return cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Artistique", layout="wide")
st.title("📸 Assistant Photographe : Tri IA & Retouche Texture")

if not IA_ARTISTIQUE_ACTIVE:
    st.sidebar.info("💡 Ajoute GEMINI_API_KEY dans les Secrets pour activer l'analyse artistique.")

with st.form("main-upload", clear_on_submit=False):
    files = st.file_uploader("Shooting JPEG (Limite 1 Go)", type=['jpg', 'jpeg'], accept_multiple_files=True)
    nb_top = st.number_input("Nombre de photos à garder", 1, 100, 20)
    submitted = st.form_submit_button("📁 Valider le dépôt")

if files and submitted:
    resultats = []
    barre = st.progress(0)
    status = st.empty()
    
    for i, f in enumerate(files):
        status.text(f"Traitement de {f.name}...")
        img_pil = Image.open(f).convert('RGB')
        
        # Double analyse
        s_tech = analyser_technique(np.array(img_pil))
        s_art = analyser_artistique(img_pil) if s_tech > 15 else 0
        
        resultats.append({"nom": f.name, "score": s_tech + s_art, "pil": img_pil})
        barre.progress((i + 1) / len(files))

    top = sorted(resultats, key=lambda x: x['score'], reverse=True)[:nb_top]
    
    # Création ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as xzip:
        for item in top:
            retouche = retoucher_pro(np.array(item['pil']))
            img_io = io.BytesIO()
            Image.fromarray(retouche).save(img_io, format='JPEG', quality=95)
            xzip.writestr(f"PRO_{item['nom']}", img_io.getvalue())

    st.divider()
    st.download_button("📥 TÉLÉCHARGER LE PACK ZIP", data=zip_buf.getvalue(), file_name="shooting_ia.zip")
    
    cols = st.columns(3)
    for idx, item in enumerate(top):
        cols[idx % 3].image(item['pil'], caption=f"Score: {item['score']}/100")
