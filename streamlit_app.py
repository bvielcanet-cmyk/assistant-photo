import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import google.generativeai as genai

# --- CONFIGURATION GEMINI ---
# Remplace par ta clé API ou utilise les secrets Streamlit
API_KEY = st.sidebar.text_input("Clé API Gemini (Optionnel)", type="password")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash') # Rapide et efficace pour l'image

# --- FONCTIONS TECHNIQUES (EXISTANTES) ---
def analyser_technique(image_np):
    """Note la netteté et les yeux (0-50)."""
    gris = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    return min((lap_var / 150) * 50, 50)

# --- NOUVELLE FONCTION : ANALYSE ARTISTIQUE GEMINI ---
def analyser_artistique_gemini(image_pil):
    """Demande à Gemini de noter l'esthétique et l'émotion."""
    if not API_KEY:
        return 25 # Score moyen si pas de clé
    
    prompt = """
    Agis comme un photographe de mode pro. Note cette photo sur 50 points selon :
    1. L'expression du visage (naturel, regard).
    2. La composition et la pose.
    3. L'esthétique générale.
    Réponds UNIQUEMENT par un chiffre entre 0 et 50.
    """
    try:
        response = model.generate_content([prompt, image_pil])
        # On extrait juste le chiffre de la réponse
        score = int(''.join(filter(str.isdigit, response.text)))
        return min(score, 50)
    except:
        return 25

# --- RETOUCHE STUDIO (LA MÊME) ---
def retoucher_studio_expert(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    img_clean = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
    return img_clean

# --- INTERFACE ---
st.set_page_config(page_title="IA Studio Artistique", layout="wide")
st.title("📸 Assistant Studio : Tri Technique + Analyse Artistique Gemini")

if not API_KEY:
    st.warning("⚠️ Entre ta clé API Gemini dans la barre latérale pour activer le 'Cerveau Artistique'.")

with st.form("upload-form", clear_on_submit=False):
    uploaded_files = st.file_uploader("Déposez votre shooting", type=['jpg', 'jpeg'], accept_multiple_files=True)
    submitted = st.form_submit_button("📁 Valider le dépôt")

if uploaded_files and submitted:
    resultats = []
    barre = st.progress(0)
    status = st.empty()
    
    for i, f in enumerate(uploaded_files):
        status.text(f"Analyse de {f.name}...")
        img_pil = Image.open(f).convert('RGB')
        img_np = np.array(img_pil)
        
        # 1. Score Technique (OpenCV)
        score_tech = analyser_technique(img_np)
        
        # 2. Score Artistique (Gemini) - On ne le fait que si la photo est assez nette
        score_art = 0
        if score_tech > 15: # On évite de gaspiller l'IA sur des photos floues
            score_art = analyser_artistique_gemini(img_pil)
        
        total = int(score_tech + score_art)
        resultats.append({"nom": f.name, "score": total, "img": img_np, "bytes": f.getvalue()})
        barre.progress((i + 1) / len(uploaded_files))

    top_selection = sorted(resultats, key=lambda x: x['score'], reverse=True)[:15] # Top 15

    # Création ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as xzip:
        for item in top_selection:
            final = retoucher_studio_expert(item['img'])
            img_io = io.BytesIO()
            Image.fromarray(final).save(img_io, format='JPEG', quality=95)
            xzip.writestr(f"PRO_{item['nom']}", img_io.getvalue())

    st.divider()
    st.download_button("📥 TÉLÉCHARGER LE PACK ARTISTIQUE (ZIP)", data=zip_buf.getvalue(), file_name="shooting_gemini.zip")

    # Aperçu
    cols = st.columns(3)
    for idx, item in enumerate(top_selection):
        with cols[idx % 3]:
            prev = retoucher_studio_expert(item['img'])
            st.image(prev, caption=f"Score Artistique : {item['score']}/100")
