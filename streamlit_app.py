import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import google.generativeai as genai
import gc # Garbage Collector pour vider la RAM

# --- CONFIGURATION GEMINI ---
@st.cache_resource
def load_gemini_model():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

model = load_gemini_model()

# --- FONCTIONS DE TRAITEMENT ---
def process_analysis(file):
    """Analyse une seule image et libère la mémoire immédiatement."""
    img_pil = Image.open(file).convert('RGB')
    img_np = np.array(img_pil)
    
    # Analyse technique (Netteté)
    gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    score_tech = min((cv2.Laplacian(gris, cv2.CV_64F).var() / 150) * 50, 50)
    
    # Analyse Artistique (Gemini)
    score_art = 25
    if score_tech > 15 and model:
        try:
            prompt = "Note cette photo (expression, pose) sur 50. Réponds juste le chiffre."
            response = model.generate_content([prompt, img_pil])
            score_art = min(int(''.join(filter(str.isdigit, response.text))), 50)
        except: pass
        
    return int(score_tech + score_art)

def apply_retouche(image_bytes):
    """Applique la retouche studio sur les bytes pour économiser la RAM."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8)).apply(l)
    img_res = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    img_res = cv2.bilateralFilter(img_res, d=5, sigmaColor=30, sigmaSpace=30)
    return img_res

# --- INTERFACE ---
st.set_page_config(page_title="IA Studio Pro", layout="wide")
st.title("📸 Assistant Studio : Stable & Intelligent")

# Étape 1 : Upload avec cache forcé
uploaded_files = st.file_uploader("1. Déposez vos photos (JPEG)", type=['jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    # Bouton pour fixer les fichiers et éviter le vidage de cache
    if st.button("🚀 2. Lancer l'analyse du shooting"):
        resultats = []
        barre = st.progress(0)
        status = st.empty()
        
        for i, f in enumerate(uploaded_files):
            status.text(f"Analyse de {f.name}...")
            # On récupère les bytes une seule fois
            file_bytes = f.read()
            score = process_analysis(io.BytesIO(file_bytes))
            
            resultats.append({"nom": f.name, "score": score, "bytes": file_bytes})
            barre.progress((i + 1) / len(uploaded_files))
            
            # NETTOYAGE RAM : On force Python à vider les objets inutiles
            if i % 5 == 0: gc.collect()

        # Tri et Sélection (Top 20)
        top = sorted(resultats, key=lambda x: x['score'], reverse=True)[:20]

        # Création ZIP
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as xzip:
            for item in top:
                retouche = apply_retouche(item['bytes'])
                img_io = io.BytesIO()
                Image.fromarray(retouche).save(img_io, format='JPEG', quality=90)
                xzip.writestr(f"PRO_{item['nom']}", img_io.getvalue())
                gc.collect() # Nettoyage après chaque image retouchée

        st.divider()
        st.success("Analyse terminée !")
        st.download_button("📥 TÉLÉCHARGER LE PACK ZIP", data=zip_buf.getvalue(), file_name="shooting_pro.zip")

        # Affichage
        cols = st.columns(3)
        for idx, item in enumerate(top):
            cols[idx % 3].image(item['bytes'], caption=f"Score: {item['score']}/100")
