import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import google.generativeai as genai

# --- CONFIGURATION SÉCURISÉE ---
IA_ARTISTIQUE_ACTIVE = False
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    IA_ARTISTIQUE_ACTIVE = True
except:
    pass

# --- FONCTIONS ---
def analyser_technique(image_np):
    gris = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    return min((lap_var / 150) * 50, 50)

def analyser_artistique(image_pil):
    if not IA_ARTISTIQUE_ACTIVE: return 25
    prompt = "Note cette photo de mode sur 50 points (expression, pose). Réponds UNIQUEMENT par un nombre."
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
st.set_page_config(page_title="Studio IA Pro", layout="wide")
st.title("📸 Assistant Studio : Dépôt Haute Performance")

# Initialisation de la mémoire tampon pour éviter les pertes au chargement
if 'shooting_prets' not in st.session_state:
    st.session_state.shooting_prets = False

# Zone d'Upload (HORS DU FORMULAIRE pour plus de fluidité)
uploaded_files = st.file_uploader(
    "1. Glissez vos photos ici (Attendez que toutes les barres soient terminées)", 
    type=['jpg', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} fichiers détectés par le serveur.")
    
    # On utilise le bouton pour "fixer" les fichiers dans la mémoire
    if st.button("2. Valider et Préparer l'Analyse"):
        st.session_state.shooting_prets = True
        st.success("Photos verrouillées. Prêt pour l'IA.")

if st.session_state.shooting_prets and uploaded_files:
    nb_top = st.slider("Nombre de photos à garder", 1, 100, 20)
    
    if st.button("🚀 3. Lancer le Cerveau IA Artistique"):
        resultats = []
        barre = st.progress(0)
        status = st.empty()
        
        for i, f in enumerate(uploaded_files):
            status.text(f"Traitement de {f.name}...")
            # On lit le fichier UNIQUEMENT au moment du besoin
            img_bytes = f.read()
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            s_tech = analyser_technique(np.array(img_pil))
            s_art = analyser_artistique(img_pil) if s_tech > 15 else 0
            
            resultats.append({"nom": f.name, "score": s_tech + s_art, "bytes": img_bytes})
            barre.progress((i + 1) / len(uploaded_files))

        top = sorted(resultats, key=lambda x: x['score'], reverse=True)[:nb_top]
        
        # Création ZIP
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as xzip:
            for item in top:
                img_pil = Image.open(io.BytesIO(item['bytes'])).convert('RGB')
                retouche = retoucher_pro(np.array(img_pil))
                img_io = io.BytesIO()
                Image.fromarray(retouche).save(img_io, format='JPEG', quality=95)
                xzip.writestr(f"PRO_{item['nom']}", img_io.getvalue())

        st.divider()
        st.download_button("📥 TÉLÉCHARGER LE PACK ZIP", data=zip_buf.getvalue(), file_name="shooting_ia_pro.zip")
        
        cols = st.columns(3)
        for idx, item in enumerate(top):
            cols[idx % 3].image(Image.open(io.BytesIO(item['bytes'])), caption=f"Score: {item['score']}/100")
        
        status.text("✅ Terminé !")
