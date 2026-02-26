import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import google.generativeai as genai
from skimage import exposure 
import gc

# --- CONFIGURATION IA SÉCURISÉE ---
@st.cache_resource
def load_models():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        # MediaPipe pour le visage
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        return model, face_mesh
    except: return None, None

gemini_model, mp_mesh = load_models()

# --- FONCTIONS DE SÉLECTION INTELLIGENTE ---

def analyser_score_complet(img_pil, img_np):
    """Note de 0 à 100 : Technique (30) + Regard (30) + Gemini (40)."""
    score = 0
    h, w, _ = img_np.shape
    gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 1. NETTETÉ LOCALE (30 pts)
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    score += min((lap_var / 150) * 30, 30)

    # 2. ANALYSE DU REGARD (30 pts)
    if mp_mesh:
        results = mp_mesh.process(img_np)
        if results.multi_face_landmarks:
            score += 15 # Visage présent
            face = results.multi_face_landmarks[0]
            # Test ouverture yeux (points 159 et 145)
            p_sup, p_inf = face.landmark[159].y, face.landmark[145].y
            if abs(p_inf - p_sup) > 0.012: score += 15
    
    # 3. GEMINI ARTISTIQUE (40 pts)
    # On n'appelle Gemini que si la photo est un minimum nette (>10)
    if gemini_model and score > 10:
        try:
            prompt = "Note cette photo de mode sur 40 points (pose, charisme, émotion). Réponds juste le chiffre."
            response = gemini_model.generate_content([prompt, img_pil])
            score_ia = int(''.join(filter(str.isdigit, response.text)))
            score += min(score_ia, 40)
        except: score += 20
        
    return int(score)

def filtrer_doublons(liste_triee, nb_voulus):
    """Élimine les poses trop similaires pour garantir la variété."""
    finalistes = []
    for item in liste_triee:
        is_duplicate = False
        curr_small = cv2.resize(item['np'], (64, 64))
        for f in finalistes:
            prev_small = cv2.resize(f['np'], (64, 64))
            res = cv2.matchTemplate(curr_small, prev_small, cv2.TM_CCOEFF_NORMED)[0][0]
            if res > 0.88: # Seuil de ressemblance
                is_duplicate = True
                break
        if not is_duplicate:
            finalistes.append(item)
        if len(finalistes) >= nb_voulus: break
    return finalistes

# --- FONCTION RETOUCHE DOSABLE ---

def apply_retouche_expert(img_np, intensite, ref_np=None):
    # Style Transfer si référence
    if ref_np is not None:
        try: img_np = exposure.match_histograms(img_np, ref_np, channel_axis=-1).astype(np.uint8)
        except: pass
    
    # Retouche Studio
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.2 + (intensite * 0.8)).apply(l)
    img_base = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    
    # Lissage peau
    sigma = int(10 + (intensite * 60))
    img_lisse = cv2.bilateralFilter(img_base, d=7, sigmaColor=sigma, sigmaSpace=sigma)
    
    return cv2.addWeighted(img_base, 1.0 - intensite, img_lisse, intensite, 0)

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Director", layout="wide")

with st.sidebar:
    st.header("⚙️ Paramètres de Sélection")
    nb_final = st.slider("Nombre de photos à garder", 1, 50, 15)
    
    st.header("✨ Retouche IA")
    intensite_val = st.slider("Intensité du rendu (Peau/Éclat)", 0, 100, 50)
    
    st.divider()
    ref_file = st.file_uploader("🖼️ Style de référence (Optionnel)", type=['jpg', 'jpeg'])

st.title("📸 Assistant Studio : Sélection Intelligente & Retouche")

files = st.file_uploader("Déposez vos photos (JPEG)", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"🚀 Analyser et Créer la Sélection"):
        data_pool = []
        barre = st.progress(0)
        status = st.empty()
        
        # 1. Analyse Individuelle
        for i, f in enumerate(files):
            status.text(f"Analyse IA de {f.name}...")
            img_pil = Image.open(f).convert('RGB')
            img_np = np.array(img_pil)
            
            score = analyser_score_complet(img_pil, img_np)
            data_pool.append({"nom": f.name, "score": score, "np": img_np, "pil": img_pil})
            
            barre.progress((i + 1) / len(files))
            gc.collect()

        # 2. Tri et Filtrage des doublons
        status.text("🧠 Optimisation de la variété du pack...")
        pool_trie = sorted(data_pool, key=lambda x: x['score'], reverse=True)
        selection_finale = filtrer_doublons(pool_trie, nb_final)

        # 3. Traitement et ZIP
        ref_np = np.array(Image.open(ref_file).convert('RGB')) if ref_file else None
        zip_buf = io.BytesIO()
        
        with zipfile.ZipFile(zip_buf, "w") as xzip:
            st.subheader(f"✅ Top {len(selection_finale)} - Aperçu Avant/Après")
            for idx, item in enumerate(selection_finale):
                # Retouche finale avec dosage
                final_img = apply_retouche_expert(item['np'], intensite_val/100.0, ref_np)
                
                # ZIP
                img_io = io.BytesIO()
                Image.fromarray(final_img).save(img_io, format='JPEG', quality=95)
                xzip.writestr(f"TOP_{item['nom']}", img_io.getvalue())
                
                # Affichage UX
                with st.expander(f"Photo : {item['nom']} (Score: {item['score']}/100)"):
                    c1, c2 = st.columns(2)
                    c1.image(item['np'], caption="Original")
                    c2.image(final_img, caption=f"Retouche IA ({intensite_val}%)")
                gc.collect()

        st.divider()
        st.download_button("📥 TÉLÉCHARGER LE PACK ZIP FINAL", data=zip_buf.getvalue(), file_name="shooting_ia_expert.zip")
