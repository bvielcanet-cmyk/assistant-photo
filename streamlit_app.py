import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import google.generativeai as genai
from skimage import exposure 
import gc
from concurrent.futures import ThreadPoolExecutor # Pour le multi-threading

# --- CONFIGURATION IA ---
@st.cache_resource
def load_models():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except: return None

gemini_model = load_models()

# --- FONCTIONS RAPIDES ---

def analyser_technique_rapide(file):
    """Analyse ultra-rapide sur miniature."""
    img_pil = Image.open(file).convert('RGB')
    # On travaille sur une version réduite pour la vitesse
    thumb = img_pil.copy()
    thumb.thumbnail((512, 512))
    img_np = np.array(thumb)
    
    gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    score_tech = min((lap_var / 120) * 40, 40) # Max 40 pts
    
    return score_tech, img_pil

def analyser_artistique_batch(items):
    """Analyse Gemini sur un petit groupe de photos pour réduire les appels API."""
    if not gemini_model: return [25] * len(items)
    
    imgs_pil = [it['pil'].resize((400, 400)) for it in items]
    prompt = f"Note ces {len(items)} photos de mode sur 60 points (charisme, pose). Réponds par une liste de chiffres séparés par des virgules."
    
    try:
        response = gemini_model.generate_content([prompt] + imgs_pil)
        scores = [int(s.strip()) for s in response.text.split(',') if s.strip().isdigit()]
        return scores if len(scores) == len(items) else [30] * len(items)
    except:
        return [30] * len(items)

def retouche_rapide(img_np, intensite, ref_np=None):
    """Retouche optimisée."""
    if ref_np is not None:
        img_np = exposure.match_histograms(img_np, ref_np, channel_axis=-1).astype(np.uint8)
    
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.2 + (intensite * 0.8)).apply(l)
    img_base = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    
    sigma = int(10 + (intensite * 60))
    img_lisse = cv2.bilateralFilter(img_base, d=5, sigmaColor=sigma, sigmaSpace=sigma)
    return cv2.addWeighted(img_base, 1.0 - intensite, img_lisse, intensite, 0)

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Turbo", layout="wide")

with st.sidebar:
    st.header("⚡ Mode Turbo")
    nb_final = st.slider("Photos à garder", 1, 50, 15)
    intensite_val = st.slider("Intensité Retouche", 0, 100, 50)
    ref_file = st.file_uploader("🖼️ Référence Style", type=['jpg', 'jpeg'])

st.title("🚀 Assistant Studio Turbo : Vitesse & Intelligence")

files = st.file_uploader("Déposez vos photos", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"⚡ Lancer le traitement Rapide ({len(files)} photos)"):
        data_pool = []
        barre = st.progress(0)
        status = st.empty()
        
        # ÉTAPE 1 : Analyse Technique en Parallèle (Multi-threading)
        status.text("⚙️ Analyse technique ultra-rapide...")
        with ThreadPoolExecutor() as executor:
            # On lance l'analyse technique sur tous les coeurs du processeur
            results = list(executor.map(analyser_technique_rapide, files))
            
        for i, (score_tech, img_pil) in enumerate(results):
            data_pool.append({
                "nom": files[i].name, 
                "score_tech": score_tech, 
                "pil": img_pil, 
                "np": np.array(img_pil)
            })
            barre.progress((i + 1) / (len(files) * 2)) # Première moitié de la barre

        # ÉTAPE 2 : Analyse Artistique Groupée (Batch)
        status.text("🧠 Analyse Artistique Gemini (Mode Batch)...")
        # On ne traite que les 30 meilleures en technique pour gagner du temps
        pool_trie = sorted(data_pool, key=lambda x: x['score_tech'], reverse=True)[:30]
        
        # On traite par paquets de 5 photos pour Gemini
        for i in range(0, len(pool_trie), 5):
            batch = pool_trie[i:i+5]
            scores_art = analyser_artistique_batch(batch)
            for j, score in enumerate(scores_art):
                batch[j]['score_total'] = batch[j]['score_tech'] + score
            barre.progress(0.5 + (i / len(pool_trie)) / 2)

        # ÉTAPE 3 : Filtrage & ZIP
        selection_finale = sorted(pool_trie, key=lambda x: x.get('score_total', 0), reverse=True)[:nb_final]
        
        ref_np = np.array(Image.open(ref_file).convert('RGB')) if ref_file else None
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as xzip:
            for item in selection_finale:
                final_img = retouche_rapide(item['np'], intensite_val/100.0, ref_np)
                img_io = io.BytesIO()
                Image.fromarray(final_img).save(img_io, format='JPEG', quality=90)
                xzip.writestr(f"TOP_{item['nom']}", img_io.getvalue())
                gc.collect()

        st.divider()
        st.download_button("📥 TÉLÉCHARGER LE ZIP TURBO", data=zip_buf.getvalue(), file_name="shooting_turbo.zip")
        
        cols = st.columns(3)
        for idx, item in enumerate(selection_finale):
            cols[idx % 3].image(item['pil'], caption=f"Score: {item.get('score_total', 0):.0f}/100")
