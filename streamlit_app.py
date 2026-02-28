import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from google import genai
from skimage import exposure 
import gc
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION IA ---
@st.cache_resource
def get_ai_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except:
        return None

client = get_ai_client()

# --- FONCTIONS ---

def analyser_technique_rapide(file):
    try:
        # On lit les bytes du fichier directement pour Windows
        file_bytes = file.getvalue()
        img_pil = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        thumb = img_pil.copy()
        thumb.thumbnail((500, 500)) 
        img_np = np.array(thumb)
        
        gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        score_tech = min((cv2.Laplacian(gris, cv2.CV_64F).var() / 120) * 40, 40)
        
        return {"score_tech": score_tech, "thumb": thumb, "nom": file.name, "file_ref": file}
    except Exception as e:
        return None

def analyser_artistique_batch(items):
    if not client: return [25] * len(items)
    prompt = f"Note ces {len(items)} photos sur 60 points (pose, charisme). Réponds uniquement par les chiffres séparés par des virgules."
    try:
        contents = [prompt]
        for it in items:
            contents.append(it['thumb'].resize((300, 300)))
        response = client.models.generate_content(model="gemini-1.5-flash", contents=contents)
        scores = [int(s.strip()) for s in response.text.split(',') if s.strip().isdigit()]
        return scores if len(scores) == len(items) else [30] * len(items)
    except:
        return [30] * len(items)

def retouche_turbo(img_np, intensite, ref_np=None):
    if ref_np is not None:
        try: img_np = exposure.match_histograms(img_np, ref_np, channel_axis=-1).astype(np.uint8)
        except: pass
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.2 + (intensite * 0.8)).apply(l)
    img_base = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    sigma = int(10 + (intensite * 60))
    img_lisse = cv2.bilateralFilter(img_base, d=5, sigmaColor=sigma, sigmaSpace=sigma)
    return cv2.addWeighted(img_base, 1.0 - intensite, img_lisse, intensite, 0)

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Turbo", layout="wide")

# Utilisation du State pour Windows afin de garder les résultats après le clic
if 'finalistes' not in st.session_state:
    st.session_state.finalistes = []
if 'zip_data' not in st.session_state:
    st.session_state.zip_data = None

with st.sidebar:
    st.header("⚙️ Réglages")
    nb_final = st.slider("Photos à garder", 1, 50, 15)
    intensite_val = st.slider("Intensité Retouche", 0, 100, 50)
    st.divider()
    ref_file = st.file_uploader("🖼️ Style de référence", type=['jpg', 'jpeg'])

st.title("🚀 Assistant Studio : Tri & Retouche")

files = st.file_uploader("Glissez vos photos ici", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"⚡ Analyser {len(files)} photos"):
        with st.spinner("Analyse en cours..."):
            data_pool = []
            
            # 1. ANALYSE TECHNIQUE
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(analyser_technique_rapide, files))
            data_pool = [r for r in results if r is not None]
            
            # 2. IA GEMINI
            if client and data_pool:
                pool_trie = sorted(data_pool, key=lambda x: x['score_tech'], reverse=True)[:25]
                for i in range(0, len(pool_trie), 5):
                    batch = pool_trie[i:i+5]
                    scores = analyser_artistique_batch(batch)
                    for j, s in enumerate(scores):
                        batch[j]['score_total'] = batch[j]['score_tech'] + s
                
                final_results = sorted(pool_trie, key=lambda x: x.get('score_total', 0), reverse=True)[:nb_final]
            else:
                final_results = sorted(data_pool, key=lambda x: x['score_tech'], reverse=True)[:nb_final]

            # 3. CRÉATION ZIP
            ref_np = np.array(Image.open(ref_file).convert('RGB')) if ref_file else None
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as xzip:
                for item in final_results:
                    img_hd = np.array(Image.open(item['file_ref']).convert('RGB'))
                    img_edit = retouche_turbo(img_hd, intensite_val/100.0, ref_np)
                    img_io = io.BytesIO()
                    Image.fromarray(img_edit).save(img_io, format='JPEG', quality=85)
                    xzip.writestr(f"PRO_{item['nom']}", img_io.getvalue())
            
            # Sauvegarde dans le State pour forcer l'affichage
            st.session_state.finalistes = final_results
            st.session_state.zip_data = zip_buf.getvalue()
            st.rerun() # On force le rafraîchissement de la page

# --- AFFICHAGE DES RÉSULTATS (Hors du bouton) ---
if st.session_state.zip_data:
    st.divider()
    st.success(f"Analyse terminée ! {len(st.session_state.finalistes)} photos prêtes.")
    
    st.download_button(
        label="📥 TÉLÉCHARGER LE PACK ZIP HD", 
        data=st.session_state.zip_data, 
        file_name="mon_shooting_ia.zip",
        mime="application/zip"
    )
    
    st.subheader("Aperçu de la sélection")
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.finalistes):
        cols[idx % 3].image(item['thumb'], caption=f"Score: {item.get('score_total', 0):.0f}/100")
