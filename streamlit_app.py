import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from google import genai # Nouvelle bibliothèque 2026
from skimage import exposure 
import gc
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION IA MODERNE ---
@st.cache_resource
def get_ai_client():
    try:
        # Utilise le nouveau client Google GenAI
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        return client
    except: return None

client = get_ai_client()

# --- FONCTIONS TURBO ---

def analyser_technique_rapide(file):
    try:
        img_pil = Image.open(file).convert('RGB')
        thumb = img_pil.copy()
        thumb.thumbnail((600, 600)) 
        img_np = np.array(thumb)
        
        gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        score_tech = min((cv2.Laplacian(gris, cv2.CV_64F).var() / 120) * 40, 40)
        
        return {"score_tech": score_tech, "thumb": thumb, "nom": file.name, "file_ref": file}
    except: return None

def analyser_artistique_batch(items):
    """Nouveau client GenAI 2026 en mode Batch."""
    if not client: return [25] * len(items)
    
    prompt = f"Note ces {len(items)} photos sur 60 points (pose, charisme). Réponds uniquement par les chiffres séparés par des virgules."
    
    try:
        # Conversion des images pour le nouveau client
        contents = [prompt]
        for it in items:
            contents.append(it['thumb'].resize((320, 320)))
            
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents
        )
        scores = [int(s.strip()) for s in response.text.split(',') if s.strip().isdigit()]
        return scores if len(scores) == len(items) else [30] * len(items)
    except: return [30] * len(items)

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
st.set_page_config(page_title="Studio IA Turbo 2026", layout="wide")

with st.sidebar:
    st.header("⚡ Réglages Turbo")
    nb_final = st.slider("Photos à garder", 1, 50, 15)
    intensite_val = st.slider("Intensité Retouche", 0, 100, 50)
    st.divider()
    ref_file = st.file_uploader("🖼️ Style de référence", type=['jpg', 'jpeg'])

st.title("🚀 Assistant Studio Turbo : Version 2026")

files = st.file_uploader("Déposez vos photos (JPEG)", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"⚡ Lancer le Traitement ({len(files)} photos)"):
        data_pool = []
        barre = st.progress(0)
        status = st.empty()
        
        # 1. ANALYSE TECHNIQUE
        status.text("⚙️ Analyse technique multi-coeurs...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(analyser_technique_rapide, files))
        
        data_pool = [r for r in results if r is not None]
        barre.progress(0.4)
        gc.collect()

        # 2. ANALYSE ARTISTIQUE BATCH
        if client:
            status.text("🧠 Analyse artistique Gemini (Client 2026)...")
            pool_trie = sorted(data_pool, key=lambda x: x['score_tech'], reverse=True)[:25]
            for i in range(0, len(pool_trie), 5):
                batch = pool_trie[i:i+5]
                scores_art = analyser_artistique_batch(batch)
                for j, s in enumerate(scores_art):
                    batch[j]['score_total'] = batch[j]['score_tech'] + s
                barre.progress(0.4 + (i / len(pool_trie)) * 0.4)
        else:
            for item in data_pool: item['score_total'] = item['score_tech'] + 25

        # 3. SÉLECTION ET ZIP
        finalistes = sorted(data_pool, key=lambda x: x.get('score_total', 0), reverse=True)[:nb_final]
        
        ref_np = np.array(Image.open(ref_file).convert('RGB')) if ref_file else None
        zip_buf = io.BytesIO()
        
        with zipfile.ZipFile(zip_buf, "w") as xzip:
            status.text("Finalisation du pack ZIP HD...")
            for item in finalistes:
                img_hd = np.array(Image.open(item['file_ref']).convert('RGB'))
                img_edit = retouche_turbo(img_hd, intensite_val/100.0, ref_np)
                img_io = io.BytesIO()
                Image.fromarray(img_edit).save(img_io, format='JPEG', quality=90)
                xzip.writestr(f"PRO_{item['nom']}", img_io.getvalue())
                gc.collect()
        
        barre.progress(1.0)
        st.divider()
        st.download_button("📥 TÉLÉCHARGER LE ZIP", data=zip_buf.getvalue(), file_name="shooting_ia.zip")
        
        cols = st.columns(3)
        for idx, item in enumerate(finalistes):
            cols[idx % 3].image(item['thumb'], caption=f"Score: {item.get('score_total', 0):.0f}/100")
        
        status.success("Traitement terminé !")
