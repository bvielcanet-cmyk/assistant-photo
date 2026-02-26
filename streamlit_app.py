import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from google import genai  # Bibliothèque officielle 2026
from skimage import exposure 
import gc
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION IA MODERNE ---
@st.cache_resource
def get_ai_client():
    try:
        # Récupère la clé API dans les Secrets de Streamlit Cloud
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except:
        return None

client = get_ai_client()

# --- FONCTIONS TURBO ---

def analyser_technique_rapide(file):
    """Analyse technique sur miniature pour ne pas saturer la RAM."""
    try:
        # Lecture sans charger la HD en mémoire
        img_pil = Image.open(file).convert('RGB')
        thumb = img_pil.copy()
        thumb.thumbnail((600, 600)) 
        img_np = np.array(thumb)
        
        # Calcul de netteté (Laplacien)
        gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        score_tech = min((cv2.Laplacian(gris, cv2.CV_64F).var() / 120) * 40, 40)
        
        return {"score_tech": score_tech, "thumb": thumb, "nom": file.name, "file_ref": file}
    except:
        return None

def analyser_artistique_batch(items):
    """Analyse par paquets de 5 avec le nouveau client GenAI 2026."""
    if not client:
        return [25] * len(items)
    
    prompt = f"Note ces {len(items)} photos sur 60 points (pose, charisme). Réponds uniquement par les chiffres séparés par des virgules."
    
    try:
        # Préparation des contenus (Texte + Images miniatures)
        contents = [prompt]
        for it in items:
            contents.append(it['thumb'].resize((320, 320)))
            
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents
        )
        # Extraction des scores
        scores = [int(s.strip()) for s in response.text.split(',') if s.strip().isdigit()]
        return scores if len(scores) == len(items) else [30] * len(items)
    except:
        return [30] * len(items)

def retouche_turbo(img_np, intensite, ref_np=None):
    """Applique le Style Transfer et la Retouche dosable."""
    # 1. Style Transfer (Color Matching)
    if ref_np is not None:
        try:
            img_np = exposure.match_histograms(img_np, ref_np, channel_axis=-1).astype(np.uint8)
        except:
            pass
    
    # 2. Éclat Studio (LAB)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.2 + (intensite * 0.8)).apply(l)
    img_base = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    
    # 3. Lissage de peau (Bilateral) dosé
    sigma = int(10 + (intensite * 60))
    img_lisse = cv2.bilateralFilter(img_base, d=5, sigmaColor=sigma, sigmaSpace=sigma)
    
    # 4. Fusion selon l'intensité choisie
    return cv2.addWeighted(img_base, 1.0 - intensite, img_lisse, intensite, 0)

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Turbo 2026", layout="wide")

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Réglages Turbo")
    nb_final = st.slider("Photos à garder dans le pack", 1, 50, 15)
    intensite_val = st.slider("Intensité de la retouche (Peau/Éclat)", 0, 100, 50)
    
    st.divider()
    st.subheader("🎨 Style Transfer")
    ref_file = st.file_uploader("Photo de référence (Couleurs)", type=['jpg', 'jpeg'])
    
    st.divider()
    if client:
        st.success("✅ IA Gemini 2026 : Connectée")
    else:
        st.error("❌ IA déconnectée (Vérifiez GEMINI_API_KEY)")

st.title("🚀 Assistant Studio Turbo : Tri & Retouche Intelligent")
st.info("Astuce : Sélectionnez toutes vos photos (Ctrl+A) et glissez-les ici d'un coup.")

# Zone d'upload
files = st.file_uploader("", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"⚡ Lancer le Traitement de {len(files)} photos"):
        data_pool = []
        barre = st.progress(0)
        status = st.empty()
        
        # ÉTAPE 1 : Analyse Technique (Multi-threading pour la vitesse)
        status.text("⚙️ Analyse technique ultra-rapide...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(analyser_technique_rapide, files))
        
        data_pool = [r for r in results if r is not None]
        barre.progress(0.4)
        gc.collect()

        # ÉTAPE 2 : Analyse Artistique Gemini (Mode Batch)
        if client and len(data_pool) > 0:
            status.text("🧠 Analyse artistique Gemini (Sélection des meilleures poses)...")
            # On trie d'abord par netteté pour n'envoyer que les 25 meilleures à l'IA
            pool_trie = sorted(data_pool, key=lambda x: x['score_tech'], reverse=True)[:25]
            
            for i in range(0, len(pool_trie), 5):
                batch = pool_trie[i:i+5]
                scores_art = analyser_artistique_batch(batch)
                for j, s in enumerate(scores_art):
                    batch[j]['score_total'] = batch[j]['score_tech'] + s
                
                # Mise à jour de la barre (de 40% à 80%)
                barre.progress(0.4 + (i / len(pool_trie)) * 0.4)
            
            finalistes = sorted(pool_trie, key=lambda x: x.get('score_total', 0), reverse=True)[:nb_final]
        else:
            # Si pas d'IA, on prend juste les meilleures en netteté
            finalistes = sorted(data_pool, key=lambda x: x['score_tech'], reverse=True)[:nb_final]

        # ÉTAPE 3 : Retouche HD et Création du ZIP
        ref_np = np.array(Image.open(ref_file).convert('RGB')) if ref_file else None
        zip_buf = io.BytesIO()
        
        with zipfile.ZipFile(zip_buf, "w") as xzip:
            status.text("Finalisation et Retouche HD du pack ZIP...")
            for idx, item in enumerate(finalistes):
                # On ouvre l'original HD seulement maintenant pour économiser la RAM
                img_hd = np.array(Image.open(item['file_ref']).convert('RGB'))
                img_edit = retouche_turbo(img_hd, intensite_val/100.0, ref_np)
                
                # Sauvegarde dans le ZIP
                img_io = io.BytesIO()
                Image.fromarray(img_edit).save(img_io, format='JPEG', quality=90)
                xzip.writestr(f"PRO_{item['nom']}", img_io.getvalue())
                
                # Nettoyage mémoire après chaque photo lourde
                gc.collect()
        
        barre.progress(1.0)
        st.divider()
        st.success(f"Traitement terminé ! {len(finalistes)} photos sélectionnées.")
        
        # Bouton de téléchargement final
        st.download_button(
            label="📥 TÉLÉCHARGER LE PACK ZIP HD", 
            data=zip_buf.getvalue(), 
            file_name="mon_shooting_ia_turbo.zip",
            mime="application/zip"
        )
        
        # Affichage des miniatures gagnantes
        st.subheader("Aperçu de la sélection")
        cols = st.columns(3)
        for idx, item in enumerate(finalistes):
            cols[idx % 3].image(item['thumb'], caption=f"Score: {item.get('score_total', 0):.0f}/100")
        
        status.empty()
