import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

# --- CONFIGURATION IA SÉCURISÉE ---
IA_DISPONIBLE = False
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    IA_DISPONIBLE = True
except Exception as e:
    IA_DISPONIBLE = False
    IA_ERREUR = str(e)

# --- FONCTIONS TECHNIQUES ---

def analyser_photo(image_np):
    """Note la photo sur 100 points (Netteté + IA Yeux)."""
    score = 0
    gris = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 1. NETTETÉ (50 pts)
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    score_nettete = min((lap_var / 150) * 50, 50) 
    score += score_nettete

    # 2. IA FACIALE (50 pts)
    if IA_DISPONIBLE:
        results = face_mesh.process(image_np)
        if results.multi_face_landmarks:
            score += 25 # Visage détecté
            for face_landmarks in results.multi_face_landmarks:
                p_sup = face_landmarks.landmark[159].y
                p_inf = face_landmarks.landmark[145].y
                if abs(p_inf - p_sup) > 0.012: # Oeil ouvert
                    score += 25
        else:
            score -= 10
    else:
        score += score_nettete # Compensation si IA HS

    return int(max(0, min(score, 100)))

def retoucher_image(image_np):
    """Retouche auto Pro : CLAHE + Sharpening."""
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    # Filtre de netteté léger
    noyau = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, noyau)
    return img

# --- INTERFACE ---
st.set_page_config(page_title="IA Photo Studio Pro", layout="wide")

st.title("📸 Assistant Studio : Sélection & ZIP Automatique")
st.info("Astuce : Pour de gros shootings, glissez vos fichiers par lots de 50 pour éviter les ralentissements réseau.")

# Barre latérale
with st.sidebar:
    st.header("⚙️ Réglages")
    if IA_DISPONIBLE:
        st.success("✅ IA Faciale active")
    else:
        st.error("⚠️ Mode netteté seule")
    
    nb_a_garder = st.slider("Nombre de meilleures photos à extraire", 1, 100, 25)

# Zone d'upload
uploaded_files = st.file_uploader(
    "Glissez vos photos (JPEG uniquement)", 
    type=['jpg', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"📂 **{len(uploaded_files)}** fichiers prêts pour l'analyse.")
    
    if st.button(f"🚀 Lancer le tri de {len(uploaded_files)} photos"):
        resultats = []
        barre = st.progress(0)
        status = st.empty()
        
        # Phase 1 : Analyse (Légère en RAM)
        for i, f in enumerate(uploaded_files):
            status.text(f"Analyse technique de {f.name}...")
            img = Image.open(f).convert('RGB')
            img_np = np.array(img)
            
            score = analyser_photo(img_np)
            # On ne stocke que le score et les bytes pour économiser la RAM
            resultats.append({"nom": f.name, "score": score, "bytes": f.getvalue()})
            barre.progress((i + 1) / len(uploaded_files))

        status.text("Tri et préparation de la sélection...")
        
        # Sélection du TOP
        top_selection = sorted(resultats, key=lambda x: x['score'], reverse=True)[:nb_a_garder]

        # Phase 2 : Création du ZIP (Retouche au vol)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as xzip:
            for item in top_selection:
                # Charger et retoucher uniquement les gagnantes
                img_pil = Image.open(io.BytesIO(item['bytes'])).convert('RGB')
                img_ret = retoucher_image(np.array(img_pil))
                
                # Sauvegarde en JPEG haute qualité
                out_pil = Image.fromarray(img_ret)
                out_io = io.BytesIO()
                out_pil.save(out_io, format='JPEG', quality=95)
                xzip.writestr(f"TOP_{item['nom']}", out_io.getvalue())
        
        st.divider()
        
        # BOUTON DE TÉLÉCHARGEMENT PRINCIPAL
        st.download_button(
            label=f"📥 TÉLÉCHARGER LE ZIP ({len(top_selection)} photos)",
            data=zip_buffer.getvalue(),
            file_name="ma_selection_ia.zip",
            mime="application/zip"
        )

        # Phase 3 : Aperçu visuel
        st.subheader("Aperçu de la sélection")
        cols = st.columns(3)
        for idx, item in enumerate(top_selection):
            with cols[idx % 3]:
                # On réaffiche une version légère pour l'aperçu
                img_prev = Image.open(io.BytesIO(item['bytes']))
                st.image(img_prev, caption=f"Score: {item['score']}/100 - {item['nom']}")

        status.text("✅ Traitement terminé avec succès !")
