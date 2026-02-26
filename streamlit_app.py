import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

# --- CONFIGURATION IA ---
IA_DISPONIBLE = False
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    IA_DISPONIBLE = True
except:
    IA_DISPONIBLE = False

# --- FONCTIONS ---

def analyser_photo(image_np):
    score = 0
    gris = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    score += min((lap_var / 150) * 50, 50)
    if IA_DISPONIBLE:
        results = face_mesh.process(image_np)
        if results.multi_face_landmarks: score += 50
    return int(score)

def retoucher_image(image_np):
    """Retouche auto corrigée."""
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Correction ici : createCLAHE (avec un A)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

# --- INTERFACE ---
st.set_page_config(page_title="IA Photo Pro", layout="wide")

st.title("📸 Assistant Studio : Sélection & ZIP")

# Upload (La limite de 1Go est gérée par le fichier config.toml créé précédemment)
files = st.file_uploader("Glissez vos photos (JPEG)", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"Lancer l'analyse de {len(files)} photos"):
        resultats = []
        barre = st.progress(0)
        status = st.empty()
        
        for i, f in enumerate(files):
            status.text(f"Analyse de {f.name}...")
            img = Image.open(f).convert('RGB')
            img_np = np.array(img)
            
            score = analyser_photo(img_np)
            # On stocke l'image originale pour économiser de la RAM pendant le tri
            resultats.append({"nom": f.name, "score": score, "img": img_np})
            barre.progress((i + 1) / len(files))

        status.text("Création de la sélection finale...")
        
        # Sélectionner les 25 meilleures
        top_photos = sorted(resultats, key=lambda x: x['score'], reverse=True)[:25]

        # --- PRÉPARATION DU ZIP ---
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as xzip:
            for photo in top_photos:
                # Retouche au dernier moment pour économiser la mémoire
                img_ret = retoucher_image(photo['img'])
                img_pil = Image.fromarray(img_ret)
                
                # Conversion en bytes pour le ZIP
                img_io = io.BytesIO()
                img_pil.save(img_io, format='JPEG', quality=90)
                xzip.writestr(f"TOP_{photo['nom']}", img_io.getvalue())
        
        st.divider()
        
        # --- AFFICHAGE DES RÉSULTATS ---
        # 1. Bouton ZIP en haut pour un accès rapide
        st.download_button(
            label="🚀 TÉLÉCHARGER LE PACK ZIP (25 PHOTOS)",
            data=buf.getvalue(),
            file_name="selection_studio_ia.zip",
            mime="application/zip"
        )

        # 2. Grille d'aperçu
        st.subheader("Aperçu de la sélection :")
        cols = st.columns(3)
        for idx, item in enumerate(top_photos):
            with cols[idx % 3]:
                # On ré-affiche l'image retouchée
                img_preview = retoucher_image(item['img'])
                st.image(img_preview, caption=f"Score: {item['score']} - {item['nom']}")

        status.text("Traitement terminé !")
