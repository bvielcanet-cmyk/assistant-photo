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
    mp_face_mesh = mp.solutions.face_mesh
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
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createClHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

# --- INTERFACE ---
st.set_page_config(page_title="IA Photo Pro", layout="wide")

st.title("📸 Assistant Studio : Tri & ZIP Automatique")

files = st.file_uploader("Upload (Limite 1 Go)", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"Analyser {len(files)} photos"):
        resultats = []
        barre = st.progress(0)
        
        for i, f in enumerate(files):
            img = Image.open(f).convert('RGB')
            img_np = np.array(img)
            score = analyser_photo(img_np)
            
            # On stocke l'image retouchée pour le ZIP
            img_retouchee = retoucher_image(img_np)
            resultats.append({"nom": f.name, "score": score, "img": img_retouchee})
            barre.progress((i + 1) / len(files))

        # Sélectionner les 25 meilleures
        top_photos = sorted(resultats, key=lambda x: x['score'], reverse=True)[:25]

        # --- CRÉATION DU FICHIER ZIP ---
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "x") as csv_zip:
            for photo in top_photos:
                # Conversion numpy -> JPEG pour le ZIP
                img_pil = Image.fromarray(photo['img'])
                img_byte_arr = io.BytesIO()
                img_pil.save(img_byte_arr, format='JPEG')
                csv_zip.writestr(f"TOP_{photo['nom']}", img_byte_arr.getvalue())
        
        st.divider()
        
        # Bouton de téléchargement global
        st.download_button(
            label="🚀 TÉLÉCHARGER LA SÉLECTION (ZIP)",
            data=buf.getvalue(),
            file_name="selection_retouchee.zip",
            mime="application/zip"
        )

        # Affichage aperçu
        cols = st.columns(3)
        for idx, item in enumerate(top_photos):
            cols[idx % 3].image(item['img'], caption=f"Score: {item['score']}")
