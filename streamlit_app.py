import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Configuration IA
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def analyser_photo(image_np):
    score = 0
    gris = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Netteté
    score_flou = cv2.Laplacian(gris, cv2.CV_64F).var()
    if score_flou > 100: score += 50
    
    # Visage et Yeux
    results = face_mesh.process(image_np)
    if results.multi_face_landmarks:
        score += 50
    return score

def retoucher(image_np):
    # Correction simple : contraste et luminosité
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_ret = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)
    return img_ret

# --- INTERFACE STREAMLIT ---
st.title("📸 Assistant Studio IA")
st.write("Uploadez votre shooting, je sélectionne et retouche les meilleures pépites.")

uploaded_files = st.file_uploader("Choisir des photos JPG", accept_multiple_files=True, type=['jpg', 'jpeg'])

if uploaded_files:
    if st.button("Lancer le tri et la retouche"):
        resultats = []
        barre_progression = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # Conversion fichier -> OpenCV
            img = Image.open(file)
            img_np = np.array(img)
            
            score = analyser_photo(img_np)
            resultats.append({"score": score, "img": img_np, "nom": file.name})
            barre_progression.progress((i + 1) / len(uploaded_files))

        # Sélection des 3 meilleures (pour l'essai)
        top_photos = sorted(resultats, key=lambda x: x['score'], reverse=True)[:3]

        st.success("Voici votre sélection retouchée :")
        cols = st.columns(3)
        for idx, item in enumerate(top_photos):
            final = retoucher(item['img'])
            cols[idx].image(final, caption=f"Score: {item['score']}")
          
