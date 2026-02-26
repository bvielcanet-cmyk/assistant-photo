import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION IA SÉCURISÉE ---
IA_DISPONIBLE = False
try:
    import mediapipe as mp
    # Accès direct aux solutions pour éviter les erreurs d'import sur serveur
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    IA_DISPONIBLE = True
except Exception as e:
    st.warning(f"Note : L'IA faciale est indisponible sur ce serveur (Erreur: {e}). Le tri se basera uniquement sur la netteté.")

# --- FONCTIONS TECHNIQUES ---

def analyser_photo(image_np):
    """Calcule un score de qualité (0 à 100)."""
    score = 0
    # Conversion en niveaux de gris pour la netteté
    gris = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 1. TEST DE NETTETÉ (Laplacian) - 50 points
    var_laplacien = cv2.Laplacian(gris, cv2.CV_64F).var()
    score_nettete = min(var_laplacien / 1.5, 50) # On plafonne à 50
    score += score_nettete

    # 2. IA FACIALE - 50 points
    if IA_DISPONIBLE:
        results = face_mesh.process(image_np)
        if results.multi_face_landmarks:
            score += 30 # Visage détecté
            # Vérification sommaire de l'ouverture des yeux
            for face_landmarks in results.multi_face_landmarks:
                paupiere_sup = face_landmarks.landmark[159].y
                paupiere_inf = face_landmarks.landmark[145].y
                if abs(paupiere_inf - paupiere_sup) > 0.012: # Seuil oeil ouvert
                    score += 20
    else:
        # Si l'IA est HS, on double l'importance de la netteté pour compenser
        score += score_nettete 

    return int(score)

def retoucher_image(image_np):
    """Applique une retouche automatique 'Studio'."""
    # Amélioration du contraste (CLAHE)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_ret = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    
    # Légère augmentation de la netteté
    noyau = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_ret = cv2.filter2D(img_ret, -1, noyau)
    
    return img_ret

# --- INTERFACE UTILISATEUR (STREAMLIT) ---

st.set_page_config(page_title="IA Photo Studio Pro", page_icon="📸")

st.title("📸 Assistant de Tri & Retouche IA")
st.markdown("""
    **Gagnez du temps sur vos shootings :** 1. Uploadez vos photos (JPEG).
    2. L'IA sélectionne les **25 meilleures** (netteté, yeux ouverts).
    3. Retouche automatique instantanée.
""")

# Upload des fichiers
files = st.file_uploader("Glissez vos photos ici...", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"Analyser et Retoucher {len(files)} photos"):
        resultats = []
        barre = st.progress(0)
        
        # Traitement
        for i, f in enumerate(files):
            # Charger l'image
            img = Image.open(f).convert('RGB')
            img_np = np.array(img)
            
            # Analyser
            score = analyser_photo(img_np)
            resultats.append({"nom": f.name, "score": score, "img": img_np})
            
            barre.progress((i + 1) / len(files))

        # Trier et prendre les 25 meilleures (ou moins si pas assez de photos)
        top_25 = sorted(resultats, key=lambda x: x['score'], reverse=True)[:25]

        st.divider()
        st.subheader(f"✅ Sélection des {len(top_25)} meilleures pépites")

        # Affichage en grille
        cols = st.columns(3)
        for idx, item in enumerate(top_25):
            with cols[idx % 3]:
                # Appliquer la retouche
                img_final = retoucher_image(item['img'])
                st.image(img_final, caption=f"Score: {item['score']}/100")
                
                # Bouton de téléchargement individuel
                res_pil = Image.fromarray(img_final)
                st.download_button(label="Télécharger", data=f.getvalue(), file_name=f"RET_{item['nom']}", key=f"btn_{idx}")

st.sidebar.info("Projet en test : Idéal pour les photographes de portrait et studio.")
