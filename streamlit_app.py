import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION IA SÉCURISÉE (MÉTHODE PRO) ---
IA_DISPONIBLE = False
try:
    # On force l'importation profonde pour éviter l'erreur 'no attribute solutions'
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    IA_DISPONIBLE = True
except Exception as e:
    # Tentative d'importation alternative si la première échoue
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
        IA_DISPONIBLE = True
    except:
        IA_DISPONIBLE = False
        IA_ERREUR = str(e)

# --- FONCTIONS DE TRAITEMENT ---

def analyser_photo(image_np):
    """Note la photo sur 100 points."""
    score = 0
    gris = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 1. NETTETÉ (50 points)
    # Calcul de la variance du Laplacien
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    # On considère qu'au dessus de 150 c'est parfait pour du studio
    score_nettete = min((lap_var / 150) * 50, 50)
    score += score_nettete

    # 2. IA FACIALE (50 points)
    if IA_DISPONIBLE:
        # Mediapipe travaille en RGB
        results = face_mesh.process(image_np)
        if results.multi_face_landmarks:
            score += 25 # Visage détecté
            
            # Analyse des yeux (Paupières)
            for face_landmarks in results.multi_face_landmarks:
                # Indices Mediapipe pour les paupières oeil gauche
                p_sup = face_landmarks.landmark[159].y
                p_inf = face_landmarks.landmark[145].y
                ouverture = abs(p_inf - p_sup)
                
                if ouverture > 0.012: # Seuil oeil ouvert
                    score += 25
        else:
            score -= 20 # Pénalité si aucun visage (pour du portrait)
    else:
        # Compensation si l'IA est absente
        score += score_nettete 

    return int(max(0, min(score, 100)))

def retoucher_image(image_np):
    """Retouche auto : Contraste, Netteté et Vibrance."""
    # A. Contraste adaptatif (CLAHE)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    # B. Filtre de netteté léger
    noyau = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, noyau)
    
    return img

# --- INTERFACE STREAMLIT ---

st.set_page_config(page_title="IA Photo Studio Pro", page_icon="📸", layout="wide")

# Affichage de l'état de l'IA dans la barre latérale
with st.sidebar:
    st.title("Paramètres")
    if IA_DISPONIBLE:
        st.success("✅ IA Faciale : Active")
    else:
        st.error(f"⚠️ IA Faciale : Hors ligne")
        st.info(f"Détail : {IA_ERREUR if 'IA_ERREUR' in locals() else 'Inconnu'}")
    
    st.divider()
    seuil_tri = st.slider("Seuil de sélection (25 meilleures)", 5, 50, 25)

st.title("📸 Assistant Studio : Tri & Retouche Automatique")
st.write("Uploadez vos JPEG, laissez l'IA travailler, téléchargez le meilleur.")

files = st.file_uploader("Shooting à traiter", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"Lancer le traitement de {len(files)} images"):
        resultats = []
        progress_bar = st.progress(0)
        
        for i, f in enumerate(files):
            # Lecture
            img_pil = Image.open(f).convert('RGB')
            img_np = np.array(img_pil)
            
            # Analyse
            note = analyser_photo(img_np)
            
            # Stockage
            resultats.append({"nom": f.name, "score": note, "img": img_np, "bytes": f.getvalue()})
            progress_bar.progress((i + 1) / len(files))

        # Tri par score
        top_selection = sorted(resultats, key=lambda x: x['score'], reverse=True)[:seuil_tri]

        st.divider()
        st.header(f"✨ Sélection : Les {len(top_selection)} meilleures images")

        # Grille d'affichage
        cols = st.columns(3)
        for idx, item in enumerate(top_selection):
            with cols[idx % 3]:
                # Retouche
                final_img = retoucher_image(item['img'])
                st.image(final_img, caption=f"Score : {item['score']}/100")
                
                # Téléchargement
                st.download_button(
                    label=f"Télécharger {item['nom']}",
                    data=item['bytes'], # Pour l'essai on rend le fichier original
                    file_name=f"TOP_{item['nom']}",
                    key=f"dl_{idx}"
                )
