import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION IA SÉCURISÉE ---
# On utilise l'importation standard qui fonctionne parfaitement sur Python 3.11
IA_DISPONIBLE = False
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    
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

# --- FONCTIONS DE TRAITEMENT ---

def analyser_photo(image_np):
    """Note la photo sur 100 points (Netteté + IA Yeux)."""
    score = 0
    # Conversion gris pour OpenCV
    gris = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 1. TEST DE NETTETÉ (50 pts)
    # Plus la variance est haute, plus les bords sont nets
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    score_nettete = min((lap_var / 100) * 50, 50) 
    score += score_nettete

    # 2. IA FACIALE (50 pts)
    if IA_DISPONIBLE:
        results = face_mesh.process(image_np)
        if results.multi_face_landmarks:
            score += 25 # Visage détecté
            
            # Analyse de l'ouverture des yeux
            for face_landmarks in results.multi_face_landmarks:
                # Points de repère pour les paupières (oeil gauche)
                p_sup = face_landmarks.landmark[159].y
                p_inf = face_landmarks.landmark[145].y
                ouverture = abs(p_inf - p_sup)
                
                # Si l'oeil est suffisamment ouvert (seuil empirique)
                if ouverture > 0.012:
                    score += 25
        else:
            score -= 10 # Pénalité si aucun visage détecté
    else:
        # Compensation si l'IA est hors-ligne
        score += score_nettete

    return int(max(0, min(score, 100)))

def retoucher_image(image_np):
    """Retouche auto : Contraste, Netteté et Vibrance."""
    # A. Amélioration du contraste (CLAHE)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    # B. Filtre de netteté (Unsharp Mask)
    noyau = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, noyau)
    
    return img

# --- INTERFACE UTILISATEUR (STREAMLIT) ---

st.set_page_config(page_title="IA Photo Studio Pro", page_icon="📸", layout="wide")

# Barre latérale (Sidebar)
with st.sidebar:
    st.title("⚙️ Paramètres")
    if IA_DISPONIBLE:
        st.success("✅ IA Faciale : Active")
    else:
        st.error("⚠️ IA Faciale : Hors-ligne")
        st.info(f"Détail : {IA_ERREUR if 'IA_ERREUR' in locals() else 'Non installé'}")
    
    st.divider()
    seuil_selection = st.slider("Nombre de photos à garder", 1, 50, 25)

st.title("📸 Assistant de Tri & Retouche IA")
st.write("Identifiez instantanément les meilleures photos de votre shooting.")

# Upload
files = st.file_uploader("Glissez vos JPEG ici", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"Lancer l'analyse de {len(files)} photos"):
        resultats = []
        barre = st.progress(0)
        
        for i, f in enumerate(files):
            # Charger l'image
            img_pil = Image.open(f).convert('RGB')
            img_np = np.array(img_pil)
            
            # Analyser
            note = analyser_photo(img_np)
            
            # Stocker
            resultats.append({"nom": f.name, "score": note, "img": img_np})
            barre.progress((i + 1) / len(files))

        # Sélectionner le TOP
        top_photos = sorted(resultats, key=lambda x: x['score'], reverse=True)[:seuil_selection]

        st.divider()
        st.subheader(f"✨ Votre sélection ({len(top_photos)} photos)")

        # Grille d'affichage
        cols = st.columns(3)
        for idx, item in enumerate(top_photos):
            with cols[idx % 3]:
                # Appliquer la retouche au moment de l'affichage
                img_final = retoucher_image(item['img'])
                st.image(img_final, caption=f"Score: {item['score']}/100 - {item['nom']}")
                
                # Préparation du bouton de téléchargement
                # (Note : Pour un vrai service, on créerait un ZIP)
                result_pil = Image.fromarray(img_final)
                st.download_button(
                    label="Télécharger",
                    data=f.getvalue(), # On renvoie l'original pour l'essai
                    file_name=f"TOP_{item['nom']}",
                    key=f"dl_{idx}"
                )

st.sidebar.markdown("---")
st.sidebar.caption("Développé pour les photographes de studio.")
