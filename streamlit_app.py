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
    """Note la photo (0-100) : Netteté parfaite + IA Yeux."""
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
            score += 25 
            for face_landmarks in results.multi_face_landmarks:
                p_sup = face_landmarks.landmark[159].y
                p_inf = face_landmarks.landmark[145].y
                if abs(p_inf - p_sup) > 0.012: # Oeil ouvert
                    score += 25
        else:
            score -= 10
    else:
        score += score_nettete

    return int(max(0, min(score, 100)))

def retoucher_instagram_pro(image_np):
    """Retouche IA : Peau douce (Bilateral), Éclat (LAB) et Vibrance."""
    
    # 1. ÉCLAIRAGE STUDIO (Espace LAB pour ne pas dénaturer les couleurs)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_light = cv2.merge((l,a,b))
    img_rgb = cv2.cvtColor(img_light, cv2.COLOR_LAB2RGB)

    # 2. PEAU DOUCE / LISSAGE (Flou Bilatéral)
    # Lisse les imperfections mais garde les contours nets (yeux, lèvres)
    img_smooth = cv2.bilateralFilter(img_rgb, d=7, sigmaColor=50, sigmaSpace=50)
    # On mixe l'original et le lissé pour garder un grain de peau naturel (60% lissé)
    img_final = cv2.addWeighted(img_rgb, 0.4, img_smooth, 0.6, 0)

    # 3. VIBRANCE INSTAGRAM (Saturation sélective)
    hsv = cv2.cvtColor(img_final, cv2.COLOR_RGB2HSV).astype("float32")
    hsv[:, :, 1] *= 1.12  # Boost saturation de 12%
    hsv[:, :, 2] *= 1.05  # Boost luminosité de 5%
    hsv = np.clip(hsv, 0, 255).astype("uint8")
    img_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 4. NETTETÉ FINALE (High-pass filter léger)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_final = cv2.filter2D(img_final, -1, kernel)

    return img_final

# --- INTERFACE ---
st.set_page_config(page_title="Studio IA Pro", layout="wide")

st.title("✨ Assistant Studio : Sélection & Retouche Instagram Pro")
st.markdown("*Lissage de peau, éclairage studio et tri intelligent par IA.*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    if IA_DISPONIBLE:
        st.success("✅ IA Faciale : Active")
    else:
        st.error("⚠️ IA Faciale : Hors-ligne")
    
    nb_top = st.slider("Nombre de pépites à extraire", 1, 50, 25)
    st.divider()
    st.caption("Développé pour photographes de studio et créateurs.")

# Zone d'Upload
files = st.file_uploader("Glissez vos photos JPEG ici", type=['jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if st.button(f"🚀 Générer mon Pack Instagram ({len(files)} photos)"):
        resultats = []
        barre = st.progress(0)
        status = st.empty()
        
        # Phase Analyse
        for i, f in enumerate(files):
            status.text(f"Analyse IA de {f.name}...")
            img_pil = Image.open(f).convert('RGB')
            img_np = np.array(img_pil)
            
            score = analyser_photo(img_np)
            resultats.append({"nom": f.name, "score": score, "bytes": f.getvalue()})
            barre.progress((i + 1) / len(files))

        status.text("Tri et Retouche de la sélection...")
        
        # Sélection TOP
        top_selection = sorted(resultats, key=lambda x: x['score'], reverse=True)[:nb_top]

        # Phase ZIP & Retouche Pro
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as xzip:
            for item in top_selection:
                # Charger et appliquer le filtre Instagram Pro uniquement aux gagnantes
                raw_img = Image.open(io.BytesIO(item['bytes'])).convert('RGB')
                img_insta = retoucher_instagram_pro(np.array(raw_img))
                
                # Sauvegarde JPEG Haute Qualité
                out_pil = Image.fromarray(img_insta)
                out_io = io.BytesIO()
                out_pil.save(out_io, format='JPEG', quality=95)
                xzip.writestr(f"INSTA_{item['nom']}", out_io.getvalue())

        st.divider()
        
        # BOUTON TÉLÉCHARGEMENT ZIP
        st.download_button(
            label=f"📥 TÉLÉCHARGER LE PACK RETOUCHÉ ({len(top_selection)} PHOTOS)",
            data=zip_buf.getvalue(),
            file_name="pack_instagram_pro.zip",
            mime="application/zip"
        )

        # Aperçu
        st.subheader("Aperçu de vos photos Instagrammables")
        cols = st.columns(3)
        for idx, item in enumerate(top_selection):
            with cols[idx % 3]:
                # On ré-applique la retouche pour l'affichage (version plus rapide)
                prev_img = Image.open(io.BytesIO(item['bytes'])).convert('RGB')
                res_img = retoucher_instagram_pro(np.array(prev_img))
                st.image(res_img, caption=f"Score: {item['score']}/100")

        status.text("✅ Shooting traité avec succès !")
