# canvas_iterative_deconv_v3.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json, os
from numba import njit

# -------------------- Session-State Initialisierung --------------------
def init_session_state():
    defaults = {
        "vector_mode_active": False,
        "stain_samples": [],
        "current_stain_vector": None,
        "groups": [],
        "all_points": [],
        "C_cache": None,
        "last_M_hash": None,
        "history": [],
        "last_file": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

st.set_page_config(page_title="Iterative Kern-Zählung (OD + Deconv) — v3", layout="wide")

# -------------------- Optimierte Hilfsfunktionen --------------------
@njit
def od_from_patch(patch):
    patch_f = patch.astype(np.float32)
    return -np.log(np.clip((patch_f + 1e-6) / 255.0, 1e-8, 1.0))

@njit
def normalize_vector(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

@st.cache_data
def deconvolve_cached(img_rgb, M_flat):
    M = np.array(M_flat, dtype=np.float32).reshape(3,3)
    img = img_rgb.astype(np.float32)
    OD = -np.log(np.clip((img + 1e-6)/255.0,1e-8,1.0)).reshape(-1,3)
    pinv = np.linalg.pinv(M)
    C = (pinv @ OD.T).T
    return C.reshape(img_rgb.shape)

# Skala zeichnen bleibt unverändert

def draw_scale_bar(img_disp, scale, length_orig=200, bar_height=10, margin=20, color=(0,0,0)):
    h, w = img_disp.shape[:2]
    length_disp = int(round(length_orig * scale))
    x1, y1 = margin, h - margin - bar_height
    x2, y2 = x1 + length_disp, h - margin
    cv2.rectangle(img_disp, (x1,y1), (x2,y2), color, -1)
    cv2.putText(img_disp, f"{length_orig} px", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,1,cv2.LINE_AA)
    return img_disp

def is_near(p1, p2, r=6):
    return np.linalg.norm(np.array(p1)-np.array(p2)) < r

def dedup_new_points(candidates, existing, min_dist=6):
    out = []
    for c in candidates:
        if not any(is_near(c, e, min_dist) for e in existing):
            out.append(c)
    return out

def extract_patch(img, x, y, radius=5):
    y_min = max(0, y - radius)
    y_max = min(img.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(img.shape[1], x + radius + 1)
    return img[y_min:y_max, x_min:x_max]

def median_od_vector_from_patch(patch):
    OD = od_from_patch(patch)
    vec = np.median(OD.reshape(-1,3), axis=0)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm>1e-12 else vec

# -------------------- Session state vorbereiten --------------------
for k in ["groups", "all_points", "last_file", "disp_width", "C_cache", "last_M_hash", "history"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k in ["groups","all_points","history"] else 1200 if k=="disp_width" else None

# -------------------- Upload & Parameter --------------------
st.sidebar.markdown("<h5 style='color:darkred; font-size:22px;'>🧬 Iterative Kern-Zählung V.3</h5>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Wähle ein Bild (jpg/png/tif)", type=["jpg","png","tif","tiff"])
if not uploaded_file: st.info("Bitte zuerst ein Bild hochladen."); st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups, st.session_state.all_points = [], []
    st.session_state.C_cache, st.session_state.last_M_hash = None, None
    st.session_state.history = []
    st.session_state.last_file = uploaded_file.name

DISPLAY_WIDTH = st.sidebar.slider("Anzeige-Breite (px)", 300,1600, st.session_state.disp_width)
st.session_state.disp_width = DISPLAY_WIDTH

# Parameter laden (wie in v2) bleibt unverändert
PARAM_FILE = "params.json"
def load_parameter_sets():
    if os.path.exists(PARAM_FILE):
        with open(PARAM_FILE,"r") as f: return json.load(f)
    else: 
        default_sets = {"default": {"kalibrier_radius":10,"min_konturflaeche":1000,"dedup_distanz":50,"kernel_size_open":3,"kernel_size_close":3,"marker_radius":5,"hema_vec":"0.65,0.70,0.29","aec_vec":"0.27,0.57,0.78"}}
        with open(PARAM_FILE,"w") as f: json.dump(default_sets,f)
        return default_sets

parameter_sets = load_parameter_sets()
choice = st.sidebar.radio("Wähle Parameterset", list(parameter_sets.keys()), index=list(parameter_sets.keys()).index("default"))
params = parameter_sets[choice]

calib_radius = params["kalibrier_radius"]
min_area_orig = params["min_konturflaeche"]
dedup_dist_orig = params["dedup_distanz"]
kernel_size_open = params["kernel_size_open"]
kernel_size_close = params["kernel_size_close"]
circle_radius = params["marker_radius"]
hema_vec0 = np.array([float(x.strip()) for x in params["hema_vec"].split(",")],dtype=float)
aec_vec0  = np.array([float(x.strip()) for x in params["aec_vec"].split(",")],dtype=float)

# -------------------- Bildvorbereitung --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / float(W_orig)
H_disp = int(round(H_orig*scale))
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH,H_disp), interpolation=cv2.INTER_AREA)

# Display Canvas vorbereiten
display_canvas = draw_scale_bar(image_disp.copy(), scale, length_orig=200, bar_height=3, color=(0,0,0))

# Punkte der Gruppen zeichnen
PRESET_COLORS = [(220,20,60),(0,128,0),(30,144,255),(255,165,0),(148,0,211),(0,255,255)]
for i,g in enumerate(st.session_state.groups):
    col = tuple(int(x) for x in g.get("color", PRESET_COLORS[i % len(PRESET_COLORS)]))
    for (x_orig,y_orig) in g["points"]:
        cv2.circle(display_canvas, (int(round(x_orig*scale)), int(round(y_orig*scale))), circle_radius, col, -1)
    if g["points"]:
        cv2.putText(display_canvas, f"G{i+1}:{len(g['points'])}", (int(round(g['points'][0][0]*scale))+6, int(round(g['points'][0][1]*scale))-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1,cv2.LINE_AA)

coords = streamlit_image_coordinates(Image.fromarray(display_canvas), key=f"clickable_image_v3_{st.session_state.last_file}", width=DISPLAY_WIDTH)

# -------------------- Click Handling --------------------
if coords:
    x_disp, y_disp = int(coords['x']), int(coords['y'])
    x_orig, y_orig = int(round(x_disp/scale)), int(round(y_disp/scale))

    patch = extract_patch(image_orig, x_orig, y_orig, calib_radius)
    vec = median_od_vector_from_patch(patch)

    if vec is not None:
        M = np.column_stack([normalize_vector(vec), hema_vec0, np.cross(vec, hema_vec0)])
        M_hash = tuple(np.round(M.flatten(),6).tolist())
        if st.session_state.C_cache is None or st.session_state.last_M_hash != M_hash:
            st.session_state.C_cache = deconvolve_cached(image_orig, M.flatten())
            st.session_state.last_M_hash = M_hash
        C_full = st.session_state.C_cache
        channel_full = C_full[:,:,0]

        # Downsampling für schnelle Konturerkennung
        ds_factor = max(1, int(channel_full.shape[1]/800))
        if ds_factor > 1:
            channel_ds = cv2.resize(channel_full, (channel_full.shape[1]//ds_factor, channel_full.shape[0]//ds_factor))
        else:
            channel_ds = channel_full
        _, mask = cv2.threshold((channel_ds*255).astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers_orig = [(int(cnt[:,0,0]*ds_factor), int(cnt[:,0,1]*ds_factor)) for cnt in contours]

        new_centers = dedup_new_points(centers_orig, st.session_state.all_points, dedup_dist_orig)
        if not any(is_near(p,(x_orig,y_orig),dedup_dist_orig) for p in st.session_state.all_points):
            new_centers.append((x_orig,y_orig))
        if new_centers:
            color = PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]
            st.session_state.groups.append({"vec": vec.tolist(), "points": new_centers, "color": color})
            st.session_state.all_points.extend(new_centers)
            st.success(f"Gruppe hinzugefügt — neue Kerne: {len(new_centers)}")
