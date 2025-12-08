# canvas_iterative_deconv_v2_A2.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json, os

# -------------------- Session-State Initialisierung --------------------
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

st.set_page_config(page_title="Iterative Kern-Zählung (OD + Deconv) — v2.A2", layout="wide")

# -------------------- Hilfsfunktionen --------------------
def draw_scale_bar(img_disp, scale, length_orig=200, bar_height=10, margin=20, color=(0,0,0)):
    h, w = img_disp.shape[:2]
    length_disp = int(round(length_orig * scale))
    x1, y1 = margin, h - margin - bar_height
    x2, y2 = x1 + length_disp, h - margin
    cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, -1)
    cv2.putText(img_disp, f"{length_orig} px", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img_disp

def is_near(p1, p2, r=6):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_new_points(candidates, existing, min_dist=6):
    if not existing:
        return candidates
    existing_arr = np.array(existing)
    out = []
    for c in candidates:
        dists = np.linalg.norm(existing_arr - c, axis=1)
        if np.all(dists >= min_dist):
            out.append(c)
    return out

def extract_patch(img, x, y, radius=5):
    y_min, y_max = max(0, y - radius), min(img.shape[0], y + radius + 1)
    x_min, x_max = max(0, x - radius), min(img.shape[1], x + radius + 1)
    return img[y_min:y_max, x_min:x_max]

def median_od_vector_from_patch(patch, eps=1e-6):
    if patch is None or patch.size == 0:
        return None
    patch_f = patch.astype(np.float32)
    OD = -np.log(np.clip((patch_f + eps) / 255.0, 1e-8, 1.0))
    vec = np.median(OD.reshape(-1, 3), axis=0)
    norm = np.linalg.norm(vec)
    if norm < 1e-8 or np.any(np.isnan(vec)):
        return None
    return (vec / norm).astype(np.float32)

def normalize_vector(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return (v / n).astype(float) if n > 1e-12 else v

def make_stain_matrix(target_vec, hema_vec, bg_vec=None):
    t, h = normalize_vector(target_vec), normalize_vector(hema_vec)
    if bg_vec is None:
        bg = np.cross(t, h)
        if np.linalg.norm(bg) < 1e-6:
            if abs(t[0]) > 0.1 or abs(t[1]) > 0.1:
                bg = np.array([t[1], -t[0], 0.0], dtype=float)
            else:
                bg = np.array([0.0, t[2], -t[1]], dtype=float)
        bg = normalize_vector(bg)
    else:
        bg = normalize_vector(bg_vec)
    M = np.column_stack([t, h, bg]).astype(np.float32)
    M += np.eye(3, dtype=np.float32) * 1e-8
    return M

def deconvolve(img_rgb, M):
    img_f = img_rgb.astype(np.float32)
    OD = -np.log(np.clip((img_f + 1e-6) / 255.0, 1e-8, 1.0)).reshape(-1, 3)
    try:
        pinv = np.linalg.pinv(M)
        C = (pinv @ OD.T).T
    except Exception:
        return None
    return C.reshape(img_rgb.shape)

def detect_centers_from_channel_v2(channel, threshold=0.2, min_area=30, debug=False):
    arr = np.maximum(np.array(channel, dtype=np.float32), 0.0)
    vmin, vmax = np.percentile(arr, [2, 99.5])
    if vmax - vmin < 1e-5:
        return [], np.zeros_like(arr, dtype=np.uint8)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    try:
        mask = cv2.adaptiveThreshold(u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 35, -2)
    except Exception:
        _, mask = cv2.threshold(u8, int(threshold*255), 255, cv2.THRESH_BINARY)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_open, kernel_size_open))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_close, kernel_size_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
    if debug:
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for (cx, cy) in centers: cv2.circle(dbg, (cx, cy), 5, (0,0,255), -1)
        return centers, dbg
    return centers, mask

# -------------------- Upload & Sidebar --------------------
st.sidebar.markdown("<h5 style='color:darkred; font-size:22px;'>🧬 Iterative Kern-Zählung V.2.A2</h5>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Wähle ein Bild (jpg/png/tif)", type=["jpg","png","tif","tiff"])
if not uploaded_file: st.stop()
if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups = []
    st.session_state.all_points = []
    st.session_state.C_cache = None
    st.session_state.last_M_hash = None
    st.session_state.history = []
    st.session_state.last_file = uploaded_file.name

DISPLAY_WIDTH = st.sidebar.slider("Anzeige-Breite (px)", 300, 1600, 1200)
PARAM_FILE = "params.json"
default_sets = {
    "default": {"kalibrier_radius":10,"min_konturflaeche":1000,"dedup_distanz":50,
                "kernel_size_open":3,"kernel_size_close":3,"marker_radius":5,
                "hema_vec":"0.65,0.70,0.29","aec_vec":"0.27,0.57,0.78"}}
if os.path.exists(PARAM_FILE):
    with open(PARAM_FILE,"r") as f: parameter_sets = json.load(f)
else:
    parameter_sets = default_sets
    with open(PARAM_FILE,"w") as f: json.dump(parameter_sets,f)

choice = st.sidebar.radio("Wähle Parameterset", list(parameter_sets.keys()), index=list(parameter_sets.keys()).index("default"))
params = parameter_sets[choice]

calib_radius, min_area_orig, dedup_dist_orig = params["kalibrier_radius"], params["min_konturflaeche"], params["dedup_distanz"]
kernel_size_open, kernel_size_close, circle_radius = params["kernel_size_open"], params["kernel_size_close"], params["marker_radius"]
hema_vec0 = np.array([float(x.strip()) for x in params["hema_vec"].split(",")],dtype=float)
aec_vec0  = np.array([float(x.strip()) for x in params["aec_vec"].split(",")],dtype=float)

# -------------------- Bild vorbereiten --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / float(W_orig)
H_disp = int(round(H_orig * scale))
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH,H_disp), interpolation=cv2.INTER_AREA)
display_canvas = image_disp.copy()
display_canvas = draw_scale_bar(display_canvas, scale, length_orig=200, bar_height=3, color=(0,0,0))

# -------------------- Gruppen Rendering --------------------
PRESET_COLORS=[(220,20,60),(0,128,0),(30,144,255),(255,165,0),(148,0,211),(0,255,255)]
for i,g in enumerate(st.session_state.groups):
    col=tuple(int(x) for x in g.get("color",PRESET_COLORS[i%len(PRESET_COLORS)]))
    for (x_orig,y_orig) in g["points"]:
        x_disp=int(round(x_orig*scale)); y_disp=int(round(y_orig*scale))
        cv2.circle(display_canvas,(x_disp,y_disp),circle_radius,col,-1)
    if g["points"]:
        px_disp=int(round(g["points"][0][0]*scale)); py_disp=int(round(g["points"][0][1]*scale))
        cv2.putText(display_canvas,f"G{i+1}:{len(g['points'])}",(px_disp+6,py_disp-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1,cv2.LINE_AA)

coords = streamlit_image_coordinates(Image.fromarray(display_canvas), key=f"clickable_image_v2_{st.session_state.last_file}", width=DISPLAY_WIDTH)

# -------------------- Click Handling & Aktionen --------------------
mode = st.sidebar.radio("Aktion", ["Kalibriere und zähle Gruppe (Klick)","Punkt löschen","Undo letzte Aktion"])
if st.sidebar.button("Reset (Alle Gruppen)"):
    st.session_state.history.append(("reset", {"groups":st.session_state.groups.copy(),"all_points":st.session_state.all_points.copy()}))
    st.session_state.groups=[]; st.session_state.all_points=[]; st.session_state.C_cache=None; st.success("Zurückgesetzt.")

if coords:
    x_disp, y_disp=int(coords["x"]), int(coords["y"])
    x_orig, y_orig=int(round(x_disp/scale)), int(round(y_disp/scale))
    if mode=="Punkt löschen":
        removed=[]
        new_all=[]
        for p in st.session_state.all_points:
            if is_near(p,(x_orig,y_orig),dedup_dist_orig): removed.append(p)
            else: new_all.append(p)
        if removed:
            st.session_state.history.append(("delete_points", {"removed": removed}))
            st.session_state.all_points=new_all
            for g in st.session_state.groups:
                g["points"]=[p for p in g["points"] if not is_near(p,(x_orig,y_orig),dedup_dist_orig)]
            st.success(f"{len(removed)} Punkt(e) gelöscht.")
        else: st.info("Kein Punkt in der Nähe gefunden.")
    elif mode=="Undo letzte Aktion":
        if st.session_state.history:
            action,payload=st.session_state.history.pop()
            if action=="add_group":
                idx=payload["group_idx"]
                if 0<=idx<len(st.session_state.groups):
                    grp=st.session_state.groups.pop(idx)
                    for pt in grp["points"]:
                        st.session_state.all_points=[p for p in st.session_state.all_points if p!=pt]
                st.success("Letzte Gruppen-Aktion rückgängig gemacht.")
            elif action=="delete_points":
                removed=payload["removed"]
                st.session_state.all_points.extend(removed)
                st.session_state.groups.append({"vec":None,"points":removed,"color":PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]})
                st.success("Gelöschte Punkte wiederhergestellt (als neue Gruppe).")
            elif action=="reset":
                st.session_state.groups=payload["groups"]; st.session_state.all_points=payload["all_points"]
                st.success("Reset rückgängig gemacht.")
            else: st.warning("Undo: unbekannte Aktion.")
        else: st.info("Keine Aktion zum Rückgängig machen.")
    else:
        patch=extract_patch(image_orig,x_orig,y_orig,calib_radius)
        vec=median_od_vector_from_patch(patch)
        if vec is None: st.warning("Patch unbrauchbar (zu homogen oder außerhalb). Bitte anders klicken.")
        else:
            M=make_stain_matrix(vec,hema_vec0)
            M_hash=tuple(np.round(M.flatten(),6).tolist())
            recompute=False
            if st.session_state.C_cache is None or st.session_state.last_M_hash!=M_hash:
                recompute=True
            if recompute:
                C_full=deconvolve(image_orig,M)
                if C_full is None: st.error("Deconvolution fehlgeschlagen."); st.stop()
                st.session_state.C_cache=C_full; st.session_state.last_M_hash=M_hash
            else: C_full=st.session_state.C_cache
            channel_full=C_full[:,:,0]
            centers_orig,_=detect_centers_from_channel_v2(channel_full,threshold=0.2,min_area=min_area_orig)
            new_centers=dedup_new_points(centers_orig,st.session_state.all_points,min_dist=dedup_dist_orig)
            if not any(is_near(p,(x_orig,y_orig),dedup_dist_orig) for p in st.session_state.all_points):
                new_centers.append((x_orig,y_orig))
            if new_centers:
                color=PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]
                group={"vec":vec.tolist(),"points":new_centers,"color":color}
                st.session_state.history.append(("add_group",{"group_idx":len(st.session_state.groups)}))
                st.session_state.groups.append(group)
                st.session_state.all_points.extend(new_centers)
                st.success(f"Gruppe hinzugefügt — neue Kerne: {len(new_centers)} (inkl. Klickpunkt)")
            else: st.info("Keine neuen Kerne (alle bereits gezählt oder keine Detektion).")

# -------------------- Ergebnis-Anzeige & CSV --------------------
st.markdown("## Ergebnisse")
colA,colB=st.columns([2,1])
with colA: st.image(display_canvas,caption="Gezählte Kerne (Gruppenfarben)",use_column_width=True)
with colB:
    st.markdown("### Zusammenfassung")
    st.write(f"🔹 Gruppen gesamt: {len(st.session_state.groups)}")
    for i,g in enumerate(st.session_state.groups):
        st.write(f"• Gruppe {i+1}: {len(g['points'])} neue Kerne")
    st.markdown(f"**Gesamt (unique Kerne): {len(st.session_state.all_points)}**")

if st.session_state.all_points:
    rows=[]
    for i,g in enumerate(st.session_state.groups):
        for (x_orig,y_orig) in g["points"]:
            rows.append({"Group":i+1,"X_display":int(round(x_orig*scale)),"Y_display":int(round(y_orig*scale)),
                         "X_original":int(x_orig),"Y_original":int(y_orig)})
    df=pd.DataFrame(rows)
    st.download_button("📥 CSV exportieren (Gruppen inkl. Original)",df.to_csv(index=False).encode("utf-8"),
                       file_name="kern_gruppen_v2.csv",mime="text/csv")
    df_unique=pd.DataFrame(st.session_state.all_points,columns=["X_original","Y_original"])
    df_unique["X_display"]=(df_unique["X_original"]*scale).round().astype(int)
    df_unique["Y_display"]=(df_unique["Y_original"]*scale).round().astype(int)
    st.download_button("📥 CSV exportieren (unique Gesamt)",df_unique.to_csv(index=False).encode("utf-8"),
                       file_name="kern_unique_v2.csv",mime="text/csv")
