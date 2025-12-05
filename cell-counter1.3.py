"""
iterative_deconv_refactor_v3.py
Refaktorisierte, modulare und optimierte Version der Iterativen Kern-Zählung (Streamlit)
- modular: Stain/OD utilities, Detection, UI
- Performance: optional Numba acceleration, caching mit streamlit.cache_data
- Robustheit: Logging, stabilere Pseudoinverse, klarere Session-State-API
- UI: übersichtlichere Sidebar, Legende, Debug-Optionen

Hinweis: Numba ist optional — das Skript läuft ohne Numba. Wenn installiert, werden kritische Pfade beschleunigt.

Benutzung: `streamlit run iterative_deconv_refactor_v3.py`

"""

from __future__ import annotations
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import json
import os
import logging
from typing import Tuple, List, Optional, Dict

# Optional acceleration
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# -------------------- Logging --------------------
LOG_FMT = "%(asctime)s %(levelname)s:%(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("iter_deconv_v3")

# -------------------- Configuration / Defaults --------------------
PARAM_FILE = "params.json"
DEFAULT_PARAMS = {
    "kalibrier_radius": 10,
    "min_konturflaeche": 1000,
    "dedup_distanz": 50,
    "kernel_size_open": 3,
    "kernel_size_close": 3,
    "marker_radius": 5,
    "hema_vec": "0.65,0.70,0.29",
    "aec_vec": "0.27,0.57,0.78"
}

PRESET_COLORS = [
    (220, 20, 60),    # crimson
    (0, 128, 0),      # green
    (30, 144, 255),   # dodger
    (255, 165, 0),    # orange
    (148, 0, 211),    # purple
    (0, 255, 255),    # cyan
]

# -------------------- Utilities: Vector, OD, Matrix --------------------

def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return (v / n).astype(float) if n > 1e-12 else v.astype(float)


def make_stain_matrix(target_vec: np.ndarray, hema_vec: np.ndarray, bg_vec: Optional[np.ndarray] = None) -> np.ndarray:
    """Build a stable 3x3 stain matrix (columns = target, hema, background).
    Adds a tiny regularization to avoid singular matrices.
    """
    t = normalize_vector(target_vec)
    h = normalize_vector(hema_vec)
    if bg_vec is None:
        bg = np.cross(t, h)
        if np.linalg.norm(bg) < 1e-6:
            # fallback orthogonal guess
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


@st.cache_data(show_spinner=False)
def compute_deconvolution_cache(img_rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Compute concentration matrix C for the whole original image. Cached by Streamlit.
    Returns image-shaped array (H,W,3) of concentrations.
    """
    logger.info("Computing deconvolution (this will be cached)")
    img = img_rgb.astype(np.float32)
    # safe OD conversion
    OD = -np.log(np.clip((img + 1e-6) / 255.0, 1e-8, 1.0))  # H x W x 3
    H, W, _ = img.shape
    OD_flat = OD.reshape(-1, 3).T  # 3 x N
    # pinv with slight regularization
    pinv = np.linalg.pinv(M, rcond=1e-3)
    C_flat = (pinv @ OD_flat).T  # N x 3
    C = C_flat.reshape(H, W, 3)
    return C.astype(np.float32)


# Optional numba-accelerated OD conversion (elementwise) for large workloads
if NUMBA_AVAILABLE:
    @njit
    def _od_from_uint8_nb(img_arr):
        # expects uint8 H,W,3 -> float32 H,W,3
        H, W = img_arr.shape[0], img_arr.shape[1]
        out = np.empty((H, W, 3), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                for k in range(3):
                    v = (img_arr[i, j, k] + 1e-6) / 255.0
                    if v <= 1e-8:
                        v = 1e-8
                    out[i, j, k] = -np.log(v)
        return out
else:
    _od_from_uint8_nb = None

# -------------------- Detection utilities --------------------

def is_near(p1: Tuple[int, int], p2: Tuple[int, int], r: float = 6.0) -> bool:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1])) < float(r)


def dedup_new_points(candidates: List[Tuple[int, int]], existing: List[Tuple[int, int]], min_dist: float = 6.0) -> List[Tuple[int, int]]:
    out = []
    for c in candidates:
        if not any(is_near(c, e, min_dist) for e in existing):
            out.append(c)
    return out


def extract_patch(img: np.ndarray, x: int, y: int, radius: int = 5) -> np.ndarray:
    y_min = max(0, y - radius)
    y_max = min(img.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(img.shape[1], x + radius + 1)
    return img[y_min:y_max, x_min:x_max]


def median_od_vector_from_patch(patch: np.ndarray, eps: float = 1e-6) -> Optional[np.ndarray]:
    if patch is None or patch.size == 0:
        return None
    if NUMBA_AVAILABLE and _od_from_uint8_nb is not None and patch.dtype == np.uint8 and patch.size > 500:
        od = _od_from_uint8_nb(patch)
        vec = np.median(od.reshape(-1, 3), axis=0)
    else:
        patch = patch.astype(np.float32)
        OD = -np.log(np.clip((patch + eps) / 255.0, 1e-8, 1.0))
        vec = np.median(OD.reshape(-1, 3), axis=0)
    norm = np.linalg.norm(vec)
    if norm <= 1e-12 or np.any(np.isnan(vec)):
        return None
    return (vec / norm).astype(np.float32)


def detect_centers_from_channel(channel: np.ndarray, threshold: float = 0.2, min_area: int = 30,
                                kernel_size_open: int = 3, kernel_size_close: int = 3, debug: bool = False) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Detect centers on a single concentration channel.
    Returns list of (x,y) in original coordinates and mask image (uint8).
    """
    arr = np.array(channel, dtype=np.float32)
    arr = np.maximum(arr, 0.0)

    vmin, vmax = np.percentile(arr, [2, 99.5])
    if vmax - vmin < 1e-5:
        return [], np.zeros_like(arr, dtype=np.uint8)

    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)

    try:
        mask = cv2.adaptiveThreshold(u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -2)
    except Exception:
        _, mask = cv2.threshold(u8, int(threshold * 255), 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_open, kernel_size_open))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_close, kernel_size_close))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= max(1, min_area):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

    if debug:
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for (cx, cy) in centers:
            cv2.circle(dbg, (cx, cy), 5, (0, 0, 255), -1)
        return centers, dbg

    return centers, mask

# -------------------- Streamlit UI helpers --------------------

def draw_scale_bar(img_disp: np.ndarray, scale: float, length_orig: int = 200, bar_height: int = 10, margin: int = 20, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w = img_disp.shape[:2]
    length_disp = int(round(length_orig * scale))
    x1 = margin
    y1 = h - margin - bar_height
    x2 = min(w - margin, x1 + length_disp)
    y2 = h - margin
    cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, -1)
    cv2.putText(img_disp, f"{length_orig} px", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img_disp

# -------------------- Session-state utilities --------------------

def ensure_session_state():
    defaults = {
        "groups": [],
        "all_points": [],
        "last_file": None,
        "disp_width": 1200,
        "C_cache": None,
        "last_M_hash": None,
        "history": [],
        "vector_mode_active": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -------------------- Main Streamlit App --------------------

def main():
    st.set_page_config(page_title="Iterative Kern-Zählung (OD + Deconv) — v3", layout="wide")
    ensure_session_state()

    st.sidebar.markdown("<h4 style='color:darkred;'>🧬 Iterative Kern-Zählung — v3</h4>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1])

    # Upload
    with col_right:
        uploaded_file = st.file_uploader("Bild hochladen (jpg/png/tif)", type=["jpg", "png", "tif", "tiff"])
        st.markdown("---")

    if not uploaded_file:
        st.info("Bitte zuerst ein Bild hochladen.")
        st.stop()

    # Reset on new file
    if uploaded_file.name != st.session_state.last_file:
        st.session_state.groups = []
        st.session_state.all_points = []
        st.session_state.C_cache = None
        st.session_state.last_M_hash = None
        st.session_state.history = []
        st.session_state.last_file = uploaded_file.name

    # Load image
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]

    # Parameters
    param_sets = load_or_create_params()
    choice = st.sidebar.selectbox("Wähle Parameterset", list(param_sets.keys()), index=0)
    params = param_sets[choice]

    with st.sidebar.expander("Feintuning"):
        calib_radius = st.slider("Kalibrier-Radius", 1, 30, params.get("kalibrier_radius", 10))
        min_area_orig = st.number_input("Minimale Konturfläche", 1, 100000, params.get("min_konturflaeche", 1000))
        dedup_dist_orig = st.number_input("Dedup-Distanz", 1, 2000, params.get("dedup_distanz", 50))
        kernel_size_open = st.slider("Kernel Größe (Öffnen)", 1, 31, params.get("kernel_size_open", 3))
        kernel_size_close = st.slider("Kernel Größe (Schließen)", 1, 31, params.get("kernel_size_close", 3))
        circle_radius = st.slider("Marker-Radius (Display)", 1, 24, params.get("marker_radius", 5))
        hema_vec_str = st.text_input("Hematoxylin vector (R,G,B)", value=params.get("hema_vec", DEFAULT_PARAMS["hema_vec"]))
        chromo_vec_str = st.text_input("Chromogen vector (R,G,B)", value=params.get("aec_vec", DEFAULT_PARAMS["aec_vec"]))

    hema_vec0 = np.array([float(x.strip()) for x in hema_vec_str.split(",")], dtype=float)
    chromo_vec0 = np.array([float(x.strip()) for x in chromo_vec_str.split(",")], dtype=float)

    # Display sizing
    disp_width = st.sidebar.slider("Anzeige-Breite (px)", 300, 1600, st.session_state.disp_width)
    st.session_state.disp_width = disp_width
    scale = disp_width / float(W_orig)
    H_disp = int(round(H_orig * scale))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image_disp = cv2.resize(image_orig, (disp_width, H_disp), interpolation=interp)

    # Draw display canvas and existing groups
    display_canvas = image_disp.copy()
    display_canvas = draw_scale_bar(display_canvas, scale, length_orig=200, bar_height=3)

    for i, g in enumerate(st.session_state.groups):
        col = tuple(int(x) for x in g.get("color", PRESET_COLORS[i % len(PRESET_COLORS)]))
        for (x_orig, y_orig) in g["points"]:
            x_disp = int(round(x_orig * scale))
            y_disp = int(round(y_orig * scale))
            cv2.circle(display_canvas, (x_disp, y_disp), circle_radius, col, -1)
        if g["points"]:
            px_disp = int(round(g["points"][0][0] * scale))
            py_disp = int(round(g["points"][0][1] * scale))
            cv2.putText(display_canvas, f"G{i+1}:{len(g['points'])}", (px_disp + 6, py_disp - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # Image click handling using streamlit_image_coordinates if available, otherwise simple clickable fallback
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        coords = streamlit_image_coordinates(Image.fromarray(display_canvas), key=f"clickable_{st.session_state.last_file}", width=disp_width)
    except Exception:
        # fallback: show image and request coordinates via number inputs (less convenient)
        st.image(display_canvas, caption="Gezählte Kerne (Gruppenfarben)", use_column_width=True)
        coords = None

    # Sidebar actions
    mode = st.sidebar.radio("Aktion", ["Kalibriere und zähle Gruppe (Klick)", "Punkt löschen", "Undo letzte Aktion"]) 
    if st.sidebar.button("Reset (Alle Gruppen)"):
        st.session_state.history.append(("reset", {"groups": st.session_state.groups.copy(), "all_points": st.session_state.all_points.copy()}))
        st.session_state.groups = []
        st.session_state.all_points = []
        st.session_state.C_cache = None
        st.success("Zurückgesetzt.")

    # compute M and C cache lazily
    def get_deconv_cache(target_vec: np.ndarray):
        M = make_stain_matrix(target_vec, hema_vec0)
        M_hash = hash(M.tobytes())
        recompute = (st.session_state.C_cache is None) or (st.session_state.last_M_hash != M_hash)
        if recompute:
            st.session_state.C_cache = compute_deconvolution_cache(image_orig, M)
            st.session_state.last_M_hash = M_hash
        return st.session_state.C_cache

    # Handle clicks
    if coords:
        x_disp = int(coords["x"])
        y_disp = int(coords["y"])
        x_orig = int(round(x_disp / scale))
        y_orig = int(round(y_disp / scale))

        if mode == "Punkt löschen":
            removed = []
            new_all = []
            for p in st.session_state.all_points:
                if is_near(p, (x_orig, y_orig), dedup_dist_orig):
                    removed.append(p)
                else:
                    new_all.append(p)
            if removed:
                st.session_state.history.append(("delete_points", {"removed": removed}))
                st.session_state.all_points = new_all
                for g in st.session_state.groups:
                    g["points"] = [p for p in g["points"] if not is_near(p, (x_orig, y_orig), dedup_dist_orig)]
                st.success(f"{len(removed)} Punkt(e) gelöscht.")
            else:
                st.info("Kein Punkt in der Nähe gefunden.")

        elif mode == "Undo letzte Aktion":
            if st.session_state.history:
                action, payload = st.session_state.history.pop()
                if action == "add_group":
                    idx = payload["group_idx"]
                    if 0 <= idx < len(st.session_state.groups):
                        grp = st.session_state.groups.pop(idx)
                        for pt in grp["points"]:
                            st.session_state.all_points = [p for p in st.session_state.all_points if p != pt]
                    st.success("Letzte Gruppen-Aktion rückgängig gemacht.")
                elif action == "delete_points":
                    removed = payload["removed"]
                    st.session_state.all_points.extend(removed)
                    st.session_state.groups.append({"vec": None, "points": removed, "color": PRESET_COLORS[len(st.session_state.groups) % len(PRESET_COLORS)]})
                    st.success("Gelöschte Punkte wiederhergestellt (als neue Gruppe).")
                elif action == "reset":
                    st.session_state.groups = payload["groups"]
                    st.session_state.all_points = payload["all_points"]
                    st.success("Reset rückgängig gemacht.")
                else:
                    st.warning("Undo: unbekannte Aktion.")
            else:
                st.info("Keine Aktion zum Rückgängig machen.")

        else:
            # Calibration + detection
            patch = extract_patch(image_orig, x_orig, y_orig, calib_radius)
            vec = median_od_vector_from_patch(patch)
            if vec is None:
                st.warning("Patch unbrauchbar (zu homogen oder außerhalb). Bitte anders klicken.")
            else:
                C_full = get_deconv_cache(vec)
                channel_full = C_full[:, :, 0]
                centers_orig, mask = detect_centers_from_channel(channel_full, threshold=0.2, min_area=min_area_orig,
                                                                  kernel_size_open=kernel_size_open, kernel_size_close=kernel_size_close, debug=False)
                new_centers = dedup_new_points(centers_orig, st.session_state.all_points, min_dist=dedup_dist_orig)

                # add click point if not too close
                if not any(is_near(p, (x_orig, y_orig), dedup_dist_orig) for p in st.session_state.all_points):
                    new_centers.append((x_orig, y_orig))

                if new_centers:
                    color = PRESET_COLORS[len(st.session_state.groups) % len(PRESET_COLORS)]
                    group = {"vec": vec.tolist(), "points": new_centers, "color": color}
                    st.session_state.history.append(("add_group", {"group_idx": len(st.session_state.groups)}))
                    st.session_state.groups.append(group)
                    st.session_state.all_points.extend(new_centers)
                    st.success(f"Gruppe hinzugefügt — neue Kerne: {len(new_centers)} (inkl. Klickpunkt, falls neu)")
                else:
                    st.info("Keine neuen Kerne (alle bereits gezählt oder keine Detektion).")

    # Right column: results and export
    with col_right:
        st.markdown("### Zusammenfassung")
        st.write(f"🔹 Gruppen gesamt: {len(st.session_state.groups)}")
        for i, g in enumerate(st.session_state.groups):
            st.write(f"• Gruppe {i+1}: {len(g['points'])} Kerne — Vec: {np.round(g['vec'],3) if g['vec'] else None}")
        st.markdown(f"**Gesamt (unique Kerne): {len(st.session_state.all_points)}**")

        # CSV Export
        if st.session_state.all_points:
            rows = []
            for i, g in enumerate(st.session_state.groups):
                for (x_orig, y_orig) in g["points"]:
                    x_disp = int(round(x_orig * scale))
                    y_disp = int(round(y_orig * scale))
                    rows.append({
                        "Group": i + 1,
                        "Color": str(g.get("color")),
                        "Vec_R": g.get("vec")[0] if g.get("vec") else None,
                        "Vec_G": g.get("vec")[1] if g.get("vec") else None,
                        "Vec_B": g.get("vec")[2] if g.get("vec") else None,
                        "X_display": int(x_disp),
                        "Y_display": int(y_disp),
                        "X_original": int(x_orig),
                        "Y_original": int(y_orig)
                    })
            df = pd.DataFrame(rows)
            st.download_button("📥 CSV exportieren (Gruppen, inkl. Original-Koords)", df.to_csv(index=False).encode("utf-8"), file_name="kern_gruppen_v3.csv", mime="text/csv")

    # Show image (main area)
    with col_left:
        st.image(display_canvas, caption="Gezählte Kerne (Gruppenfarben)", use_column_width=True)

    # Footer notes
    st.markdown("---")
    st.caption("Hinweise: Deconvolution wird auf dem ORIGINALbild ausgeführt. CLAHE sollte nicht vor der Deconvolution angewendet werden.")


# -------------------- Parameter storage --------------------

def load_or_create_params() -> Dict[str, dict]:
    if os.path.exists(PARAM_FILE):
        try:
            with open(PARAM_FILE, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.warning("Konnte params.json nicht lesen, benutze Default: %s", e)
    # fallback write defaults
    with open(PARAM_FILE, "w") as f:
        json.dump({"default": DEFAULT_PARAMS}, f, indent=2)
    return {"default": DEFAULT_PARAMS}


if __name__ == "__main__":
    main()
