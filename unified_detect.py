"""
TITAN PPE - Unified Safety Monitor (Production)
================================================
Model 1  │ HuggingFace PPE (YOLOv8n, 5 MB)
         │ Role: Violation classifier
         │ Classes: Hardhat, NO-Hardhat, NO-Safety Vest, Person, Safety Vest, ...
         │ Strength: explicit NO-Hardhat / NO-Safety Vest violation classes
         │
Model 2  │ Local trained (YOLOv8m, 49 MB, 82.3% precision)
         │ Role: High-precision full-PPE locator
         │ Classes: person, helmet, safety_vest, gloves, safety_glasses, safety_shoes, mask
         │ Strength: accurate bounding boxes + detects gloves, glasses, shoes
         │
Fusion   │ Both outputs mapped to a unified class space → NMS merge
         │ Model 2 boxes preferred (higher precision)
         │ Model 1 NO-Hardhat / NO-Vest classes kept exclusively
         │ Result: one super-model output with full PPE coverage
         │
Accuracy │ imgsz=1280 · conf=0.35 · multi-frame voting (3-frame buffer)
"""

import os
import sys
import cv2
import time
import threading
import torch
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import deque

# ================================================================
#  PATHS  (all local — no internet in production)
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL1_PATH = os.path.join(SCRIPT_DIR, "models", "ppe_hf.onnx")      # HuggingFace PPE (local ONNX)
MODEL2_PATH = os.path.join(SCRIPT_DIR, "w", "weights", "best.onnx")      # Local trained (ONNX)

# ================================================================
#  CONFIG
# ================================================================
WEBCAM_INDEX     = 0
IMGSZ            = 640      # Native ONNX model resolution (highly optimized for CPU/GPU)
CONF             = 0.65    # high confidence threshold to prevent false positives and background noise
NMS_IOU          = 0.45
SMOOTH_ALPHA     = 0.30     # EMA smoothing (lower = more stable boxes)
VOTE_FRAMES      = 3        # frames in voting buffer
VOTE_MIN         = 2        # min votes to confirm a violation (2 out of 3)
ALERT_COOLDOWN   = 5        # seconds between auto-saves
VIOLATIONS_DIR   = "violations"

# ================================================================
#  UNIFIED CLASS SPACE
# Both models normalised to these keys
# ================================================================
UNIFIED = {
    "person":    {"color": (200, 200,  0), "show": True},
    "helmet":    {"color": (  0, 210, 80), "show": True},
    "vest":      {"color": (  0, 180,255), "show": True},
    "gloves":    {"color": (255, 140,  0), "show": True},
    "glasses":   {"color": (255, 200,  0), "show": True},
    "shoes":     {"color": (200,  80,255), "show": True},
    "mask":      {"color": ( 80, 255,140), "show": True},
    "no_helmet": {"color": (  0,   0,240), "show": True},
    "no_vest":   {"color": (  0,  80,230), "show": True},
    "no_glasses":{"color": (  0,   0,240), "show": True},
    "no_mask":   {"color": (  0,   0,240), "show": True},
}

# Model 1 class_id → unified key  (None = discard)
M1_MAP = {
    0: "helmet",     # Hardhat
    1: "mask",       # Mask
    2: "no_helmet",  # NO-Hardhat          ← violation classifier
    3: None,         # NO-Mask             (not used)
    4: "no_vest",    # NO-Safety Vest      ← violation classifier
    5: "person",     # Person
    6: None,         # Safety Cone         (not used)
    7: "vest",       # Safety Vest
    8: None,         # machinery
    9: None,         # vehicle
}

# Model 2 class_id → unified key
M2_MAP = {
    0: "person",     # person
    1: "mask",       # mask
    2: "glasses",    # safety_glasses
    3: "gloves",     # gloves
    4: "helmet",     # helmet
    5: "vest",       # safety_vest
    6: "shoes",      # safety_shoes
}

# ================================================================
#  HELPERS: IOU / OVERLAP
# ================================================================
def _iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/union if union>0 else 0.0

def _overlap_frac(outer, inner):
    ax1,ay1,ax2,ay2 = outer
    bx1,by1,bx2,by2 = inner
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    area  = max(1,(bx2-bx1)*(by2-by1))
    return inter/area

def on_person(person_box, item_box):
    ax1,ay1,ax2,ay2 = person_box
    bx1,by1,bx2,by2 = item_box
    cx,cy = (bx1+bx2)/2, (by1+by2)/2
    center_in = ax1<=cx<=ax2 and ay1<=cy<=ay2
    return center_in or _overlap_frac(person_box, item_box) > 0.5

# ================================================================
#  EXTRACT — convert raw YOLO result to unified detection list
# ================================================================
def extract(result, class_map, priority):
    dets = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        key    = class_map.get(cls_id)
        if key is None:
            continue
        dets.append({
            "key":      key,
            "box":      box.xyxy[0].tolist(),
            "conf":     float(box.conf[0]),
            "priority": priority,
        })
    return dets

# ================================================================
#  NMS FUSION — merge both models into one detection list
# ================================================================
def fuse(r1, r2):
    # Model 2 listed first = preferred in NMS (higher precision boxes)
    m2_dets = extract(r2, M2_MAP, priority=2)
    m1_dets = extract(r1, M1_MAP, priority=1)

    # Explicit violation classes are ONLY in model 1 — always keep them
    violations = [d for d in m1_dets if d["key"] in ("no_helmet","no_vest")]
    regular    = [d for d in m2_dets] + [d for d in m1_dets if d["key"] not in ("no_helmet","no_vest")]

    # NMS per class
    by_class = {}
    for d in regular:
        by_class.setdefault(d["key"], []).append(d)

    kept = []
    for key, group in by_class.items():
        kept.extend(_nms_group(group))

    return kept + violations


def _nms_group(group):
    if len(group) == 1:
        return group
    # sort by priority then confidence
    group = sorted(group, key=lambda d: (d["priority"], d["conf"]), reverse=True)
    result = []
    while group:
        best = group.pop(0)
        result.append(best)
        group = [d for d in group if _iou(best["box"], d["box"]) < NMS_IOU]
    return result

# ================================================================
#  TEMPORAL SMOOTHER — EMA on box coords, reduces jitter
# ================================================================
class Smoother:
    def __init__(self, alpha=SMOOTH_ALPHA, iou_min=0.3, max_lost_frames=3):
        self.alpha   = alpha
        self.iou_min = iou_min
        self.max_lost_frames = max_lost_frames
        self.prev    = []  # list of dicts: {"key", "box", "conf", "priority", "lost_count", "matched"}

    def update(self, dets):
        # Mark all previous detections as unmatched
        for p in self.prev:
            p["matched"] = False

        out = []
        used_prev = set()

        for d in dets:
            best_iou, best_i = 0.0, -1
            for i, p in enumerate(self.prev):
                if p["key"] != d["key"] or i in used_prev:
                    continue
                iou = _iou(d["box"], p["box"])
                if iou > best_iou:
                    best_iou, best_i = iou, i

            if best_iou >= self.iou_min:
                used_prev.add(best_i)
                p = self.prev[best_i]
                p["matched"] = True
                p["lost_count"] = 0
                
                # EMA box smoothing
                smooth_box = [
                    self.alpha * d["box"][k] + (1-self.alpha) * p["box"][k]
                    for k in range(4)
                ]
                # Smooth confidence score
                smooth_conf = self.alpha * d["conf"] + (1-self.alpha) * p["conf"]
                
                out.append({
                    "key": d["key"],
                    "box": smooth_box,
                    "conf": smooth_conf,
                    "priority": d["priority"],
                    "lost_count": 0,
                    "matched": True
                })
            else:
                # New detection
                d["lost_count"] = 0
                d["matched"] = True
                out.append(d)

        # Keep unmatched previous detections for up to max_lost_frames to prevent flickering
        for i, p in enumerate(self.prev):
            if i not in used_prev:
                p["lost_count"] = p.get("lost_count", 0) + 1
                if p["lost_count"] <= self.max_lost_frames:
                    # Decay confidence slightly
                    p["conf"] *= 0.85
                    out.append(p)

        self.prev = out
        # Only return detections that are active or within the persistence window with sufficient confidence
        return [d for d in out if d["conf"] >= 0.25]

# ================================================================
#  MULTI-FRAME VOTING — 3-frame buffer, need 2 votes to confirm
# ================================================================
class ViolationVoter:
    def __init__(self, frames=VOTE_FRAMES, min_votes=VOTE_MIN):
        self.buf       = deque(maxlen=frames)
        self.min_votes = min_votes

    def update(self, v_h, v_v, v_g, v_gl, v_m):
        self.buf.append((v_h, v_v, v_g, v_gl, v_m))
        votes_h = sum(1 for x in self.buf if x[0])
        votes_v = sum(1 for x in self.buf if x[1])
        votes_g = sum(1 for x in self.buf if x[2])
        votes_gl = sum(1 for x in self.buf if x[3])
        votes_m = sum(1 for x in self.buf if x[4])
        confirmed_h = votes_h >= self.min_votes
        confirmed_v = votes_v >= self.min_votes
        confirmed_g = votes_g >= self.min_votes
        confirmed_gl = votes_gl >= self.min_votes
        confirmed_m = votes_m >= self.min_votes
        return confirmed_h, confirmed_v, confirmed_g, confirmed_gl, confirmed_m

# ================================================================
#  VIOLATION LOGIC — works on unified detection list
# ================================================================
def check_violations(dets):
    # Only evaluate violations on persons detected with robust confidence >= 0.55
    persons    = [d for d in dets if d["key"]=="person" and d["conf"] >= 0.55]
    helmets    = [d for d in dets if d["key"]=="helmet"]
    vests      = [d for d in dets if d["key"]=="vest"]
    gloves     = [d for d in dets if d["key"]=="gloves"]
    glasses    = [d for d in dets if d["key"]=="glasses"]
    masks      = [d for d in dets if d["key"]=="mask"]
    no_helmets = [d for d in dets if d["key"]=="no_helmet"]
    no_vests   = [d for d in dets if d["key"]=="no_vest"]

    v_h = len(no_helmets) > 0
    v_v = len(no_vests)   > 0
    v_g = False
    v_gl = False
    v_m = False

    for p in persons:
        pb = p["box"]
        has_helmet  = any(on_person(pb, h["box"]) for h in helmets)
        has_vest    = any(on_person(pb, v["box"]) for v in vests)
        has_no_h    = any(on_person(pb, n["box"]) for n in no_helmets)
        has_no_v    = any(on_person(pb, n["box"]) for n in no_vests)
        has_gloves  = any(on_person(pb, g["box"]) for g in gloves)
        has_glasses = any(on_person(pb, gl["box"]) for gl in glasses)
        has_mask    = any(on_person(pb, m["box"]) for m in masks)

        if has_no_h or not has_helmet: v_h = True
        if has_no_v or not has_vest:   v_v = True
        if not has_gloves:             v_g = True
        
        # Estimate head/face based on helmet if present, else estimate from person box
        associated_helmets = [h for h in helmets if on_person(pb, h["box"])]
        
        if associated_helmets:
            # We have a high-precision helmet box on the head!
            h_box = associated_helmets[0]["box"]
            hx1, hy1, hx2, hy2 = h_box
            h_h = hy2 - hy1
            h_w = hx2 - hx1
            
            # Glasses should be right below the helmet (eyes region)
            gl_x1 = max(0.0, hx1 + h_w * 0.1)
            gl_y1 = max(0.0, hy2 - h_h * 0.1)
            gl_x2 = max(0.0, hx2 - h_w * 0.1)
            gl_y2 = max(0.0, hy2 + h_h * 0.4)
            
            # Mask should be below the glasses (mouth & nose region)
            m_x1 = max(0.0, hx1 + h_w * 0.15)
            m_y1 = max(0.0, hy2 + h_h * 0.35)
            m_x2 = max(0.0, hx2 - h_w * 0.15)
            m_y2 = max(0.0, hy2 + h_h * 1.0)
        else:
            # No helmet, estimate head from person box width to remain invariant to vertical cropping
            px1, py1, px2, py2 = pb
            pw = px2 - px1
            ph = py2 - py1
            
            # Estimate head dimensions based on width
            hw = pw * 0.45
            hh = hw * 1.2
            center_x = px1 + pw / 2.0
            
            # Head bounding box
            hx1 = max(px1, center_x - hw / 2.0)
            hy1 = py1
            hx2 = min(px2, center_x + hw / 2.0)
            hy2 = min(py2, py1 + hh)
            
            # Eyes / Glasses (middle-upper part of the head box)
            gl_x1 = max(0.0, hx1 + (hx2 - hx1) * 0.1)
            gl_y1 = max(0.0, hy1 + (hy2 - hy1) * 0.25)
            gl_x2 = max(0.0, hx2 - (hx2 - hx1) * 0.1)
            gl_y2 = max(0.0, hy1 + (hy2 - hy1) * 0.55)
            
            # Mouth & Nose / Mask (middle-lower part of the head box)
            m_x1 = max(0.0, hx1 + (hx2 - hx1) * 0.15)
            m_y1 = max(0.0, hy1 + (hy2 - hy1) * 0.50)
            m_x2 = max(0.0, hx2 - (hx2 - hx1) * 0.15)
            m_y2 = max(0.0, hy1 + (hy2 - hy1) * 0.95)

        if not has_glasses:
            v_gl = True
            # Add virtual red box for missing glasses on face
            dets.append({
                "key": "no_glasses",
                "box": [gl_x1, gl_y1, gl_x2, gl_y2],
                "conf": 1.0,
                "priority": 3
            })
            
        if not has_mask:
            v_m = True
            # Add virtual red box for missing mask on mouth/chin region
            dets.append({
                "key": "no_mask",
                "box": [m_x1, m_y1, m_x2, m_y2],
                "conf": 1.0,
                "priority": 3
            })

    return v_h, v_v, v_g, v_gl, v_m, len(persons)

# ================================================================
#  ANNOTATION — single clean layer from fused detections
# ================================================================
def annotate(frame, dets):
    out = frame.copy()
    for d in dets:
        info = UNIFIED.get(d["key"])
        if info is None or not info["show"]:
            continue
        color = info["color"]
        x1,y1,x2,y2 = [int(v) for v in d["box"]]
        label = f'{d["key"].replace("_"," ").title()} {d["conf"]:.2f}'
        cv2.rectangle(out, (x1,y1),(x2,y2), color, 2)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-th-6),(x1+tw+4, y1), color, -1)
        cv2.putText(out, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return out


def draw_status(width, v_h, v_v, v_g, v_gl, v_m, fps, persons):
    bar = np.zeros((88, width, 3), dtype=np.uint8)
    if v_h or v_v or v_g or v_gl or v_m:
        bar[:] = (22, 0, 65)
        parts = []
        if v_h: parts.append("NO HELMET")
        if v_v: parts.append("NO VEST")
        if v_g: parts.append("NO GLOVES")
        if v_gl: parts.append("NO GLASSES")
        if v_m: parts.append("NO MASK")
        cv2.putText(bar, f"  VIOLATION: {' + '.join(parts)}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,240), 2)
        cv2.putText(bar, f"  Persons: {persons}", (10,72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210,210,210), 1)
    else:
        bar[:] = (10, 38, 10)
        msg = "ALL CLEAR" if persons > 0 else "MONITORING..."
        cv2.putText(bar, f"  {msg}", (10,52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,210,0), 2)

    cv2.putText(bar, f"FPS {fps:>3}", (width-130,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200,200,200), 2)
    return bar

# ================================================================
#  CAMERA CAPTURE THREAD — reads at full 30fps, always fresh
# ================================================================
class CameraCapture(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap    = cap
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

    def stop(self): self._stop.set()

    @property
    def frame(self):
        with self._lock:
            return self._frame

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror frame horizontally (right to left)
                with self._lock:
                    self._frame = frame
            else:
                # Camera disconnected / read failure! Attempt to reconnect.
                print("\n[Camera] Connection lost or read failed. Attempting to reconnect in 3 seconds...")
                self.cap.release()
                time.sleep(3)
                if sys.platform.startswith('win'):
                    self.cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(WEBCAM_INDEX)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if self.cap.isOpened():
                    print("[Camera] Reconnected successfully!")
                else:
                    print("[Camera] Reconnection attempt failed. Will retry...")


# ================================================================
#  INFERENCE THREAD — runs models async, never blocks display
# ================================================================
class InferenceEngine(threading.Thread):
    def __init__(self, model1, model2, device, camera: CameraCapture):
        super().__init__(daemon=True)
        self.m1       = model1
        self.m2       = model2
        self.device   = device
        self.camera   = camera
        self.smoother = Smoother()
        self.voter    = ViolationVoter()
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        # shared result (lock-protected)
        self._dets    = []
        self._v_h     = False
        self._v_v     = False
        self._v_g     = False
        self._v_gl    = False
        self._v_m     = False
        self._persons = 0

    def stop(self): self._stop.set()

    def result(self):
        with self._lock:
            return self._dets, self._v_h, self._v_v, self._v_g, self._v_gl, self._v_m, self._persons

    def run(self):
        while not self._stop.is_set():
            frame = self.camera.frame
            if frame is None:
                time.sleep(0.01)
                continue

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(self.m1.predict, frame, imgsz=IMGSZ, conf=CONF, device=self.device, verbose=False)
                f2 = executor.submit(self.m2.predict, frame, imgsz=IMGSZ, conf=CONF, device=self.device, verbose=False)
                r1 = f1.result()[0]
                r2 = f2.result()[0]

            fused    = fuse(r1, r2)
            smoothed = self.smoother.update(fused)

            raw_h, raw_v, raw_g, raw_gl, raw_m, persons = check_violations(smoothed)
            voted_h, voted_v, voted_g, voted_gl, voted_m  = self.voter.update(raw_h, raw_v, raw_g, raw_gl, raw_m)

            with self._lock:
                self._dets    = smoothed
                self._v_h     = voted_h
                self._v_v     = voted_v
                self._v_g     = voted_g
                self._v_gl    = voted_gl
                self._v_m     = voted_m
                self._persons = persons

# ================================================================
#  MAIN
# ================================================================
def main():
    global IMGSZ
    
    # Device selection (with macOS Apple Silicon GPU / MPS support)
    if torch.cuda.is_available():
        device = "0"
        gpu = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        gpu = "Apple Silicon GPU (MPS)"
    else:
        device = "cpu"
        gpu = "CPU"
        
    # ONNX Model requires exactly 640x640 size
    if IMGSZ != 640:
        print(f"\n[Config] Setting IMGSZ to 640 to match the native ONNX model dimensions.")
        IMGSZ = 640
        
    print(f"Device  : {gpu}")
    print(f"imgsz   : {IMGSZ}  |  conf: {CONF}  |  voting: {VOTE_MIN}/{VOTE_FRAMES} frames")

    os.makedirs(VIOLATIONS_DIR, exist_ok=True)

    print("Loading Model 1 (PPE classifier)  ...")
    model1 = YOLO(MODEL1_PATH)

    print("Loading Model 2 (precision locator)...")
    model2 = YOLO(MODEL2_PATH)

    print("Both models ready. Opening camera...\n")

    if sys.platform.startswith('win'):
        cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Cannot open camera!")
        return

    camera = CameraCapture(cap)
    camera.start()

    engine = InferenceEngine(model1, model2, device, camera)
    engine.start()

    win = "TITAN PPE - Safety Monitor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_alert = 0
    prev_time  = time.time()

    print("Running — press 'q' quit | 's' save snapshot")

    while True:
        # always use the live camera frame — never the processed one
        frame = camera.frame
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        dets, v_h, v_v, v_g, v_gl, v_m, persons = engine.result()
        annotated = annotate(frame, dets)

        now  = time.time()
        fps  = int(1 / max(now - prev_time, 1e-6))
        prev_time = now

        h, w = annotated.shape[:2]
        bar  = draw_status(w, v_h, v_v, v_g, v_gl, v_m, fps, persons)
        out  = np.vstack([bar, annotated])

        # Auto-save on confirmed violation
        if (v_h or v_v or v_g or v_gl or v_m) and (now - last_alert) > ALERT_COOLDOWN:
            last_alert = now
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts = []
            if v_h: parts.append("helmet")
            if v_v: parts.append("vest")
            if v_g: parts.append("gloves")
            if v_gl: parts.append("glasses")
            if v_m: parts.append("mask")
            vtype = "_".join(parts)
            path  = os.path.join(VIOLATIONS_DIR, f"{vtype}_{ts}.jpg")
            cv2.imwrite(path, out)
            print(f"[{ts}] VIOLATION — {vtype.replace('_',' ').upper()} | {path}")

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(VIOLATIONS_DIR, f"snapshot_{ts}.jpg")
            cv2.imwrite(path, out)
            print(f"[{ts}] Snapshot: {path}")

    engine.stop()
    camera.stop()
    engine.join(timeout=2)
    camera.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
