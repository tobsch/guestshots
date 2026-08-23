"""Face detection, identity clustering and per-frame quality features (InsightFace)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_app = None
_mesh = None

# MediaPipe FaceMesh eye contours for eye-aspect-ratio (p0,p3 = corners; p1/p5, p2/p4 = lids)
MP_EYE_R = (33, 160, 158, 133, 153, 144)
MP_EYE_L = (362, 385, 387, 263, 373, 380)


def get_mesh():
    global _mesh
    if _mesh is None:
        import mediapipe as mp
        _mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                refine_landmarks=True, min_detection_confidence=0.3)
    return _mesh


def mesh_ear(img: np.ndarray, bbox: tuple[int, int, int, int]) -> float | None:
    """Eye aspect ratio from MediaPipe FaceMesh on a padded face crop. None if no mesh found."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = max(0, int(x1 - 0.4 * w)), max(0, int(y1 - 0.4 * h))
    cx2, cy2 = min(W, int(x2 + 0.4 * w)), min(H, int(y2 + 0.4 * h))
    crop = img[cy1:cy2, cx1:cx2]
    res = get_mesh().process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    L = res.multi_face_landmarks[0].landmark
    ch, cw = crop.shape[:2]

    def ear(idx):
        p = np.array([[L[i].x * cw, L[i].y * ch] for i in idx])
        return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-6)

    return float((ear(MP_EYE_R) + ear(MP_EYE_L)) / 2)


def _cgroup_cpu_quota() -> int:
    """CPUs allowed by a cgroup v2 quota (docker --cpus), 0 if unlimited / not in a cgroup."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        return max(1, round(int(quota) / int(period))) if quota != "max" else 0
    except (OSError, ValueError):
        return 0


def get_app():
    """InsightFace bundle. Tunables (env): GUESTSHOTS_THREADS (onnxruntime intra-op threads — set this in
    containers with a CPU quota, otherwise ORT spawns one thread per host core and thrashes),
    GUESTSHOTS_DET_SIZE (detector input, default 640), GUESTSHOTS_CPU_ONLY=1 (skip CoreML)."""
    global _app
    if _app is None:
        import os
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import model_zoo

        threads = int(os.environ.get("GUESTSHOTS_THREADS", "0")) or _cgroup_cpu_quota()
        if threads:
            # Measured in a 6-CPU docker quota: ORT default (one spinning thread per host core) = 0.9 fps,
            # 6 threads = 6 fps, 6 threads + no spinning = 11.8 fps.
            orig_init = model_zoo.PickableInferenceSession.__init__

            def init(self, model_path, **kw):
                so = kw.get("sess_options") or ort.SessionOptions()
                so.intra_op_num_threads = threads
                so.inter_op_num_threads = 1
                so.add_session_config_entry("session.intra_op.allow_spinning", "0")
                kw["sess_options"] = so
                orig_init(self, model_path, **kw)

            model_zoo.PickableInferenceSession.__init__ = init

        wanted = ("CPUExecutionProvider",) if os.environ.get("GUESTSHOTS_CPU_ONLY") else ("CoreMLExecutionProvider", "CPUExecutionProvider")
        providers = [p for p in wanted if p in ort.get_available_providers()]
        det = int(os.environ.get("GUESTSHOTS_DET_SIZE", "640"))
        _app = FaceAnalysis(name="buffalo_l", providers=providers,
                            allowed_modules=["detection", "recognition", "landmark_2d_106"])
        _app.prepare(ctx_id=0, det_size=(det, det))
    return _app


@dataclass
class FaceObs:
    """One detected face in one frame."""
    t: float
    frame: Path
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2 in analysis-frame pixels
    det_score: float
    embedding: np.ndarray  # normalized
    yaw: float  # deg, 0 = frontal
    pitch: float
    eye_open: float  # InsightFace-landmark eye aspect ratio (unreliable for squints)
    ear_mp: float  # MediaPipe FaceMesh eye aspect ratio (≈0.25+ open, ≤0.14 closed); -1 if no mesh
    mouth_open: float  # mouth aspect ratio (≈0.2 closed, 0.5+ wide open)
    smile: float  # mouth width / inter-ocular distance (≈1.3 neutral, 1.6+ smiling)
    sharpness: float  # laplacian variance on face crop
    brightness: float  # mean luma of face crop 0..255
    area_frac: float  # face area / frame area
    identity: int = -1


# InsightFace 2d106 landmark layout
EYE_R = list(range(33, 43))   # image-left eye
EYE_L = list(range(87, 97))   # image-right eye
MOUTH = list(range(52, 72))
MOUTH_CORNERS = (52, 61)


def _aspect(pts: np.ndarray) -> float:
    """vertical extent / horizontal extent of a point set."""
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    return float(h / w) if w > 1 else 0.0


def _pose_from_kps(kps: np.ndarray) -> tuple[float, float]:
    """Approximate yaw/pitch (deg) from the 5 keypoints: eyes, nose, mouth corners."""
    le, re, nose, lm, rm = kps
    eye_c = (le + re) / 2
    mouth_c = (lm + rm) / 2
    eye_w = np.linalg.norm(re - le) + 1e-6
    # nose x offset relative to eye-center, normalized by eye width
    yaw = float(np.clip((nose[0] - eye_c[0]) / eye_w * 90, -90, 90))
    # nose y relative position between eyes and mouth (0.5 ≈ frontal)
    span = (mouth_c[1] - eye_c[1]) + 1e-6
    rel = (nose[1] - eye_c[1]) / span
    pitch = float(np.clip((rel - 0.6) * 120, -90, 90))
    return yaw, pitch


def detect(t: float, frame_path: Path, min_face: int = 60) -> list[FaceObs]:
    img = cv2.imread(str(frame_path))
    if img is None:
        return []
    return detect_img(img, t, frame_path, min_face)


def detect_img(img: np.ndarray, t: float, frame_path: Path | None = None, min_face: int = 60) -> list[FaceObs]:
    H, W = img.shape[:2]
    faces = get_app().get(img)
    out = []
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if (x2 - x1) < min_face or (y2 - y1) < min_face:
            continue
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(gray.mean())

        kps = np.asarray(f.kps)
        yaw, pitch = _pose_from_kps(kps)
        if getattr(f, "pose", None) is not None:
            pitch, yaw = float(f.pose[0]), float(f.pose[1])

        lm = getattr(f, "landmark_2d_106", None)
        eye_open = mouth_open = smile = 0.0
        if lm is not None:
            lm = np.asarray(lm)
            eye_open = (_aspect(lm[EYE_R]) + _aspect(lm[EYE_L])) / 2
            mouth_open = _aspect(lm[MOUTH])
            iod = np.linalg.norm(kps[1] - kps[0]) + 1e-6
            smile = float(np.linalg.norm(lm[MOUTH_CORNERS[1]] - lm[MOUTH_CORNERS[0]]) / iod)

        ear = mesh_ear(img, (x1, y1, x2, y2))
        emb = np.asarray(f.normed_embedding, dtype=np.float32)
        out.append(FaceObs(
            t=t, frame=frame_path, bbox=(x1, y1, x2, y2), det_score=float(f.det_score),
            embedding=emb, yaw=yaw, pitch=pitch, eye_open=eye_open, ear_mp=ear if ear is not None else -1.0,
            mouth_open=mouth_open, smile=smile,
            sharpness=sharp, brightness=bright, area_frac=((x2 - x1) * (y2 - y1)) / (W * H),
        ))
    return out


@dataclass
class Identity:
    id: int
    centroid: np.ndarray
    count: int = 0
    sample: FaceObs | None = None
    _sum: np.ndarray = field(default_factory=lambda: None)


def cluster(obs: list[FaceObs], threshold: float = 0.45) -> list[Identity]:
    """Greedy online clustering on cosine similarity of normed embeddings."""
    ids: list[Identity] = []
    for o in obs:
        best, best_sim = None, -1.0
        for ident in ids:
            sim = float(o.embedding @ ident.centroid)
            if sim > best_sim:
                best, best_sim = ident, sim
        if best is None or best_sim < threshold:
            best = Identity(id=len(ids), centroid=o.embedding.copy(), _sum=o.embedding.copy())
            ids.append(best)
        else:
            best._sum = best._sum + o.embedding
            best.centroid = best._sum / (np.linalg.norm(best._sum) + 1e-9)
        best.count += 1
        o.identity = best.id
        if best.sample is None or o.area_frac * o.sharpness > best.sample.area_frac * best.sample.sharpness:
            best.sample = o
    # re-assign with final centroids to fix early drift
    for o in obs:
        sims = [float(o.embedding @ i.centroid) for i in ids]
        o.identity = int(np.argmax(sims))
    for i in ids:
        i.count = sum(1 for o in obs if o.identity == i.id)
    return ids


def embed_image(path: Path) -> np.ndarray | None:
    """Embedding of the largest face in a reference photo."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    faces = get_app().get(img)
    if not faces:
        return None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return np.asarray(f.normed_embedding, dtype=np.float32)


def save_crop(o: FaceObs, out: Path, pad: float = 0.4, img: np.ndarray | None = None, scale: float = 1.0) -> None:
    """Crop the face (+pad) from the analysis frame, or from `img` (e.g. a full-res grab) scaled by `scale`."""
    if img is None:
        img = cv2.imread(str(o.frame))
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v * scale) for v in o.bbox]
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - pad * w)), max(0, int(y1 - pad * h))
    x2, y2 = min(W, int(x2 + pad * w)), min(H, int(y2 + pad * h))
    cv2.imwrite(str(out), img[y1:y2, x1:x2])
