from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import urllib.request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "assets"
MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2


def _load_yunet():
    MODEL_DIR.mkdir(exist_ok=True)
    if not MODEL_PATH.exists():
        try:
            print("Downloading YuNet face detector...")
            with urllib.request.urlopen(MODEL_URL, timeout=10) as r:
                MODEL_PATH.write_bytes(r.read())
        except Exception as e:
            print(f"YuNet download failed: {e}. Falling back to Haar.")
            return None
    try:
        return cv2.FaceDetectorYN.create(
            str(MODEL_PATH), "", (320, 320), 0.60, 0.30, 5000
        )
    except Exception as e:
        print(f"YuNet initialization failed: {e}. Falling back to Haar.")
        return None


YUNET = _load_yunet()
HAAR = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))


def detect_faces(image_bgr, scale=1.0):
    h, w = image_bgr.shape[:2]
    faces = []
    if YUNET is not None:
        try:
            YUNET.setInputSize((w, h))
            _, detections = YUNET.detect(image_bgr)
            if detections is not None:
                for d in detections:
                    x, y, bw, bh, score = d[:5]
                    faces.append(Face(int(x), int(y), int(bw), int(bh), float(score)))
                return deduplicate_faces(faces, iou_threshold=0.35)
        except Exception:
            pass

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if scale != 1.0:
        gray_small = cv2.resize(gray, None, fx=scale, fy=scale)
    else:
        gray_small = gray
    boxes = HAAR.detectMultiScale(gray_small, 1.08, 7, minSize=(40, 40))
    inv = 1.0 / scale
    for x, y, bw, bh in boxes:
        x, y, bw, bh = [int(v * inv) for v in (x, y, bw, bh)]
        faces.append(Face(x, y, bw, bh, 0.75))
    return deduplicate_faces(faces, iou_threshold=0.35)


def deduplicate_faces(faces, iou_threshold=0.45):
    if not faces: return []
    boxes = [[f.x, f.y, f.w, f.h] for f in faces]
    scores = [float(f.score) for f in faces]
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(idxs) == 0: return faces
    idxs = np.array(idxs).reshape(-1)
    return [faces[int(i)] for i in idxs]


def subject_box_from_faces(faces, frame_shape):
    h, w = frame_shape[:2]
    if not faces: return (0, 0, w, h)
    # Expand each face into a rough full-person composition region.
    regions = []
    for f in faces:
        regions.append((
            f.x - 0.9*f.w,
            f.y - 0.8*f.h,
            f.x + 1.9*f.w,
            f.y + 5.0*f.h
        ))
    x1=min(r[0] for r in regions); y1=min(r[1] for r in regions)
    x2=max(r[2] for r in regions); y2=max(r[3] for r in regions)
    x1=max(0,int(x1)); y1=max(0,int(y1)); x2=min(w,int(x2)); y2=min(h,int(y2))
    return (x1,y1,max(x1+1,x2),max(y1+1,y2))


def target_crop_size(w, h, ratio):
    tw, th = ratio
    if w / h > tw / th:
        ch=h; cw=int(round(h*tw/th))
    else:
        cw=w; ch=int(round(w*th/tw))
    return min(w,cw), min(h,ch)


def optimize_crop(frame_bgr, ratio_name, specs, faces=None, focus_face=None):
    h,w=frame_bgr.shape[:2]
    spec=specs[ratio_name]
    cw,ch=target_crop_size(w,h,(spec['width'],spec['height']))
    faces=deduplicate_faces(faces or [])
    if focus_face is not None: faces=[focus_face]

    if faces:
        sx1,sy1,sx2,sy2=subject_box_from_faces(faces,frame_bgr.shape)
        if sx2-sx1 <= cw and sy2-sy1 <= ch:
            cx=(sx1+sx2)/2; cy=sy1+0.47*(sy2-sy1)
        else:
            weights=np.array([max(0.1,f.score) for f in faces])
            cx=float(np.average([f.cx for f in faces],weights=weights))
            cy=float(np.average([f.cy for f in faces],weights=weights))
    else:
        cx=w/2; cy=h/2

    x1=int(round(cx-cw/2)); y1=int(round(cy-ch/2))
    x1=max(0,min(x1,w-cw)); y1=max(0,min(y1,h-ch))

    # Force all detected faces to remain inside the crop whenever physically possible.
    if faces:
        minx=min(f.x for f in faces); maxx=max(f.x+f.w for f in faces)
        miny=min(f.y for f in faces); maxy=max(f.y+f.h for f in faces)
        if minx < x1: x1=max(0,minx)
        if maxx > x1+cw: x1=min(w-cw,max(0,maxx-cw))
        if miny < y1: y1=max(0,miny)
        if maxy > y1+ch: y1=min(h-ch,max(0,maxy-ch))
    return (int(x1),int(y1),int(x1+cw),int(y1+ch))


def crop_image(frame_bgr, box, output_size):
    x1,y1,x2,y2=box
    crop=frame_bgr[y1:y2,x1:x2]
    return cv2.resize(crop,output_size,interpolation=cv2.INTER_LANCZOS4)
