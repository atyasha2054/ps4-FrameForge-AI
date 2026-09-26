from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from .crop_engine import detect_faces, deduplicate_faces


def _inside_safe_zone(box, width, height, margin):
    x, y, w, h = box
    left = width * margin
    right = width * (1 - margin)
    top = height * margin
    bottom = height * (1 - margin)
    return x >= left and y >= top and x + w <= right and y + h <= bottom


def validate_image(path, spec, min_face_margin_px=12, expected_faces=None):
    image = cv2.imread(str(path))
    checks = {}
    reasons = []
    if image is None:
        return {"asset": Path(path).name, "status": "FAIL", "checks": {"file_integrity": "FAIL"}, "reasons": ["Unreadable image"]}
    h, w = image.shape[:2]
    checks["dimensions"] = "PASS" if (w, h) == (spec["width"], spec["height"]) else "FAIL"
    checks["aspect_ratio"] = "PASS" if abs(w / h - spec["width"] / spec["height"]) < 0.002 else "FAIL"
    if checks["dimensions"] == "FAIL": reasons.append(f"Expected {spec['width']}x{spec['height']}, got {w}x{h}")

    faces = deduplicate_faces(detect_faces(image, scale=0.5))
    visible = True
    safe = True
    margins = []
    m = spec.get("face_safe_margin", 0.02)
    for f in faces:
        margins.extend([f.x, f.y, w - (f.x + f.w), h - (f.y + f.h)])
        if f.x < -min_face_margin_px or f.y < -min_face_margin_px or f.x + f.w > w + min_face_margin_px or f.y + f.h > h + min_face_margin_px:
            visible = False
        if not _inside_safe_zone((f.x, f.y, f.w, f.h), w, h, m):
            safe = False
    if expected_faces is not None and len(faces) < expected_faces:
        visible = False
        reasons.append(f"Expected at least {expected_faces} detected face(s), found {len(faces)}")
    checks["face_visibility"] = "PASS" if visible else "FAIL"
    checks["safe_zone"] = "PASS" if safe else "FAIL"
    if not visible: reasons.append("At least one detected face is clipped")
    if not safe: reasons.append("At least one detected face crosses the configured safe zone")

    size_mb = Path(path).stat().st_size / (1024 * 1024)
    checks["file_size"] = "PASS" if size_mb <= spec.get("max_file_mb", 50) else "FAIL"
    if checks["file_size"] == "FAIL": reasons.append("File exceeds platform size limit")

    status = "PASS" if all(v == "PASS" for v in checks.values()) else "FAIL"
    return {
        "asset": Path(path).name,
        "status": status,
        "checks": checks,
        "metrics": {
            "faces_detected": len(faces),
            "min_face_margin_px": int(min(margins)) if margins else None,
            "width": w,
            "height": h,
            "file_size_mb": round(size_mb, 3)
        },
        "reasons": reasons
    }


def validate_video(path, spec, sample_seconds=1.0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"asset": Path(path).name, "status": "FAIL", "checks": {"file_integrity": "FAIL"}, "reasons": ["Unreadable video"]}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n / fps if fps else 0
    checks = {
        "dimensions": "PASS" if (w, h) == (spec["width"], spec["height"]) else "FAIL",
        "aspect_ratio": "PASS" if abs(w / h - spec["width"] / spec["height"]) < 0.002 else "FAIL",
        "file_integrity": "PASS"
    }
    reasons = []
    if checks["dimensions"] == "FAIL": reasons.append(f"Expected {spec['width']}x{spec['height']}, got {w}x{h}")
    # Sample frames for face visibility and safe-zone compliance.
    safe = True; visible = True; samples = 0
    for t in np.linspace(0, max(0, duration - 0.05), max(1, min(12, int(duration / sample_seconds) + 1))):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000))
        ok, frame = cap.read()
        if not ok: continue
        faces = deduplicate_faces(detect_faces(frame, scale=0.5))
        m = spec.get("face_safe_margin", 0.02)
        for f in faces:
            samples += 1
            if f.x < 0 or f.y < 0 or f.x + f.w > w or f.y + f.h > h: visible = False
            if not _inside_safe_zone((f.x, f.y, f.w, f.h), w, h, m): safe = False
    cap.release()
    checks["face_visibility"] = "PASS" if visible else "FAIL"
    checks["safe_zone"] = "PASS" if safe else "FAIL"
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    checks["file_size"] = "PASS" if size_mb <= spec.get("max_file_mb", 50) else "FAIL"
    if not visible: reasons.append("A sampled frame contains a clipped face")
    if not safe: reasons.append("A sampled face crosses the safe zone")
    if checks["file_size"] == "FAIL": reasons.append("File exceeds platform size limit")
    return {
        "asset": Path(path).name,
        "status": "PASS" if all(v == "PASS" for v in checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {"width": w, "height": h, "fps": round(fps, 3), "duration_s": round(duration, 2), "sampled_faces": samples, "file_size_mb": round(size_mb, 3)},
        "reasons": reasons
    }
