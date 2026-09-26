from __future__ import annotations

from pathlib import Path
import json, cv2
from PIL import Image
from .crop_engine import detect_faces, deduplicate_faces, optimize_crop, crop_image
from .validator import validate_image, validate_video
from .video_reframe import build_crop_timeline, render_vertical_reel

ROOT=Path(__file__).resolve().parents[1]
SPECS=json.loads((ROOT/"specs/platform_specs.json").read_text())
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True); LIB=ROOT/"library"; LIB.mkdir(exist_ok=True)


def generate_image_variants(image_path, ratios=None):
    ratios=ratios or list(SPECS.keys())
    img=cv2.imread(str(image_path))
    if img is None: raise ValueError("Could not read image")
    faces=deduplicate_faces(detect_faces(img))
    results=[]
    for ratio in ratios:
        spec=SPECS[ratio]
        box=optimize_crop(img,ratio,SPECS,faces=faces)
        out=OUT/f"image_{ratio.replace(':','x')}.jpg"
        crop=crop_image(img,box,(spec["width"],spec["height"]))
        cv2.imwrite(str(out),crop,[cv2.IMWRITE_JPEG_QUALITY,94])
        report=validate_image(out,spec,expected_faces=len(faces))
        if report["status"] == "PASS":
            import shutil; shutil.copy2(out, LIB/out.name)
        report["crop_box_source_px"]=list(map(int,box))
        report["subjects_detected"]=len(faces)
        (OUT/f"image_{ratio.replace(':','x')}.json").write_text(json.dumps(report,indent=2))
        results.append((ratio,out,report))
    return results



def regenerate_image_variant(image_path, ratio):
    img=cv2.imread(str(image_path))
    if img is None: raise ValueError("Could not read image")
    faces=deduplicate_faces(detect_faces(img))
    spec=SPECS[ratio]
    box=optimize_crop(img,ratio,SPECS,faces=faces)
    out=OUT/f"image_{ratio.replace(':','x')}.jpg"
    crop=crop_image(img,box,(spec["width"],spec["height"]))
    cv2.imwrite(str(out),crop,[cv2.IMWRITE_JPEG_QUALITY,94])
    report=validate_image(out,spec,expected_faces=len(faces))
    report["crop_box_source_px"]=list(map(int,box))
    report["subjects_detected"]=len(faces)
    (OUT/f"image_{ratio.replace(':','x')}.json").write_text(json.dumps(report,indent=2))
    if report["status"] == "PASS":
        import shutil; shutil.copy2(out, LIB/out.name)
    return out, report

def extract_best_still(video_path):
    cap=cv2.VideoCapture(str(video_path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    duration=n/fps if fps else 0
    # Sample 30 candidates and select the sharpest frame with the most visible faces.
    best=None
    for t in [duration*i/29 for i in range(30)]:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read()
        if not ok: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        faces=len(deduplicate_faces(detect_faces(frame,scale=0.5)))
        score=sharp*(1+0.35*min(faces,2))
        if best is None or score>best[0]: best=(score,frame,t,faces)
    cap.release()
    if best is None: raise RuntimeError("Could not extract still")
    out=OUT/"video_still.jpg"
    faces=deduplicate_faces(detect_faces(best[1]))
    box=optimize_crop(best[1],"9:16",SPECS,faces=faces)
    still=crop_image(best[1],box,(SPECS["9:16"]["width"],SPECS["9:16"]["height"]))
    cv2.imwrite(str(out),still,[cv2.IMWRITE_JPEG_QUALITY,95])
    return out,{"timestamp_s":best[2],"sharpness":best[0],"faces":best[3],"crop_box_source_px":list(map(int,box))}


def best_expected_faces(video_path):
    cap=cv2.VideoCapture(str(video_path)); ok,frame=cap.read(); cap.release()
    if not ok: return 0
    return len(deduplicate_faces(detect_faces(frame,scale=0.5)))

def generate_video(video_path, analysis_fps=3.0):
    video_path=Path(video_path)
    timeline=build_crop_timeline(video_path,analysis_fps=analysis_fps)
    (OUT/"speaker_timeline.json").write_text(json.dumps(timeline,indent=2,default=lambda o: float(o)))
    reel=OUT/"vertical_reel.mp4"
    render_vertical_reel(video_path,reel,timeline["timeline"])
    reel_report=validate_video(reel,SPECS["9:16"],sample_seconds=2.0)
    if reel_report["status"] == "PASS":
        import shutil; shutil.copy2(reel, LIB/reel.name)
    (OUT/"vertical_reel_validation.json").write_text(json.dumps(reel_report,indent=2))
    still,meta=extract_best_still(video_path)
    expected=best_expected_faces(video_path)
    still_report=validate_image(still,SPECS["9:16"],expected_faces=None)
    if still_report["status"] == "PASS":
        import shutil; shutil.copy2(still, LIB/still.name)
    still_report["extraction"]=meta
    (OUT/"video_still_validation.json").write_text(json.dumps(still_report,indent=2))
    return reel,reel_report,still,still_report,timeline
