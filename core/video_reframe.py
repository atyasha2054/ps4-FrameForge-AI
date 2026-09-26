from __future__ import annotations

from pathlib import Path
import json, math, subprocess, tempfile, wave, contextlib
import cv2
import numpy as np
from .crop_engine import detect_faces, deduplicate_faces


def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    return p


def extract_audio(video, wav_path):
    run_cmd(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)])


def audio_energy_timeline(wav_path, hop=0.25):
    try:
        with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
            rate = wf.getframerate(); data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        win = max(1, int(rate * hop))
        out = []
        for i in range(0, len(samples), win):
            x = samples[i:i+win]
            if len(x) < win * 0.5: break
            rms = float(np.sqrt(np.mean(x*x) + 1e-9))
            out.append(rms)
        if not out: return []
        a = np.array(out); lo, hi = np.percentile(a, [10, 90])
        a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        return a.tolist()
    except Exception:
        return []


def bbox_iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2 = ax1+aw, ay1+ah; bx2, by2 = bx1+bw, by1+bh
    ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih; union = aw*ah+bw*bh-inter
    return inter/union if union else 0


def track_faces(prev, current, max_dist=220):
    """Greedy nearest-neighbour association, enough for a short two-person demo."""
    if not prev: return current
    used=set(); result=[]
    for f in current:
        best=-1; best_cost=1e9
        for i,p in enumerate(prev):
            if i in used: continue
            dist=math.hypot(f.cx-p.cx, f.cy-p.cy)
            cost=dist - 180*bbox_iou((f.x,f.y,f.w,f.h),(p.x,p.y,p.w,p.h))
            if cost<best_cost and dist<max_dist:
                best_cost=cost; best=i
        if best>=0:
            used.add(best)
        result.append(f)
    return result


def mouth_motion(prev_gray, gray, face):
    x1=max(0, int(face.x+0.15*face.w)); x2=min(gray.shape[1], int(face.x+0.85*face.w))
    y1=max(0, int(face.y+0.50*face.h)); y2=min(gray.shape[0], int(face.y+0.98*face.h))
    if x2<=x1 or y2<=y1: return 0.0
    a=prev_gray[y1:y2,x1:x2]; b=gray[y1:y2,x1:x2]
    if a.size==0 or b.size==0: return 0.0
    return float(np.mean(cv2.absdiff(a,b))/255.0)


def smooth_path(points, alpha=0.18):
    if not points: return []
    out=[]; x,y=points[0]
    for tx,ty in points:
        x = (1-alpha)*x + alpha*tx
        y = (1-alpha)*y + alpha*ty
        out.append((x,y))
    return out


def build_crop_timeline(video, analysis_fps=3.0):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError("Cannot open video")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration=frame_count/fps
    step=max(1,int(round(fps/analysis_fps)))
    frames=[]; prev_gray=None; prev_faces=[]
    frame_idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if frame_idx % step != 0:
            frame_idx+=1; continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=deduplicate_faces(detect_faces(frame, scale=0.5))
        faces=track_faces(prev_faces, faces)
        motions=[mouth_motion(prev_gray, gray, f) if prev_gray is not None else 0 for f in faces]
        frames.append({"t":frame_idx/fps,"faces":faces,"motions":motions,"width":frame.shape[1],"height":frame.shape[0]})
        prev_faces=faces; prev_gray=gray; frame_idx+=1
    cap.release()

    # Audio activity sampled every 0.25s, interpolated at frame-analysis timestamps.
    with tempfile.TemporaryDirectory() as td:
        wav=Path(td)/"audio.wav"
        try: extract_audio(video,wav); audio=audio_energy_timeline(wav,0.25)
        except Exception: audio=[]
    timeline=[]
    for item in frames:
        faces=item["faces"]; motions=item["motions"]; t=item["t"]; iw=item["width"]; ih=item["height"]
        if not faces:
            timeline.append({"t":t,"speaker":-1,"cx":0.5,"cy":0.5,"faces":0}); continue
        ai=int(t/0.25); speech=audio[min(ai,len(audio)-1)] if audio else 0.5
        scores=[]
        for f,m in zip(faces,motions):
            face_size=min(1.0,(f.w*f.h)/(1920*1080*0.03))
            scores.append(0.65*m + 0.25*speech + 0.10*face_size)
        # If there is little mouth activity, retain previous speaker where possible.
        best=int(np.argmax(scores))
        target=faces[best]
        timeline.append({"t":t,"speaker":best,"cx":target.cx/iw,"cy":target.cy/ih,"faces":len(faces),"score":round(float(scores[best]),4)})
    return {"fps":fps,"duration":duration,"analysis_fps":analysis_fps,"timeline":timeline}


def render_vertical_reel(video, output, timeline, out_w=1080, out_h=1920):
    """Render a per-frame dynamic crop using OpenCV, then mux original audio with FFmpeg."""
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    in_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); in_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    crop_w=int(round(in_h*out_w/out_h)); crop_h=in_h
    temp=Path(output).with_suffix(".silent.mp4")
    writer=cv2.VideoWriter(str(temp),fourcc,fps,(crop_w,crop_h))
    times=np.array([x["t"] for x in timeline],dtype=float) if timeline else np.array([0.0])
    xs=np.array([x["cx"] for x in timeline],dtype=float) if timeline else np.array([0.5])
    ys=np.array([x["cy"] for x in timeline],dtype=float) if timeline else np.array([0.5])
    sx=smooth_path(list(zip(xs,ys)),alpha=0.12)
    sx=np.array(sx)
    idx=0; frame_no=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        t=frame_no/fps
        while idx+1<len(times) and times[idx+1] <= t: idx+=1
        cx=float(sx[idx,0]*in_w); cy=float(sx[idx,1]*in_h)
        x1=int(np.clip(cx-crop_w/2,0,in_w-crop_w)); y1=int(np.clip(cy-crop_h/2,0,in_h-crop_h))
        crop=frame[y1:y1+crop_h,x1:x1+crop_w]
        writer.write(crop); frame_no+=1
    cap.release(); writer.release()
    run_cmd(["ffmpeg","-y","-i",str(temp),"-i",str(video),"-map","0:v:0","-map","1:a?","-vf",f"scale={out_w}:{out_h}:flags=fast_bilinear","-c:v","libx264","-preset","ultrafast","-crf","26","-c:a","aac","-shortest",str(output)])
    temp.unlink(missing_ok=True)
