---
title: PS4 Creative Reformatting Engine
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: "3.10.13"
app_file: app.py
---

# PS4 — Creative Reformatting Engine

A hackathon MVP for subject-aware image adaptation, active-speaker-aware vertical video reframing, still extraction, and machine-readable compliance validation.

## What it demonstrates

- One master image → 16:9, 1:1, 4:5, 9:16
- Detected faces directly influence crop placement
- Multi-face subject grouping rather than fixed center crop
- Video face tracking and time-varying crop position
- Lightweight active-speaker heuristic using speech energy + mouth-region motion + tracking
- Smooth crop movement
- Best-still extraction based on sharpness and face visibility
- JSON platform specification
- Explainable validation report
- Single-variant image regeneration
- Validated outputs are copied into `library/`; failed assets remain outside the library

## Local setup

Use Python 3.11 locally.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Install FFmpeg and make sure `ffmpeg -version` works in a terminal.

Run:

```bash
python app.py
```

Open `http://localhost:7860`.

### Provided challenge assets

- `demo/input_image.png` — supplied master image.
- `demo/input_video_sample.mp4` — 20-second demo clip extracted from the supplied 190-second source video so the public demo remains lightweight.

For the complete source video, select the original file from your local machine in the Video upload component.

## Architecture

```text
MASTER IMAGE
     │
     ▼
YuNet / Haar Face Detection
     │
     ▼
Subject Scoring + Subject Group
     │
     ▼
Crop Optimizer ──> 16:9 / 1:1 / 4:5 / 9:16
     │
     ▼
Validator ──> PASS ──> library/
     │
     └──────> FAIL ──> regenerate

MASTER VIDEO
     │
     ▼
Frame Sampling
     │
     ├── Face Detection + Tracking
     ├── Mouth-region Motion
     └── Audio Energy / Speech Activity
             │
             ▼
       Active Speaker Score
             │
             ▼
       Dynamic Crop Path
             │
             ▼
        Temporal Smoothing
             │
             ▼
          9:16 Reel
             │
             ▼
          Validator
             │
        PASS ─┴─> library/
```

## Third-party references

The implementation is original MVP code, but the design is informed by existing open-source reframing approaches. If you incorporate code from a third-party repository, preserve its license and attribution requirements.

YuNet is an OpenCV Zoo lightweight face detector; the application downloads its MIT-licensed model at first run and falls back to OpenCV Haar detection if the model cannot be downloaded.

## Important MVP scope note

The active-speaker module is intentionally lightweight. It does not claim biometric speaker identification. It uses temporal mouth motion, audio activity, face tracking, and face confidence to choose the active visual speaker. This is enough to demonstrate the PS4 mechanism without adding a large speaker-embedding/ASR stack.
