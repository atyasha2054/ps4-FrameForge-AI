from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from core.pipeline import (
    generate_image_variants,
    regenerate_image_variant,
    generate_video,
    OUT,
)

# PATHS
ROOT = Path(__file__).resolve().parent
DEMO_IMAGE = ROOT / "demo/input_image.png"
DEMO_VIDEO = ROOT / "demo/input_video_sample.mp4"


# BACKEND HELPERS
def clear_outputs():
    """Remove generated files from the output directory."""
    for p in OUT.glob("*"):
        if p.is_file():
            p.unlink()


def image_job(image, selected):
    """Generate selected image aspect-ratio variants."""
    if image is None:
        raise gr.Error("Please upload a master image first.")

    clear_outputs()

    ratios = selected or ["16:9", "1:1", "4:5", "9:16"]

    results = generate_image_variants(image, ratios)

    gallery = [str(p) for _, p, _ in results]
    report = {r: rep for r, _, rep in results}

    report_path = OUT / "image_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    return (
        gallery,
        json.dumps(report, indent=2),
        str(report_path),
    )


def regenerate_job(image, ratio):
    """Regenerate only one selected aspect ratio."""
    if image is None:
        raise gr.Error("Please upload a master image first.")

    out, report = regenerate_image_variant(image, ratio)

    report_path = (
        OUT / f"regenerated_{ratio.replace(':', 'x')}_validation.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    return (
        str(out),
        json.dumps(report, indent=2),
        str(report_path),
    )


def video_job(video, analysis_fps):
    """Generate speaker-aware vertical reel and best still."""
    if video is None:
        raise gr.Error("Please upload a master video first.")

    clear_outputs()

    reel, reel_report, still, still_report, timeline = generate_video(
        video,
        float(analysis_fps or 3),
    )

    speaker_switches = sum(
        1
        for a, b in zip(
            timeline["timeline"],
            timeline["timeline"][1:]
        )
        if a.get("speaker") != b.get("speaker")
    )

    report = {
        "vertical_reel": reel_report,
        "still": still_report,
        "speaker_switches": speaker_switches,
    }

    report_path = OUT / "video_validation_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    return (
        str(reel),
        str(still),
        json.dumps(report, indent=2),
        str(report_path),
        str(ROOT / "outputs/speaker_timeline.json"),
    )


def load_demo():
    if not DEMO_IMAGE.exists():
        raise gr.Error(f"Demo image not found: {DEMO_IMAGE}")

    if not DEMO_VIDEO.exists():
        raise gr.Error(f"Demo video not found: {DEMO_VIDEO}")

    return str(DEMO_IMAGE.resolve()), str(DEMO_VIDEO.resolve())


# CUSTOM CSS
CUSTOM_CSS = """
/*GLOBAL*/

/* Main background */
body {
    background: #0b0d10 !important;
}

/*HEADER*/

.ff-header {
    background: linear-gradient(
        135deg,
        #11151b 0%,
        #0e1116 100%
    );
    border: 1px solid #252b34;
    border-radius: 18px;
    padding: 34px 38px;
    margin-bottom: 24px;
}

.ff-brand {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #9ca8b8;
    margin-bottom: 8px;
}

.ff-title {
    font-size: 42px;
    line-height: 1.08;
    font-weight: 700;
    letter-spacing: -1.5px;
    color: #f5f7fa;
    margin: 0;
}

.ff-subtitle {
    color: #aeb7c4;
    font-size: 16px;
    line-height: 1.6;
    margin-top: 12px;
    max-width: 760px;
}

.ff-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 20px;
    padding: 7px 13px;
    border-radius: 999px;
    background: #151a20;
    border: 1px solid #29313b;
    color: #b8c2ce;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.6px;
}

.ff-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #7dd3a7;
    display: inline-block;
}

/* SECTION HEADERS */

.ff-section {
    margin-top: 28px;
    margin-bottom: 12px;
}

.ff-section-title {
    color: #f1f3f6;
    font-size: 23px;
    font-weight: 650;
    letter-spacing: -0.4px;
}

.ff-section-description {
    color: #8f9aaa;
    font-size: 14px;
    margin-top: 5px;
}

/* CARDS */

.ff-card {
    background: #101318;
    border: 1px solid #252b34;
    border-radius: 16px;
    padding: 20px;
}

.ff-card-title {
    color: #dfe4ea;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 6px;
}

/* IMAGE / VIDEO UPLOAD AREAS */

.ff-upload-area {
    border: 1px dashed #39424e !important;
    border-radius: 14px !important;
    background: #0d1014 !important;
}

.ff-upload-area:hover {
    border-color: #687585 !important;
}

/* MASTER IMAGE PREVIEW */

/* Uploaded master image */
[data-testid="image"] img {
    object-fit: contain !important;
    max-width: 100% !important;
    max-height: 100% !important;
}

/* BUTTONS */

.ff-primary button {
    background: #e7ebef !important;
    color: #101318 !important;
    border: none !important;
    font-weight: 650 !important;
    min-height: 46px !important;
    border-radius: 10px !important;
}

.ff-primary button:hover {
    background: #ffffff !important;
}

.ff-secondary button {
    background: #171b21 !important;
    color: #dce2e8 !important;
    border: 1px solid #303741 !important;
    border-radius: 10px !important;
    min-height: 43px !important;
}

.ff-secondary button:hover {
    background: #20252c !important;
}

/* FORMAT PILLS */

.ff-format-info {
    background: #0e1116;
    border: 1px solid #242b34;
    border-radius: 12px;
    padding: 14px 17px;
    color: #9ca7b4;
    font-size: 13px;
    line-height: 1.6;
}

/* OUTPUT AREA */

.ff-output {
    background: #0e1116;
    border: 1px solid #252c35;
    border-radius: 14px;
    padding: 8px;
}

/* VALIDATION */

.ff-validation {
    background: #0c0f13;
    border: 1px solid #252c35;
    border-radius: 14px;
}

.ff-pass {
    color: #8ed5ae;
}

.ff-note {
    color: #8995a4;
    font-size: 13px;
    line-height: 1.65;
}

/* FEATURE STRIP */

.ff-feature {
    background: #101318;
    border: 1px solid #252b34;
    border-radius: 14px;
    padding: 17px 19px;
    height: 100%;
}

.ff-feature-title {
    color: #e2e7ec;
    font-weight: 600;
    font-size: 14px;
}

.ff-feature-text {
    color: #8994a2;
    font-size: 12px;
    line-height: 1.5;
    margin-top: 6px;
}

/* FOOTER */

.ff-footer {
    text-align: center;
    color: #697482;
    font-size: 12px;
    padding: 25px 0 10px;
    border-top: 1px solid #20252c;
    margin-top: 35px;
}

.ff-highlight {
    color: #b9c3ce;
    font-weight: 600;
}

/* HIDE SOME DEFAULT GRADIO VISUAL NOISE */

footer {
    display: none !important;
}
"""

# UI
with gr.Blocks(
    title="FrameForge AI",
    theme=gr.themes.Base(
        primary_hue="slate",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
    ),
    css=CUSTOM_CSS,
) as demo:

    # HEADER
    gr.HTML(
        """
        <div class="ff-header">

            <div class="ff-brand">
                FRAMEFORGE AI
            </div>

            <div class="ff-title">
                Intelligent Media Reformatting
            </div>

            <div class="ff-subtitle">
                Transform one master image or video into
                platform-ready content using subject-aware
                framing, speaker-aware video reformatting,
                and automated compliance validation.
            </div>

            <div class="ff-status">
                <span class="ff-dot"></span>
                AI MEDIA PIPELINE · READY
            </div>

        </div>
        """
    )

    # MASTER ASSET
    gr.HTML(
        """
        <div class="ff-section">
            <div class="ff-section-title">
                Master Asset
            </div>
            <div class="ff-section-description">
                Upload the original media you want to adapt.
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):

            image = gr.Image(
                type="filepath",
                label="Master Image",
                sources=["upload", "clipboard"],
                height=270,
                show_label=True,
                container=True,
            )

        with gr.Column(scale=1):

            video = gr.Video(
                label="Master Video",
                height=270,
            )

    with gr.Row():

        load = gr.Button(
            "Load Provided Demo Assets",
            elem_classes=["ff-secondary"],
        )

        clear = gr.Button(
            "Clear Workspace",
            elem_classes=["ff-secondary"],
        )

    load.click(
        load_demo,
        outputs=[image, video],
    )

    clear.click(
        lambda: (None, None),
        outputs=[image, video],
    )

    # IMAGE PIPELINE
    gr.HTML(
        """
        <div class="ff-section">
            <div class="ff-section-title">
                Image Reformatting
            </div>
            <div class="ff-section-description">
                Generate composition-aware variants for
                multiple platform aspect ratios.
            </div>
        </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=2):

            ratios = gr.CheckboxGroup(
                [
                    "16:9",
                    "1:1",
                    "4:5",
                    "9:16",
                ],
                value=[
                    "16:9",
                    "1:1",
                    "4:5",
                    "9:16",
                ],
                label="Target Formats",
            )

        with gr.Column(scale=1):

            gr.HTML(
                """
                <div class="ff-format-info">
                    <b>Smart Crop</b><br>
                    Subject position influences the crop
                    instead of using a fixed center crop.
                </div>
                """
            )

    ib = gr.Button(
        "Generate Image Variants",
        elem_classes=["ff-primary"],
    )

    gr.HTML(
        """
        <div class="ff-section">
            <div class="ff-section-title">
                Generated Assets
            </div>
        </div>
        """
    )

    ig = gr.Gallery(
        label="Platform Variants",
        columns=4,
        height="auto",
        elem_classes=["ff-output"],
    )

    ir = gr.Code(
        label="Machine-Readable Validation Report",
        language="json",
        elem_classes=["ff-validation"],
    )

    idl = gr.File(
        label="Download Validation Report",
    )

    ib.click(
        image_job,
        inputs=[image, ratios],
        outputs=[ig, ir, idl],
    )

    # SINGLE VARIANT REGENERATION
    gr.HTML(
        """
        <div class="ff-section">
            <div class="ff-section-title">
                Regenerate a Single Variant
            </div>
            <div class="ff-section-description">
                Fix one problematic format without regenerating
                the entire asset set.
            </div>
        </div>
        """
    )

    with gr.Row():

        regen_ratio = gr.Dropdown(
            [
                "16:9",
                "1:1",
                "4:5",
                "9:16",
            ],
            value="9:16",
            label="Variant to Regenerate",
        )

        rb = gr.Button(
            "Regenerate Selected Variant",
            elem_classes=["ff-secondary"],
        )

    with gr.Row():

        with gr.Column():
            ro = gr.Image(
                label="Regenerated Asset",
                elem_classes=["ff-output"],
            )

        with gr.Column():
            rr = gr.Code(
                label="Regeneration Validation",
                language="json",
                elem_classes=["ff-validation"],
            )

    rj = gr.File(
        label="Download Regeneration Report",
    )

    rb.click(
        regenerate_job,
        inputs=[image, regen_ratio],
        outputs=[ro, rr, rj],
    )

    # VIDEO PIPELINE
    gr.HTML(
        """
        <div class="ff-section">
            <div class="ff-section-title">
                Video Reframing
            </div>
            <div class="ff-section-description">
                Create a vertical 9:16 reel that follows the
                active speaker and extracts a representative still.
            </div>
        </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=2):

            afps = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Analysis FPS",
                info="Lower values process faster; higher values provide denser analysis.",
            )

        with gr.Column(scale=1):

            gr.HTML(
                """
                <div class="ff-format-info">
                    <b>Speaker-Aware Reframing</b><br>
                    The crop can move between detected speakers
                    rather than remaining fixed.
                </div>
                """
            )

    vb = gr.Button(
        "Generate Speaker-Aware Reel",
        elem_classes=["ff-primary"],
    )

    with gr.Row():

        with gr.Column():
            vo = gr.Video(
                label="Vertical 9:16 Reel",
                elem_classes=["ff-output"],
            )

        with gr.Column():
            so = gr.Image(
                label="Best Still",
                elem_classes=["ff-output"],
            )

    vr = gr.Code(
        label="Video Validation Report",
        language="json",
        elem_classes=["ff-validation"],
    )

    with gr.Row():

        vjson = gr.File(
            label="Download Video Validation JSON",
        )

        timeline = gr.File(
            label="Download Speaker / Crop Timeline",
        )

    vb.click(
        video_job,
        inputs=[video, afps],
        outputs=[
            vo,
            so,
            vr,
            vjson,
            timeline,
        ],
    )

    # SYSTEM CAPABILITIES
    gr.HTML(
        """
        <div class="ff-section">
            <div class="ff-section-title">
                Pipeline Intelligence
            </div>
            <div class="ff-section-description">
                FrameForge validates generated content before
                it becomes part of the asset library.
            </div>
        </div>
        """
    )

    with gr.Row():

        gr.HTML(
            """
            <div class="ff-feature">
                <div class="ff-feature-title">
                    ◈ Subject-Aware Cropping
                </div>
                <div class="ff-feature-text">
                    Detects important visual subjects and uses
                    their position to determine crop placement.
                </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="ff-feature">
                <div class="ff-feature-title">
                    ◇ Speaker-Aware Reframing
                </div>
                <div class="ff-feature-text">
                    Tracks speaker activity and dynamically
                    reframes vertical video content.
                </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="ff-feature">
                <div class="ff-feature-title">
                    ✓ Compliance Validation
                </div>
                <div class="ff-feature-text">
                    Checks dimensions, aspect ratio, face
                    visibility, safe zones and file integrity.
                </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="ff-feature">
                <div class="ff-feature-title">
                    ↻ Targeted Regeneration
                </div>
                <div class="ff-feature-text">
                    Regenerate only the format that fails
                    validation or requires a better composition.
                </div>
            </div>
            """
        )

     # VALIDATION_NOTE
    gr.HTML(
        """
        <div class="ff-card" style="margin-top:24px;">
            <div class="ff-card-title">
                Validation Policy
            </div>

            <div class="ff-note">
                Every generated asset is evaluated against
                technical and visual constraints before being
                considered ready for the media library.
                Failed assets should be regenerated rather than
                silently accepted.
            </div>
        </div>
        """
    )

    # FOOTER
    gr.HTML(
        """
        <div class="ff-footer">
            <span class="ff-highlight">FrameForge AI</span>
            &nbsp;·&nbsp;
            Intelligent Media Reformatting & Reframing Engine
            &nbsp;·&nbsp;
            PS4 Creative Reformatting
        </div>
        """
    )


# RUN APPLICATION
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )