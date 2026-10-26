"""
mavis - Visual intelligence (Gradio desktop app)

Install:   pip install gradio anthropic pillow
Run:       set ANTHROPIC_API_KEY=your_key   (Windows)
           export ANTHROPIC_API_KEY=your_key (Mac/Linux)
           python app.py
"""

import base64
import inspect
import io
import os

import gradio as gr
from PIL import Image

MODEL = "claude-sonnet-5-5"
DEFAULT_PROMPT = "Explain this image in simple words."

CSS = """
:root, .dark, body, .gradio-container {
    --body-background-fill: #f8f6ff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8f6ff !important;
    --body-text-color: #1f1d2e !important;
    --input-background-fill: #ffffff !important;
    --block-background-fill: transparent !important;
    --border-color-primary: #e3dff3 !important;
}
body { background: #f8f6ff !important; }
.gradio-container {
    width: 100% !important;
    max-width: min(1500px, 96vw) !important;
    margin: 0 auto !important;
    background: #f8f6ff !important;
    color: #1f1d2e !important;
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    padding: clamp(16px, 3vw, 40px) !important;
}
/* ---- alignment: remove Gradio's inner padding so every block shares one left/right edge ---- */
.gradio-container .block, .gradio-container .html-container,
.gradio-container .prose, .gradio-container .form {
    padding: 0 !important; margin: 0 !important; border: none !important;
    box-shadow: none !important; background: transparent !important; }
.gradio-container .gap, .gradio-container .column { gap: 16px !important; }
#header { margin-bottom: 28px !important; }
#subtitle { margin-bottom: 24px !important; }
#dropzone { padding: 0 !important; border: 1.5px dashed #d6d0f0 !important;
            background: #ffffff !important; overflow: hidden; }
#prompt-label { margin: 0 0 4px 0 !important; }
#right-col { display: flex !important; flex-direction: column !important; }
#answer { flex: 1 1 auto; background: #ffffff !important; border: 1px solid #e3dff3 !important;
          border-radius: 14px !important; padding: 16px !important; margin-top: 0 !important; }
#footer-note { padding: 24px 0 0 0 !important; }
#main-row { gap: 40px !important; align-items: stretch; margin: 0 !important; padding: 0 !important; }
#left-col, #right-col { min-width: 300px !important; padding: 0 !important; }
#left-col { flex: 6 1 0 !important; }
#right-col { flex: 5 1 0 !important; }
#dropzone { height: clamp(380px, 58vh, 620px) !important; }
#prompt textarea { min-height: 170px !important; }
#answer { min-height: 120px; }
#header, #eyebrow, #title, #subtitle { margin-left: 0 !important; }
@media (min-width: 1200px) {
    #title { font-size: 34px !important; }
    #subtitle { font-size: 17px !important; }
    #brand-name { font-size: 20px !important; }
    #go { font-size: 17px !important; padding: 18px !important; }
}
#dropzone img { object-fit: contain !important; }
#dropzone, #dropzone * { color: #1f1d2e; }
@media (max-width: 800px) {
    #title { font-size: 22px !important; }
    #main-row { gap: 12px !important; }
}
footer { display: none !important; }

/* header */
#header { display:flex; align-items:center; justify-content:space-between;
          padding-bottom:16px; border-bottom:1px solid #e8e4f5; margin-bottom:28px; }
#brand { display:flex; align-items:center; gap:12px; }
#logo { width:36px; height:36px; border-radius:10px; background:#6355e8;
        display:flex; align-items:center; justify-content:center; }
#brand-name { font-size:18px; font-weight:600; color:#1f1d2e; }
#tagline { font-size:14px; color:#7a7790; }

/* hero */
#eyebrow { font-size:13px; font-weight:600; letter-spacing:.02em; color:#6355e8; text-transform:uppercase; }
#title { font-size:26px; font-weight:600; color:#1f1d2e; margin:10px 0 8px; }
#subtitle { font-size:15px; color:#7a7790; margin-bottom:22px; }

/* drop zone */
#dropzone { background:#ffffff !important; border:1.5px dashed #d6d0f0 !important;
            border-radius:18px !important; min-height:200px; }
#dropzone .wrap, #dropzone .image-container { color:#1f1d2e !important; }
#dropzone .icon-wrap, #dropzone svg { color:#6355e8 !important; }

/* input */
#prompt-label { font-size:14px; font-weight:600; color:#1f1d2e; margin:22px 0 8px; }
#prompt textarea, #prompt input {
    background:#ffffff !important; color:#1f1d2e !important;
    -webkit-text-fill-color:#1f1d2e !important; caret-color:#6355e8 !important;
    border:1px solid #e3dff3 !important;
    border-radius:12px !important; font-size:15px !important; padding:14px 16px !important; }
#prompt textarea::placeholder, #prompt input::placeholder {
    color:#9a97b0 !important; -webkit-text-fill-color:#9a97b0 !important; }
#answer, #answer * { color:#1f1d2e !important; }
#prompt { border:none !important; background:transparent !important; box-shadow:none !important; }

/* button */
#go { background:#6355e8 !important; color:#ffffff !important; border:none !important;
      border-radius:14px !important; font-size:16px !important; font-weight:600 !important;
      padding:16px !important; margin-top:12px; }
#go:hover { background:#5446d6 !important; }

#footer-note { text-align:center; font-size:13px; color:#8b88a0; margin-top:16px; }

/* answer */
#answer { background:#ffffff !important; border:1px solid #e3dff3 !important;
          border-radius:14px !important; padding:16px !important; margin-top:16px; }
"""

HEADER_HTML = """
<div id="header">
  <div id="brand">
    <div id="logo">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>
        <path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>
        <circle cx="12" cy="12" r="3"/><path d="M7 12c1.5-2.5 3.2-3.5 5-3.5s3.5 1 5 3.5"/>
      </svg>
    </div>
    <span id="brand-name">mavis</span>
  </div>
  <span id="tagline">Visual intelligence</span>
</div>
<div id="eyebrow">See beyond the image</div>
<div id="title">Every picture has a story.</div>
<div id="subtitle">Upload an image. Discover what it shows. Ask anything about what you see.</div>
"""


def understand_image(image, question):
    if image is None:
        raise gr.Error("Please upload an image first.")

    question = (question or "").strip() or DEFAULT_PROMPT

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise gr.Error("Set the ANTHROPIC_API_KEY environment variable and restart the app.")

    import anthropic

    img = image.convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": question},
            ],
        }],
    )
    return response.content[0].text


def build_app():
    # `placeholder` on gr.Image only exists in newer Gradio versions
    img_kwargs = dict(
        type="pil",
        sources=["upload"],
        show_label=False,
        height=520,
        elem_id="dropzone",
    )
    if "placeholder" in inspect.signature(gr.Image.__init__).parameters:
        img_kwargs["placeholder"] = "Drop your image here\nor choose a photo from your device\n\nJPG · PNG · WEBP"

    force_light = "() => { document.body.classList.remove('dark'); }"

    with gr.Blocks(css=CSS, title="mavis · Visual intelligence", js=force_light) as demo:
        gr.HTML(HEADER_HTML)

        # Two columns on laptop/desktop, stacks automatically on small screens
        with gr.Row(elem_id="main-row", equal_height=False):
            with gr.Column(scale=1, elem_id="left-col"):
                image = gr.Image(**img_kwargs)

            with gr.Column(scale=1, elem_id="right-col"):
                gr.HTML('<div id="prompt-label" style="margin-top:0">What would you like to discover?</div>')
                question = gr.Textbox(
                    placeholder="e.g. Explain this image in simple words...",
                    show_label=False,
                    lines=3,
                    max_lines=6,
                    elem_id="prompt",
                    container=False,
                )
                go = gr.Button("Understand my image →", elem_id="go")
                answer = gr.Markdown(elem_id="answer")

        gr.HTML('<div id="footer-note">Your image, your questions, your discoveries.</div>')

        go.click(understand_image, [image, question], answer, show_progress="minimal")
        question.submit(understand_image, [image, question], answer, show_progress="minimal")

    return demo


if __name__ == "__main__":
    build_app().launch(inbrowser=True)