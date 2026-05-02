import gradio as gr
from PIL import Image
from predict import predict, _load_inference_models, _load_vision_models


def analyse(img):
    if img is None:
        return "## No image uploaded", "", _empty_bar()

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    try:
        result = predict(img)
    except FileNotFoundError as e:
        return f"## Error\n{e}", "", _empty_bar()
    except Exception as e:
        return f"## Error during inference\n{e}", "", _empty_bar()

    label      = result["label"]
    prob       = result["probability"]
    confidence = result["confidence"]
    is_ai      = label == "AI-Generated"

    colour  = "#e74c3c" if is_ai else "#27ae60"
    icon    = "🤖" if is_ai else "📷"
    verdict = (
        f"<div style='text-align:center; padding:16px; "
        f"border-radius:10px; background:{colour}22; border:2px solid {colour}'>"
        f"<h2 style='color:{colour}; margin:0'>{icon} {label}</h2>"
        f"<p style='font-size:1.1em; margin:6px 0'>Confidence: <b>{confidence*100:.1f}%</b></p>"
        f"<p style='color:#888; margin:0'>AI probability: {prob*100:.1f}% "
        f"(threshold {result['threshold']:.3f})</p>"
        f"</div>"
    )

    details = (
        f"LogReg probability : {result['lr_prob']:.4f}\n"
        f"MLP×5  probability : {result['mlp_prob']:.4f}\n"
        f"Ensemble probability: {prob:.4f}\n"
        f"Decision threshold  : {result['threshold']:.3f}"
    )

    return verdict, details, _prob_bar(prob)


def _empty_bar():
    return "<div style='height:30px'></div>"


def _prob_bar(prob):
    pct  = prob * 100
    ai_w = f"{pct:.1f}%"
    rl_w = f"{100 - pct:.1f}%"
    return (
        "<div style='font-size:0.85em; margin-top:8px'>"
        "<div style='display:flex; height:28px; border-radius:6px; overflow:hidden'>"
        f"<div style='width:{rl_w}; background:#27ae60; display:flex; align-items:center;"
        f" justify-content:center; color:white; font-weight:bold'>Authentic {rl_w}</div>"
        f"<div style='width:{ai_w}; background:#e74c3c; display:flex; align-items:center;"
        f" justify-content:center; color:white; font-weight:bold'>AI {ai_w}</div>"
        "</div></div>"
    )


print("Loading inference models...")
try:
    _load_inference_models()
    print("Loading vision models (CLIP + DINOv2)...")
    _load_vision_models()
    print("Ready.")
except FileNotFoundError:
    print("WARNING: No trained models found. Run  python train.py  first.")


with gr.Blocks(
    title="AI Image Detector",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 860px !important }",
) as demo:

    gr.Markdown(
        """
        # AI Image Detector
        Upload an image to classify it as AI-generated or authentic.
        Uses CLIP + DINOv2 embeddings combined with forensic signals (NPR, FFT, ELA, PRNU).
        Requires trained models — see `README.md` for setup.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload image", height=320)
            btn       = gr.Button("Analyse", variant="primary", size="lg")
        with gr.Column(scale=1):
            verdict_out = gr.HTML(label="Verdict")
            bar_out     = gr.HTML()
            details_out = gr.Textbox(label="Model internals", lines=5,
                                     interactive=False, show_copy_button=True)

    gr.Examples(examples=[], inputs=img_input)

    btn.click(fn=analyse, inputs=img_input, outputs=[verdict_out, details_out, bar_out])
    img_input.upload(fn=analyse, inputs=img_input, outputs=[verdict_out, details_out, bar_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
