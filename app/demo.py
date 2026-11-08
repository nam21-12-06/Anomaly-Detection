"""
ANOMALY-VISION: Industrial Visual Anomaly Detection & Defect Localization Web Demo.

Interactive web application powered by Gradio for real-time visual inspection:
- Compares SOTA PatchCore vs. Convolutional Autoencoder baseline.
- Supports industrial Object ('bottle') and Texture ('carpet') categories.
- Produces pixel-level defect localization heatmaps and automatic PASS/DEFECT quality decisions.

Usage:
    python app/demo.py
"""

from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ConvAutoencoder, PatchCore
from src.utils import overlay_mask_on_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model cache to avoid re-loading weights on every query
MODEL_CACHE: Dict[str, torch.nn.Module] = {}

# Default decision thresholds calibrated on MVTec test sets
DEFAULT_THRESHOLDS = {
    ("bottle", "PatchCore"): 1.80,
    ("bottle", "Autoencoder"): 0.08,
    ("carpet", "PatchCore"): 1.60,
    ("carpet", "Autoencoder"): 0.08,
}

# Image preprocessing transform
PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def load_patchcore_model(category: str) -> Optional[PatchCore]:
    """Loads and caches PatchCore model for a specified category."""
    cache_key = f"patchcore_{category}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    ckpt_path = PROJECT_ROOT / "checkpoints" / "patchcore" / f"{category}_resnet18_c0.01.pt"
    if not ckpt_path.exists():
        return None

    model = PatchCore(
        backbone="resnet18",
        pretrained=True,
        layers=("layer2", "layer3"),
        patch_pooling=True,
        pooling_kernel=3,
        coreset_sampling_ratio=0.01,
        n_neighbors=9,
        image_size=(224, 224),
        sigma=4.0,
        device=DEVICE,
    )
    model.load(ckpt_path)
    model.eval()
    MODEL_CACHE[cache_key] = model
    return model


def load_autoencoder_model(category: str) -> Optional[ConvAutoencoder]:
    """Loads and caches Autoencoder baseline model."""
    cache_key = f"ae_{category}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{category}_ae" / "best.pth"
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    latent_dim = ckpt.get("latent_dim", 128)
    model = ConvAutoencoder(in_channels=3, base_channels=32, latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    MODEL_CACHE[cache_key] = model
    return model


def inspect_image(
    input_image: Optional[np.ndarray],
    category: str,
    model_choice: str,
    threshold: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str, str, str]:
    """
    Runs real-time anomaly detection inference on an input image.

    Returns:
        Tuple of:
            - processed_image: Preprocessed original image (H, W, 3) in [0, 1]
            - heatmap_rgb: Colored jet heatmap of pixel anomaly scores
            - overlay_rgb: Original image with anomaly heatmap blended on top
            - status_html: HTML banner showing PASS or DEFECT with styling
            - score_text: Formatted string of anomaly score
            - latency_text: Formatted string of inference latency in ms
    """
    if input_image is None:
        return (
            None,
            None,
            None,
            "<div style='color: orange; font-size: 16px;'>⚠️ Vui lòng tải lên một ảnh để kiểm tra.</div>",
            "N/A",
            "N/A",
        )

    t0 = time.time()

    # Convert NumPy array to PIL Image
    pil_img = Image.fromarray(input_image).convert("RGB")
    tensor_raw = PREPROCESS_TRANSFORM(pil_img)  # Shape (3, 224, 224) in [0, 1]
    display_img = tensor_raw.permute(1, 2, 0).numpy()

    if model_choice == "PatchCore (SOTA)":
        model = load_patchcore_model(category)
        if model is None:
            return (
                display_img,
                None,
                None,
                f"<div style='color: red;'>❌ Không tìm thấy checkpoint PatchCore cho '{category}'.</div>",
                "N/A",
                "N/A",
            )
        # PatchCore requires ImageNet normalized tensor
        tensor_norm = IMAGENET_NORMALIZE(tensor_raw).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            anomaly_map_tensor, image_score_tensor = model.predict(tensor_norm, reweight=True)

        anom_map = anomaly_map_tensor.squeeze().cpu().numpy()
        anomaly_score = float(image_score_tensor.item())

    elif model_choice == "Autoencoder (Baseline)":
        model = load_autoencoder_model(category)
        if model is None:
            return (
                display_img,
                None,
                None,
                f"<div style='color: red;'>❌ Không tìm thấy checkpoint Autoencoder cho '{category}'.</div>",
                "N/A",
                "N/A",
            )
        tensor_in = tensor_raw.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            anomaly_map_tensor = model.get_anomaly_map(tensor_in)
            image_score_tensor = model.get_anomaly_score(tensor_in)

        anom_map = anomaly_map_tensor.squeeze().cpu().numpy()
        anomaly_score = float(image_score_tensor.item())

    else:
        return display_img, None, None, "<div>Mô hình không hỗ trợ.</div>", "N/A", "N/A"

    elapsed_ms = (time.time() - t0) * 1000.0

    # Calibrated heatmap normalization and defect localization
    if model_choice == "PatchCore (SOTA)":
        v_min = 1.0
        v_max = max(float(anom_map.max()), threshold * 1.5)
        norm_map = np.clip((anom_map - v_min) / (v_max - v_min + 1e-8), 0.0, 1.0)
        defect_mask = (anom_map > threshold).astype(np.float32)
    else:
        v_min = float(anom_map.min())
        v_max = max(float(anom_map.max()), threshold * 1.5)
        norm_map = np.clip((anom_map - v_min) / (v_max - v_min + 1e-8), 0.0, 1.0)
        defect_mask = (anom_map > threshold).astype(np.float32)

    # Colorize with JET colormap
    heatmap_colored = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Create overlay on top of original image (highlighting defects only)
    overlay_rgb = overlay_mask_on_image(display_img, defect_mask, alpha=0.5, color=(1.0, 0.0, 0.0))

    # Decision logic based on threshold
    is_anomaly = anomaly_score > threshold
    if is_anomaly:
        status_html = f"""
        <div style="background-color: #ffeef0; border: 2px solid #cf222e; border-radius: 8px; padding: 12px; text-align: center;">
            <h2 style="color: #cf222e; margin: 0; font-weight: 800; font-size: 22px;">⚠️ DEFECT DETECTED (NG)</h2>
            <p style="color: #57606a; margin: 4px 0 0 0; font-size: 13px;">Sản phẩm không đạt chuẩn chất lượng (Vượt ngưỡng {threshold:.2f})</p>
        </div>
        """
    else:
        status_html = f"""
        <div style="background-color: #dafbe1; border: 2px solid #1a7f37; border-radius: 8px; padding: 12px; text-align: center;">
            <h2 style="color: #1a7f37; margin: 0; font-weight: 800; font-size: 22px;">✅ PASSED (OK)</h2>
            <p style="color: #57606a; margin: 4px 0 0 0; font-size: 13px;">Sản phẩm đạt chuẩn bình thường (Dưới ngưỡng {threshold:.2f})</p>
        </div>
        """

    score_text = f"{anomaly_score:.4f}"
    latency_text = f"{elapsed_ms:.1f} ms ({DEVICE.type.upper()})"

    return (
        np.clip(display_img, 0.0, 1.0),
        np.clip(heatmap_rgb, 0.0, 1.0),
        np.clip(overlay_rgb, 0.0, 1.0),
        status_html,
        score_text,
        latency_text,
    )


def update_threshold_default(category: str, model_choice: str):
    """Updates the threshold slider when category or model selection changes."""
    model_name = "PatchCore" if "PatchCore" in model_choice else "Autoencoder"
    if model_name == "PatchCore":
        val = DEFAULT_THRESHOLDS.get((category, "PatchCore"), 1.80)
        return gr.update(minimum=0.5, maximum=5.0, value=val, step=0.05)
    else:
        val = DEFAULT_THRESHOLDS.get((category, "Autoencoder"), 0.08)
        return gr.update(minimum=0.01, maximum=0.50, value=val, step=0.01)


# Build interactive UI with Gradio Blocks
def create_ui():
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: auto !important; }
    .header-box { text-align: center; margin-bottom: 20px; }
    """

    with gr.Blocks(title="ANOMALY-VISION: Industrial Visual Defect Detection") as demo:
        with gr.Row(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🔍 ANOMALY-VISION
                ### Hệ Thống Kiểm Tra & Định Vị Khuyết Tật Sản Phẩm Công Nghiệp Tự Động
                *Học biểu diễn từ ảnh bình thường (One-Class Visual Representation Learning) trên chuẩn MVTec AD.*
                """
            )

        with gr.Row():
            # Left Panel: Input & Settings
            with gr.Column(scale=4):
                gr.Markdown("#### 📥 1. Chọn Dữ Liệu & Mô Hình")
                input_image = gr.Image(label="Ảnh sản phẩm cần kiểm tra", type="numpy")

                with gr.Row():
                    category_dropdown = gr.Dropdown(
                        label="Danh mục sản phẩm",
                        choices=["bottle", "carpet"],
                        value="bottle",
                    )
                    model_dropdown = gr.Dropdown(
                        label="Mô hình kiểm tra",
                        choices=["PatchCore (SOTA)", "Autoencoder (Baseline)"],
                        value="PatchCore (SOTA)",
                    )

                threshold_slider = gr.Slider(
                    label="Ngưỡng phân loại lỗi (Anomaly Threshold)",
                    minimum=0.5,
                    maximum=5.0,
                    value=1.80,
                    step=0.05,
                    info="Điểm bất thường vượt quá ngưỡng này sẽ bị kết luận là DEFECT",
                )

                inspect_button = gr.Button("⚡ Bắt Đầu Kiểm Tra (Inspect)", variant="primary", size="lg")

            # Right Panel: Results & Localization
            with gr.Column(scale=6):
                gr.Markdown("#### 📊 2. Kết Quả Kiểm Định Chất Lượng")
                status_output = gr.HTML(
                    """
                    <div style="background-color: #f6f8fa; border: 1px dashed #d0d7de; border-radius: 8px; padding: 16px; text-align: center;">
                        <span style="color: #57606a;">Nhấn 'Bắt Đầu Kiểm Tra' để xem kết luận chất lượng</span>
                    </div>
                    """
                )

                with gr.Row():
                    score_metric = gr.Textbox(label="Điểm Bất Thường (Anomaly Score)", interactive=False)
                    latency_metric = gr.Textbox(label="Thời Gian Xử Lý (Latency)", interactive=False)

                gr.Markdown("#### 🗺️ 3. Định Vị Vùng Khuyết Tật (Pixel-Level Heatmap)")
                with gr.Row():
                    out_orig = gr.Image(label="Ảnh Gốc (224x224)", interactive=False)
                    out_heatmap = gr.Image(label="Heatmap Điểm Lỗi", interactive=False)
                    out_overlay = gr.Image(label="Overlay Vùng Khuyết Tật", interactive=False)

        # Pre-loaded Example Gallery
        gr.Markdown("---")
        gr.Markdown("#### 📁 Ảnh Mẫu Thử Nghiệm Nhanh (Click để nạp ảnh)")
        example_samples = [
            ["data/bottle/test/good/000.png", "bottle", "PatchCore (SOTA)", 1.80],
            ["data/bottle/test/broken_large/000.png", "bottle", "PatchCore (SOTA)", 1.80],
            ["data/bottle/test/contamination/000.png", "bottle", "PatchCore (SOTA)", 1.80],
            ["data/carpet/test/good/000.png", "carpet", "PatchCore (SOTA)", 1.60],
            ["data/carpet/test/hole/000.png", "carpet", "PatchCore (SOTA)", 1.60],
            ["data/carpet/test/thread/000.png", "carpet", "PatchCore (SOTA)", 1.60],
        ]

        # Filter only existing sample paths
        valid_examples = [ex for ex in example_samples if Path(ex[0]).exists()]
        if valid_examples:
            gr.Examples(
                examples=valid_examples,
                inputs=[input_image, category_dropdown, model_dropdown, threshold_slider],
                outputs=[out_orig, out_heatmap, out_overlay, status_output, score_metric, latency_metric],
                fn=inspect_image,
                cache_examples=False,
            )

        # Wire callbacks
        inspect_button.click(
            fn=inspect_image,
            inputs=[input_image, category_dropdown, model_dropdown, threshold_slider],
            outputs=[out_orig, out_heatmap, out_overlay, status_output, score_metric, latency_metric],
        )

        category_dropdown.change(
            fn=update_threshold_default,
            inputs=[category_dropdown, model_dropdown],
            outputs=[threshold_slider],
        )
        model_dropdown.change(
            fn=update_threshold_default,
            inputs=[category_dropdown, model_dropdown],
            outputs=[threshold_slider],
        )

    return demo


if __name__ == "__main__":
    demo_app = create_ui()
    print("Launching ANOMALY-VISION Gradio Demo on http://127.0.0.1:7860 ...")
    demo_app.launch(server_name="127.0.0.1", server_port=7860, share=False)
