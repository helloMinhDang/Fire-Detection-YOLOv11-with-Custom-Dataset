import gradio as gr
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import tempfile
import os

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('runs/detect/train/weights/best.pt')

def resize_image(image, target_height=400):
    h, w, _ = image.shape
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height))

def detect_image(image_path):
    if image_path is None:
        return "Please upload or capture an image."
    image = cv2.imread(image_path)
    if image is None:
        return "Invalid image file."
    results = model.predict(source=image, conf=0.4, device=device)
    img_bgr = results[0].plot()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def detect_video(video_source):
    if video_source is None:
        return "Please upload a video or record from webcam."
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return "Cannot open video file."

    # Lấy thông tin video gốc
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo file tạm cho video output
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(
        temp_file.name,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.4, device=device)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    
    return temp_file.name  # Trả về đường dẫn video đã xử lý

with gr.Blocks() as demo:
    gr.Markdown(
        "# 🔥 Fire Detection Demo\nDetect fire in image/video using a trained custom YOLO model."
    )

    with gr.Tab("Image Detection"):
        gr.Markdown("Upload or capture an image below:")
        with gr.Row():
            image_input = gr.Image(sources=["upload", 'clipboard'], type="filepath", label="Choose or Capture Image", height=400)
            image_output = gr.Image(label="Detection Result", height=400)
        detect_btn_image = gr.Button("Detect Fire")
        detect_btn_image.click(fn=detect_image, inputs=image_input, outputs=image_output)

    with gr.Tab("Video Detection"):
        gr.Markdown("Upload a video for detection:")
        with gr.Row():
            video_input = gr.Video(sources=["upload"], label="Upload or Record Video", height=400)
            video_output = gr.Video(label="Detection Result Video", height=400)
        detect_btn_video = gr.Button("Detect Fire in Video")
        detect_btn_video.click(fn=detect_video, inputs=video_input, outputs=video_output)

demo.launch()
