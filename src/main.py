import gradio as gr
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch

import config
from data_processing import labels, id2label, label2id

model = AutoModelForImageClassification.from_pretrained(config.MODEL_CHECKPOINT, num_labels=len(labels), id2label=id2label, label2id=label2id)
processor = AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT, use_fast=True)

def emotion_analysis(image):
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probability = logits.softmax(-1)
        predicted_class_idx = probability.argmax(-1).item()

        predicted_label = id2label[str(predicted_class_idx)]
        # predicted_probability = probability[0][predicted_class_idx].item()

    return f"This person emotion is {predicted_label}"

def emotion_classifier():
    demo_app = gr.Interface(
        fn = emotion_analysis,
        inputs = gr.Image(type="pil"),
        outputs = gr.TextArea(),
        title = "Face Emotion Analyser"
    )

    demo_app.launch()

if __name__=="__main__":
    emotion_classifier()