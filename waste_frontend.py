import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model(r"E:\waste dataset\waste_classifier_model.keras")

def classify_waste(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    label = "Recyclable" if prediction[0][0] > 0.5 else " Non Recyclable"
    
    return f"Prediction: {label}"
interface = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Waste Classifier",
    description="Upload an image of waste to classify it as Recyclable or Non-Recyclable.",
    theme="huggingface",
)

interface.launch()
