import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

#Models to be Loaded
bs = tf.keras.models.load_model("")
cs = tf.keras.models.load_model("")
ls = tf.keras.models.load_model("")

def prediction(model, image):
    image=np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return"positive" if prediction[0]>0.5 else"negative"

st.title("CT-SCAN AIDER")
st.sidebar.title("Select diagnosis type")
diagnoses_type = st.sidebar.radio(
    "choose the desired CT Scan Predictor:",
    ("Brain Stroke Detection", "Lung Cancer Detection", "COVID Lung Detection")
)
st.header(f"{diagnosis_type}")

if diagnosis_type == "Brain Stroke Detection":
    st.write("Upload a respective CT scan image .")
elif diagnosis_type == "Lung Cancer Detection":
    st.write("Upload a respective CT scan image .")
elif diagnosis_type == "COVID Lung Detection":
    st.write("Upload a respective CT scan image.")

uploaded_file = st.file_uploader("Choose a CT Scan image...", type=["jpg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="CT Scan", use_column_width=True)

    image = np.array(image)

    if diagnosis_type == "Brain Stroke Detection":
        pred = prediction(bs, image)
        st.write(f"### Brain Stroke Prediction: {pred}")

    elif diagnosis_type == "Lung Cancer Detection":
        pred = prediction(ls, image)
        st.write(f"### Lung Cancer Prediction: {pred}")

    elif diagnosis_type == "COVID Lung Detection":
        pred = prediction(cs, image)
        st.write(f"### COVID Lungs Prediction: {pred}")

st.sidebar.write("### Additional Info")
st.sidebar.markdown("[👤 | Akshay Balaji](https://github.com/Mr-Ak-Shay)")
