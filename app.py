import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

st.markdown("""
    <style>
        .stMainBlockContainer {
            border: 1px solid #111; 
            padding: 3rem 4rem; 
            width: 100%; 
            border-radius: 2rem;
            box-shadow: 5px 5px 30px rgba(0, 0, 0, 0.2); 
        }
            
        .stMain{
            padding: 5rem;
            height: 100dvh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            
        }
        .prediction-box {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            background-color: rgba(255, 255, 255, 0.85);
            text-align: center;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }
            
        .st-emotion-cache-1gulkj5{
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid #0f0;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            text-align: center;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("cnn_model.h5")
class_labels = ['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain']

st.title(" Nature Scene Classification ")
st.write("Upload an image to classify it into one of five categories.")
st.write("The categories are as follows:")
st.markdown(""" - Forest - Mountain - Desert - Glacier - Coast """)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Layout: Left -> uploader + title | Right -> image & prediction
    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("### Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", use_container_width=True)

    with col2:
        # Preprocessing
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (150, 150))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.markdown(
            f"""
            <div class='prediction-box'>
                <p>Prediction</p>
                <h3><b>{predicted_class}</b></h3>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True
        )
