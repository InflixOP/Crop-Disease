import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


def prepare(file):
    img_array = np.array(file)  # Convert the image to a NumPy array
    img_array = img_array / 255.0  # Normalize the image
    return img_array.reshape(-1, 128, 128, 3)  # Reshape for model input

class_dict = {
    'Tomato Bacterial spot': 0,
    'Tomato Early blight': 1,
    'Tomato Late blight': 2,
    'Tomato Leaf Mold': 3,
    'Tomato Septoria leaf spot': 4,
    'Tomato Spider mites Two-spotted spider mite': 5,
    'Tomato Target Spot': 6,
    'Tomato Tomato Yellow Leaf Curl Virus': 7,
    'Tomato Tomato mosaic virus': 8,
    'Tomato healthy': 9
}

def prediction_cls(prediction):
    for key, clss in class_dict.items():
        if np.argmax(prediction) == clss:
            return key

@st.cache_data  # Replaced st.cache with st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))
    return img

def main():
    st.image("./img2.jpg", use_container_width=True)  # Changed to use_container_width

    st.title("Tomato Disease Prediction")
    st.subheader("Please upload the Tomato leaf image : ")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if st.button("Process"):
        if image_file is not None:
            img = load_image(image_file)
            st.image(img, caption="Uploaded Image")
            img = tf.keras.preprocessing.image.img_to_array(img)
            model = tf.keras.models.load_model("model_vgg19.h5")
            img = prepare(img)
            st.subheader(prediction_cls(model.predict(img)))
        else:
            st.warning("Please upload an image.")

if __name__ == "__main__":
    main()
