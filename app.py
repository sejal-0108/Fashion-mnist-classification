import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


# Load the pre-trained CNN model
model = keras.models.load_model('C:/Users/Nilesh/OneDrive/Documents/mnist/model3.h5')

# Define the product labels
product_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize the image to 28x28 pixels
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Main function for Streamlit app
def main():
    st.title('Fashion Item Classification')
    st.write('Upload an image of a fashion item to classify it.')

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image and make prediction
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        # Display the prediction and confidence with styled output
        st.markdown(
            f'<div class="prediction-container"><p class="prediction">Prediction: {product_labels[predicted_class]}</p>'
            f'<p class="confidence">Confidence: {confidence:.2%}</p></div>',
            unsafe_allow_html=True
        )

# CSS styles
st.markdown(
    """
    <style>
        .prediction-container {
            background: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(9,9,121,1) 35%, rgba(0,212,255,1) 100%);
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }
        .prediction {
            font-size: 18px;
            font-weight: bold;
            color: white;
        }
        .confidence {
            font-size: 16px;
            font-weight: bold;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Run the app
if __name__ == '__main__':
    main()
