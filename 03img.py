import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

product_label = {
    0: 'T-shirt',
    1: 'Trousers',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}

model = tf.keras.models.load_model('C:/Users/Nilesh/OneDrive/Documents/mnist/model3.h5')

def predict_image(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.expand_dims(image, 0)
    prediction = model.predict(image)
    return prediction.argmax(-1), (100 * prediction.max(-1)).astype('int32')

def display_prediction(prediction, confidence):
     predicted_class_index = np.argmax(prediction)
     predicted_label = product_label[predicted_class_index]
     st.write(f'Predicted Class: {predicted_label[predicted_class_index]}')
     st.write(f'Confidence : {confidence.item()}')
     st.write(f'Predicted Class : {prediction.item()}')

if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('Fashion MNIST Classifier')
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'], key='file_uploader_key')
    if uploaded_file is not None:
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(28, 28))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image = tf.keras.preprocessing.image.img_to_array(image)
        prediction, confidence = predict_image(image)
        display_prediction(prediction, confidence)