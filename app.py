
import streamlit as st
import tensorflow as tf

from keras.preprocessing import image


from tensorflow import keras
import numpy as np
from PIL import Image # Strreamlit works with PIL library very easily for Images
import cv2
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model



st.title("Bean Disease Detection and Classifier App")
upload = st.file_uploader('Upload the image of the bean plant')


model=tf.keras.models.load_model('/content/bean_disease_classifier_model_MobileNet3.hdf5')


  
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
  img = Image.open(upload)

  st.image(img,caption='Uploaded Image',width=300)

  if(st.button('Predict')):

    x = cv2.resize(opencv_image,(224,224))
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    y = model.predict(x)

    ans=max(y[0])
    

    for i in range(0,3):
      if(ans==y[0][i]):
        if(i==0):
          st.write("Bean plant is diseased with angular_leaf_spot")
        elif(i==1):
          st.write("Bean plant is diseased with bean rust")
        else:
          st.write('Bean plant is healthy')
    
   