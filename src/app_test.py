#import json
#from io import BytesIO
#import os

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image # Strreamlit works with PIL library very easily for Images
import cv2

#import boto3
#from botocore import UNSIGNED  # contact public s3 buckets anonymously
#from botocore.client import Config  # contact public s3 buckets anonymously
#import pandas as pd

model_path='convNetvgg16_100species.h5'

st.title("East Coast Bird Test")
upload = st.file_uploader('Upload a EC image')
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  if(st.button('Predict')):
    model = tf.keras.models.load_model(model_path)
    x = cv2.resize(opencv_image,(224,224))
    #x = np.expand_dims(x,axis=0)  
    x = x.reshape(-1,7*7*512)  
    y = model.predict(x)
    ans=np.argmax(y,axis=1)

