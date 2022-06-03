import json
#from io import BytesIO
#import os

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image # Strreamlit works with PIL library very easily for Images
import cv2

#import boto3
#from botocore import UNSIGNED  # contact public s3 buckets anonymously
#from botocore.client import Config  # contact public s3 buckets anonymously

@st.cache()
def load_index_to_label_dict(
        path: str = '../src/index_to_class_label.json'
        ) -> dict:
    """Retrieves and formats the
    index to class label
    lookup dictionary needed to
    make sense of the predictions.
    When loaded in, the keys are strings, this also
    processes those keys to integers."""
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {
        int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict


def load_files_from_s3(
        keys: list,
        bucket_name: str = 'bird-classification-bucket'
        ) -> list:
    """Retrieves files anonymously from my public S3 bucket"""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_files = []
    for key in keys:
        s3_file_raw = s3.get_object(Bucket=bucket_name, Key=key)
        s3_file_cleaned = s3_file_raw['Body'].read()
        s3_file_image = Image.open(BytesIO(s3_file_cleaned))
        s3_files.append(s3_file_image)
    return s3_files


@st.cache()
def load_s3_file_structure(path: str = '../src/all_image_files.json') -> dict:
    """Retrieves JSON document outining the S3 file structure"""
    with open(path, 'r') as f:
        return json.load(f)


@st.cache()
def load_list_of_images_available(
        all_image_files: dict,
        image_files_dtype: str,
        bird_species: str
        ) -> list:
    """Retrieves list of available images given the current selections"""
    species_dict = all_image_files.get(image_files_dtype)
    list_of_files = species_dict.get(bird_species)
    return list_of_files


index_to_class_label_dict = load_index_to_label_dict()
all_image_files = load_s3_file_structure()
types_of_birds = sorted(list(all_image_files['test'].keys()))
types_of_birds = [bird.title() for bird in types_of_birds]

model_path='convNetvgg16_100species.h5'

st.title('East Coast Bird Classification')
instructions = """
    Upload an image of a north american bird species or select from the 
    sidebar to get pick an image taken by an amateur photographer. 
    The image you select will be fed
    through the network of your choice in real-time
    and the output will be displayed to the screen.
    """
st.write(instructions)

upload = st.file_uploader('Upload a north american bird image')


dtype_file_structure_mapping = {
        'All Images': 'consolidated',
        'Images Used To Train The Model': 'train',
        'Images Used To Tune The Model': 'valid',
        'Images The Model Has Never Seen': 'test'
    }
data_split_names = list(dtype_file_structure_mapping.keys())

if upload:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  
  if(st.button('Predict')):
    model = tf.keras.models.load_model(model_path)
    x = cv2.resize(opencv_image,(224,224))
    x = np.expand_dims(x,axis=0)  
    x = x.reshape(-1,7*7*512)  
    y = model.predict(x)
    ans=np.argmax(y,axis=1)

    st.title("Here are the five most likely bird species")
    df = pd.DataFrame(data=np.zeros((5, 2)),
                      columns=['Species', 'Confidence Level'],
                      index=np.linspace(1, 5, 5, dtype=int))
    st.write(df.to_html(escape=False), unsafe_allow_html=True)

else:
    dataset_type = st.sidebar.selectbox(
        "Data Portion Type", data_split_names)
    image_files_subset = dtype_file_structure_mapping[dataset_type]

    selected_species = st.sidebar.selectbox("Bird Type", types_of_birds)
    available_images = load_list_of_images_available(
        all_image_files, image_files_subset, selected_species.upper())
    image_name = st.sidebar.selectbox("Image Name", available_images)

    if(st.button('Predict')):
        model = tf.keras.models.load_model(model_path)
        x = cv2.resize(opencv_image,(224,224))
        x = np.expand_dims(x,axis=0)  
        x = x.reshape(-1,7*7*512)  
        y = model.predict(x)
        ans=np.argmax(y,axis=1)

        if image_files_subset == 'consolidated':
            s3_key_prefix = 'consolidated/consolidated'
        else:
            s3_key_prefix = image_files_subset
        key_path = os.path.join(
            s3_key_prefix, selected_species.upper(), image_name)
        files_to_get_from_s3 = [key_path]
        examples_of_species = np.random.choice(available_images, size=3)

        for im in examples_of_species:
            path = os.path.join(s3_key_prefix, selected_species.upper(), im)
            files_to_get_from_s3.append(path)
        images_from_s3 = load_files_from_s3(keys=files_to_get_from_s3)
        img = images_from_s3.pop(0)
        prediction = predict(img, index_to_class_label_dict, model, 5)
        
        st.title("Here are the five most likely bird species")
        df = pd.DataFrame(data=np.zeros((5, 2)),
                      columns=['Species', 'Confidence Level'],
                      index=np.linspace(1, 5, 5, dtype=int))
        st.write(df.to_html(escape=False), unsafe_allow_html=True)














