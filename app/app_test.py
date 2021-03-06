import json
import glob
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import cv2


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

@st.cache()
def load_image_file_structure(path: str = './demo_image_list.json') -> dict:
    """Retrieves JSON document outining the image directory structure"""
    with open(os.path.join(os.path.dirname(__file__), path), 'r') as f:
        return json.load(f)



#index_to_class_label_dict = load_index_to_label_dict()
all_image_files = load_image_file_structure()
all_image_paths = glob.glob("demo_img/*/*/*")
types_of_birds = sorted(list(all_image_files['clean_demo_22'].keys()))
types_of_birds = [bird.title() for bird in types_of_birds]
labelsDF = pd.read_csv(os.path.join(os.path.dirname(__file__), 'labelsDF.csv'))

model_path='./convNetvgg16_AugFT100NYa.h5'
logo_path = './logo.png'

st.image(os.path.join(os.path.dirname(__file__), logo_path))
#st.title("ChickID")
st.subheader('ChickID - New York Bird Identification')
instructions = """
    Upload an image of a bird species found in New York State OR select from the 
    sidebar to get pick an image taken by an amateur photographer, or from our Kaggle testing set. 

    The image you select will be fed
    through the network of your choice in real-time
    and the output will be displayed to the screen.

    Note that none of the images availible for selection have been seen 
    by the model before.
    """
st.write(instructions)

upload = st.file_uploader('Upload a north american bird image')


dtype_file_structure_mapping = {
        'Cleaned Images': 'clean_demo_22',
        'Minimally Preprocessed Images': 'rough_demo_22',
        'Raw Images': 'raw_demo_22',
        'Images From Testing Set':'test_22'
         }

data_split_names = list(dtype_file_structure_mapping.keys())

if upload:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
    img = Image.open(upload)
    st.image(img,caption='Uploaded Image',width=300)
  
  #if(st.button('Predict')):
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), model_path))
    x = cv2.resize(opencv_image,(224,224))
    x = np.expand_dims(x,axis=0)  
    y = model.predict(x)
    ans=np.argmax(y,axis=1)

    #st.title("Here are the five most likely bird species")
    st.write('Top Predicted Bird:', 
      str(labelsDF.loc[labelsDF['label_index'] == y.argmax(), 'labels'].values[0]))
    #st.write('Predicted Bird 2:', 
    #  str(labelsDF.loc[labelsDF['label_index'] == np.argsort(np.max(y, axis=0))[-2], 'labels'].values[0]),
    #  "{:.3%}".format(np.sort(np.max(y, axis=0))[-2]))
    #st.write('Predicted Bird 3:', 
    #  str(labelsDF.loc[labelsDF['label_index'] == np.argsort(np.max(y, axis=0))[-3], 'labels'].values[0]),
    #  "{:.3%}".format(np.sort(np.max(y, axis=0))[-3]))
    #st.write('Predicted Bird 4:', 
    #  str(labelsDF.loc[labelsDF['label_index'] == np.argsort(np.max(y, axis=0))[-4], 'labels'].values[0]),
    #  "{:.3%}".format(np.sort(np.max(y, axis=0))[-4]))
    #st.write('Predicted Bird 5:', 
    #  str(labelsDF.loc[labelsDF['label_index'] == np.argsort(np.max(y, axis=0))[-5], 'labels'].values[0]),
    #  "{:.3%}".format(np.sort(np.max(y, axis=0))[-5]))

else:
    dataset_type = st.sidebar.selectbox(
        "Preprocessing Type", data_split_names)
    image_files_subset = dtype_file_structure_mapping[dataset_type]

    selected_species = st.sidebar.selectbox("Bird Type", types_of_birds)
    available_images = load_list_of_images_available(
        all_image_files, image_files_subset, selected_species.upper())
    image_name = st.sidebar.selectbox("Image Name", available_images)

    demo_img_path = os.path.join(os.path.dirname(__file__), './demo_img/',image_files_subset, selected_species.upper(), image_name)

    image_from_existing_demo = Image.open(demo_img_path)
    st.image(image_from_existing_demo,caption=str(selected_species),width=300)

    #if(st.button('Predict')):
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), model_path))
    x = cv2.resize(np.float32(image_from_existing_demo),(224,224))
    x = np.expand_dims(x,axis=0)  
    y = model.predict(x)
    ans=np.argmax(y,axis=1)
    
    #st.title("Here are the five most likely bird species")
    st.write('Top Predicted Bird:', 
      str(labelsDF.loc[labelsDF['label_index'] == y.argmax(), 'labels'].values[0]))
    #st.write('Predicted Bird 2:', 
    #  str(labelsDF.loc[labelsDF['label_index'] == np.argsort(np.max(y, axis=0))[-2], 'labels'].values[0]),
    #  "{:.3%}".format(np.sort(np.max(y, axis=0))[-2]))
    #st.write('Predicted Bird 3:', 
    #  str(labelsDF.loc[labelsDF['label_index'] == np.argsort(np.max(y, axis=0))[-3], 'labels'].values[0]),
    #  "{:.3%}".format(np.sort(np.max(y, axis=0))[-3]))
    #st.write('Predicted Bird 4:', 
    #  str(labelsDF.loc[labelsDF['label_index'] == np.argsort(np.max(y, axis=0))[-4], 'labels'].values[0]),
    #  "{:.3%}".format(np.sort(np.max(y, axis=0))[-4]))
    #st.write('Predicted Bird 5:', 
    #  str(labelsDF.loc[labelsDF['label_index'] == np.argsort(np.max(y, axis=0))[-5], 'labels'].values[0]),
    #  "{:.3%}".format(np.sort(np.max(y, axis=0))[-5]))



Credits = """
    

    A special thanks to Isaac Ahuvia for providing demonstration images.

    Test images were taken from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species?resource=download).
    """

st.write(Credits)









