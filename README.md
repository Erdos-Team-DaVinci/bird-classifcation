# bird-classifcation
The Davinci Team of the [Erdős Institute Data Science Bootcamp 2022](https://www.erdosinstitute.org/code) has utilized advances in computer vision technology with the goal to train a machine learning model to classify species of birds. In doing so our project addresses two primary goals:
1. Generate an algorithm that could take images of birds to identify the species.
2. Ensure our model could function even using amateur-level images with a high degree of accuracy, to ensure accessibility of identification.


### Team Members:
- [Soumen Deb](https://www.linkedin.com/in/soumen-deb-193005b0/)
- [Adam Kawash](https://www.linkedin.com/in/adam-kawash-90077b215/)
- [Allison Londeree](https://www.linkedin.com/in/allison-londeree/)
- [Moeka Ono](https://www.linkedin.com/in/moeka-ono/)

## Summary

## Dataset
### Original dataset
The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species). The original dataset "Birds 400" includes 400 bird species with 58388 training images, 2000 test images, and 2000 validation images. All images are 224 X 224 X 3 color images in jpg format. Each image contains only one bird and the bird typically takes up at least 50% of the pixels in the image. 


![random_pics](https://user-images.githubusercontent.com/90373346/171992600-dbc8619b-2b11-44c6-97f0-3a05628a4816.jpg)


In our project, we focused on the species can be found in NY. By cross-referencing the information from [Wikipedia](https://en.wikipedia.org/wiki/List_of_birds_of_New_York_(state)) and Kaggle (as of 6/3/2022), we created a dataset of images of 100 bird species from the original dataset to explore optimal algorithms to classify the selected species. The training dataset of the 100 species had 120-249 images (avg: 149 images) per species and the total of 14,940 images. Both validation and test data included 5 images per species. 


![download](https://user-images.githubusercontent.com/90373346/171978402-7e27502d-81ec-4cb0-a431-84a57647619b.png)

### Demo dataset
We created an independent dataset of 22 bird species with 1-3 images per species, photographed by an amateur photographer, Isaac Ahuvia, in the east coast. The images were minimally preprocessed in 2 ways. 1) All images were cropped in relation to the center of the image and resampled to our desired size of 224 x 224 pixels. 2) Images were cropped manually such that the bird consumed approximately 50% or more of the image, and resampled to our desired size. An example of the preprocession is as below:

![RUBY THROATED HUMMINGBIRD](https://user-images.githubusercontent.com/90373346/171991573-f5b31a99-1e62-4639-a631-040c44b6b15f.jpg)



## Methodology
We combined the original training and validation data images, then splited the dataset into 15% validation, 85% training. We trained the 2 following models.

### 1. Custom convolutional neural network
As our baseline model, we deployed a simple convolutional neural network (CNN). The following is the simple architecture:
1. 3 sets of a convolutional layer with window size 2x2 and a pooling layer with window size 2x2
2. An additional convolutional layer with window size 2x2
3. A flattening layer 
4. A dropout layer of 0.5 to drop inputs randomly to prevent overfitting 
5. 2 fully connected dense layers
6. An output dense layer

We used ReLU activation functions in all convolutional layers and 2 fully connected dense layers, as well as L2 regulation to these dense layers. Softmax was applied in the output layer. We ran the network of xxx epochs to see the training and validation accuracies. 

### 2. VGG-16

The simple architecture is as follows:



## Results
We compared the model performances of 3 models. 

## Web Application
We develoyed our VGG-16 model on [a web app](https://share.streamlit.io/erdos-team-davinci/bird-classifcation/main/app/app_test.py) via Stlearmlit. This web app can provide the 5 most likely bird species of unseen model-naive images photographed by an amatuer photographer in the east coast, as well as the user uploaded images. 

![]

## Future works
There are some of our future productionisation ideas.
### 1. Model improvement 
Due to the time and resource constrains, we were not able to train the species out of NY. We would love to work on models that can predict broader areas on national or continental scales. 

### 2. 
