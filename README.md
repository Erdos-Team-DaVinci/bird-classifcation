# Bird Species Classification
The Davinci Team of the [Erdős Institute Data Science Bootcamp 2022](https://www.erdosinstitute.org/code) has utilized Deep Convolutional Neural network with the goal to train a machine learning application to classify species of birds based on images. In doing so our project addresses two primary goals:
1. Generate an algorithm that could take images of birds to identify the species.
2. Ensure our model could function even using amateur-level images with a high degree of accuracy, to ensure accessibility of identification.


### Team Members:
- [Soumen Deb](https://www.linkedin.com/in/soumen-deb-193005b0/)
- [Adam Kawash](https://www.linkedin.com/in/adam-kawash-90077b215/)
- [Allison Londeree](https://www.linkedin.com/in/allison-londeree/)
- [Moeka Ono](https://www.linkedin.com/in/moeka-ono/)

## Introduction 
The diversity of bird species on Earth is immense. With over 11,000 species identified today, it is no wonder many humans, or birders, take keen interest in observing and documenting their locations, but the list of species that are endangered or critically endangered grows as the threats of shifting birds’ breeding and migratory seasons due to climate warming. For these reasons, identifying species of birds is not only a leisure activity for birders, but it is also crucial to protect the diversity of birdlife we know of today.


## Dataset
### Original Dataset
The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species). The original dataset "Birds 400" includes 400 bird species with 58,388 training images, 2,000 test images, and 2000 validation images. All images are 224 X 224 X 3 color images in jpg format. Each image contains only one bird and the bird typically takes up at least 50% of the pixels in the image. 


![rgb](https://user-images.githubusercontent.com/90373346/172022489-a669d05c-2d3d-4f0c-923c-126a3a7adb06.jpg)


### Our Dataset
In our project, we focused on the species can be found in NY. By cross-referencing the information from [Wikipedia](https://en.wikipedia.org/wiki/List_of_birds_of_New_York_(state)) and Kaggle (as of 6/3/2022), we created a dataset of images of 100 bird species from the original dataset to explore optimal algorithms to classify the selected species. The training dataset of the 100 species had 120-249 images (avg: 149 images) per species and the total of 14,940 images. Both validation and test data included 5 images per species. 

![download](https://user-images.githubusercontent.com/90373346/171978402-7e27502d-81ec-4cb0-a431-84a57647619b.png)


### Demo Dataset
We created an independent dataset of 22 bird species with 1-3 images per species, photographed by an amateur photographer, Isaac Ahuvia, in Long Island, NY. The images were minimally preprocessed in 2 ways. 1) All images were cropped in relation to the center of the image and resampled to our desired size of 224 x 224 pixels. 2) Images were cropped manually such that the bird consumed approximately 50% or more of the image, and resampled to our desired size. An example of the preprocession is as below:

![RUBY THROATED HUMMINGBIRD](https://user-images.githubusercontent.com/90373346/171991573-f5b31a99-1e62-4639-a631-040c44b6b15f.jpg)



## Training and Model Selection
We combined the original training and validation data images, then splited the dataset into 15% validation and 85% training. We used 2 convolutional neural network (CNN) models: custom CNN model and VGG16.

### Custom CNN
As our baseline model, we deployed a simple convolutional neural network (CNN). The following is the simple architecture:
1. 3 sets of a convolutional layer with window size 2x2 and a pooling layer with window size 2x2
2. An additional convolutional layer with window size 2x2
3. A flattening layer 
4. A dropout layer of 0.5 to drop inputs randomly to prevent overfitting 
5. 2 fully connected dense layers
6. An output dense layer

We used ReLU activation functions in all convolutional layers and 2 fully connected dense layers, as well as L2 regulation to these dense layers. Softmax was applied in the output layer. We ran the network of xxx epochs to see the training and validation accuracies. 

### VGG-16

VGG-16 contains 13 convolutional layers, 5 Max Pooling layers, and 3 Dense layers over 6 blocks. On the top of that, 



## Results
VGG16 was used as the model for this project to predict the test set. The model achieved the precision score of 0.81.

![confusion_matrix](https://user-images.githubusercontent.com/90373346/172018515-fac99367-490f-42eb-8060-c7b16f8bc6d8.png)

## Web Application

![chickid_logo](https://user-images.githubusercontent.com/90373346/172003264-b1015d19-24bf-4304-a24a-7e4935ae61e6.jpeg)

We develoyed our VGG-16 model in a prototype app [*ChickID*](https://share.streamlit.io/erdos-team-davinci/bird-classifcation/main/app/app_test.py) via Stlearmlit. The user can select any of demo images that the model has never seen before, or upload their own images to see how the model would predict it. The output of this web app includes the top species predictions with a confidence level.  


![RUBY THROATED HUMMINGBIRD_app](https://user-images.githubusercontent.com/90373346/172022336-4511ccf4-57f4-4a29-87fb-1fc3b3b36295.jpg)

## Future Directions
There are some of our ideas to improve this project.
### Enhance model preprocessing
Our models may predict poorly with the user uploaded images without clearly visible features. When we tarin less cleaned data, the predictions may improve.

### Model improvement
Male birds typically has brighter, more vivid colors than females. Thus, it is possible for the model to predict the same species but different sexes differently. We would like to train the model to handle sexual dimporphim to improve the issue.    
### Model expantion
- Due to the time and resource constrains, we were not able to train the species out of NY. We would love to expand more bird species on national or continental scales.
- Currently, we can only predict species in our trained dataset. We hope to implement a function that outputs not included in a list when predicted confidence is under a given threshold.   

