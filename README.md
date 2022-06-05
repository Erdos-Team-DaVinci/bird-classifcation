# Bird Species Classification
The Davinci Team of the [Erdős Institute Data Science Bootcamp 2022](https://www.erdosinstitute.org/code) has utilized Deep Convolutional Neural network with the goal to train a machine learning application to classify species of birds based on images. In doing so our project addresses two primary goals:
1. Generate an algorithm that could take images of birds to identify the species.
2. Ensure our model could function even using amateur-level images with a high degree of accuracy, to ensure accessibility of identification.


### Team Members:
- [Soumen Deb](https://www.linkedin.com/in/soumen-deb-193005b0/)
- [Adam Kawash](https://www.linkedin.com/in/adam-kawash-90077b215/)
- [Allison Londeree](https://www.linkedin.com/in/allison-londeree/)
- [Moeka Ono](https://www.linkedin.com/in/moeka-ono/)

Our presentation record can be found at [here](https://www.erdosinstitute.org/project-database).


## Introduction 
The diversity of bird species on Earth is immense. With over 11,000 species identified today, it is no wonder many humans, or birders, take keen interest in observing and documenting their locations, but the list of species that are endangered or critically endangered grows as the threats of shifting birds’ breeding and migratory seasons due to climate warming. For these reasons, identifying species of birds is not only a leisure activity for birders, but it is also crucial to protect the diversity of birdlife we know of today.


## Dataset
### Original Dataset
The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species). The original dataset "Birds 400" includes 400 bird species with 58,388 training images, 2,000 test images, and 2000 validation images. All images are 224 X 224 X 3 color images in jpg format. Each image contains only one bird, typically taking up at least 50% of the pixels in the images.


![rgb](https://user-images.githubusercontent.com/90373346/172022489-a669d05c-2d3d-4f0c-923c-126a3a7adb06.jpg)


### Our Dataset
In our project, we focused on the species can be found in NY. By cross-referencing the scraped information from [Wikipedia](https://en.wikipedia.org/wiki/List_of_birds_of_New_York_(state)) and Kaggle (as of 6/3/2022), we created a dataset of images of 100 bird species from the original dataset to explore optimal algorithms to classify the selected species. The training dataset of the 100 species had 120-249 images (avg: 149 images) per species and the total of 14,940 images. Both validation and test data included 5 images per species. 


### Demo Dataset
We created an independent dataset of 22 bird species with 1-3 images per species, photographed by an amateur photographer, Isaac Ahuvia, in Long Island, NY. The images were minimally preprocessed in 2 ways: 1) all images were cropped in relation to the center of the image and resampled to our desired size of 224 x 224 pixels, and 2) images were cropped manually such that the bird took up approximately 50% or more of the image, and resampled to our desired size. An example of the preprocessing is as below:

![RUBY THROATED HUMMINGBIRD](https://user-images.githubusercontent.com/90373346/171991573-f5b31a99-1e62-4639-a631-040c44b6b15f.jpg)



## Training and Model Selection
We deployed 2 convolutional neural network (CNN) models: custom CNN model and VGG16.

### Custom CNN
We deployed a simple collection of convolutional layers based on two fundamental principles of computer vision: 1. Translational invariance, and 2. Spatial hierarchy. This base model contains 4 convolutional layers, 3 Max Pooling layers, and 2 Dense layers. We used ReLU activation functions in all convolutional layers and 2 fully connected dense layers, as well as L2 regulation to these dense layers. We used softmax for output probabilities. We ran a network of 260 epochs to see the training and validation accuracies. We found our model overfitted likely due to our small dataset. Then, we also used augmented data that ensures the neural network does not see the same image twice during the training, effectively creating an illusion in the model that the training sample is more than it is. This strategy resolved the initial overfitting trend, but the accuracy scores, 76% as maximum, were still not great.

![base_result](https://user-images.githubusercontent.com/90373346/172034366-5a18756b-b3ff-4edd-893e-d1105dc00cde.jpg)

### VGG16
We also used a pre-trained model, VGG16, from imageNet was trained on 1.4 M images. VGG16 is one of the most popular and highly-performing models, yet easier to interpret. We used transfer learning (i.e., a frozen convolutional base) with a new classifier. We kept the pre-trained weights from the convolutional base of VGG16, except for the last 3 convolutional layers that we fine-tuned with our data. This method worked better for our model, achieving a validation accuracy of 95%.   

![vgg16_result](https://user-images.githubusercontent.com/90373346/172034664-1afb6699-377b-4a05-b474-bfa582a2d5f5.jpg)


## Results
To predict the test set, we used the VGG16 model. With fine-tuned VGG models and data augmentation, we achieved 80% test accuracy despite having a small dataset to train. We found progressive improvements with augmentation techniques and transfer learning. 

![confusion_matrix](https://user-images.githubusercontent.com/90373346/172018515-fac99367-490f-42eb-8060-c7b16f8bc6d8.png)

## Web Application

![chickid_logo](https://user-images.githubusercontent.com/90373346/172003264-b1015d19-24bf-4304-a24a-7e4935ae61e6.jpeg)

With our VGG16 model, we developed a user-friendly prototype app [*ChickID*](https://share.streamlit.io/erdos-team-davinci/bird-classifcation/main/app/app_test.py) via Stlearmlit. The user can select any demo images with different preprocess levels and our test image set that the model had never trained or upload their photographs to see how the model would predict it. The output of this web app includes the top species prediction.  


![RUBY THROATED HUMMINGBIRD_app](https://user-images.githubusercontent.com/90373346/172029212-c2b41d81-86ac-4c3f-86b4-432b747966c6.jpg)


## Future Directions
There are some of our ideas to improve this project.
### Enhance model preprocessing
Our model performance reduced when we tested with naturalistic images, likely due to a lack of clearly visible features. We hope to address this through improved preprocessing.

### Model improvement
There is rich variation between the appearance of bird sexes: male birds typically have brighter, more vivid colors than females. Thus, the model can predict the same species but different sexes differently. Additional training is needed as the majority of our training data is on male birds.    
### Model expantion
- Due to the time and resource constraints, we were not able to train the species out of NY. We would love to expand more bird species on national or continental scales.
- Currently, we can only predict species in our trained dataset. We hope to implement a function that outputs not included in a list when a predicted confidence is under a given threshold.     

