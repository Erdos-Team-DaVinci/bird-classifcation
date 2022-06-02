# bird-classifcation


## Summary

## Dataset
The dataset used for this project can be found on [kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species). The original dataset "Birds 400" includes 400 bird species with 58388 training images, 2000 test images, and 2000 validation images. All images are 224 X 224 X 3 color images in jpg format. Each image contains only one bird and the bird typically takes up at least 50% of the pixels in the image. 

In our project, we focused on the bird species that have been found in [North America](https://en.wikipedia.org/wiki/List_of_birds_of_the_United_States)(as of 6/1/2022). We have chosen 93 bird species in North America from the original dataset to explore optimal algorithms to classify the selected species. The training dataset of the 93 species had 120-249 images (avg: 147 images) per species. Both validation and test data included 5 images per species. 
![](imgs/NA93_count.png)
