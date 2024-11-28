# distinguish between cats and dogs based on their image
# Table of Contents
- Project Goal
- Personal note
- Installation
- Background
- Data Preparation
- Implementation
- Results and Analysis
- Conclusion
- Sources
## Project Goal
The goal of the project is to develop an image classification system in Matlab 
that can distinguish between cats and dogs based on their images.
Two methods are used: Harris Corner Detection for a self-trained model and a pre-trained EfficientNet-B0 model.
## Personal note
This project is my first practical project in the field of image processing and machine learning. Many of the ideas and solutions were developed through independent research on the internet and with the assistance of ChatGPT. 
Working on this project has significantly deepened my understanding of key concepts and techniques.
## Installation
- MATLAB
- Deep Learning Toolbox
- EfficientNet-B0 Model Add-On

![image](https://github.com/user-attachments/assets/21ea62b1-ed6d-433f-b5ff-e72708085d19)

## Background
### Introduction to Harris Corner Detection
Harris Corner Detection is an algorithm that is used to identify corners in an image.
A corner is defined as a point where the intensity of the image changes significantly in two directions.
#### How it works
1. the algorithm converts the image to grayscale.
2. it calculates the gradients (changes) in the x and y directions.
3. a measure for the strength of the corner is calculated and points with high values are regarded as corners.
### Introduction to EfficientNet-B0
EfficientNet-B0 is a pre-trained model that has been trained on a large dataset called ImageNet. ImageNet contains over 14 million images and covers more than 20,000 classes.
EfficientNet belongs to the Convolutional Neural Networks (CNNs), a special architecture for image processing.
#### Why EfficientNet-B0
I previously had AlexNet, but after personal consultation it was recommended that I should not use AlexNet as it is relatively old.
## Data Preparation
The data set used consists of cat and dog pictures stored in a folder called my_pics. The images are stored without subfolders and the file names contain either “cat.(number)” or “dog.(number)”, which is used for label assignment.
Size of the data set: The data set contains a limited number of images that were collected specifically for the project. A small data set was used because a larger data set would take too long.
Challenge: As the data set is small, there is a risk of overfitting, especially with Harris Corner Detection.

## Implementation
## Results and Analysis
## Conclusion
## Sources

