# **_distinguish between cats and dogs based on their image_**
# Table of Contents
- [Project Goal](#project-goal)
- [Personal note](#personal-note)
- [Installation](#installation)
- [Background](#background)
- [Data Preparation](#data-preparation)
- [Implementation](#implementation)
- [Results and Analysis](#results-and-analysis)
- [Conclusion](#conclusion)
- [Sources](#sources)
## Project Goal
The goal of the project is to develop an image classification system in Matlab 
that can distinguish between cats and dogs based on their images.
Two methods are used: Harris Corner Detection for a self-trained model and a pre-trained EfficientNet-B0 model.
## Personal note
This project is my first practical project in the field of image processing and machine learning. Many of the ideas and solutions were developed through independent research on the internet. 
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
- **Size of the data set:** The data set contains a limited number of images that were collected specifically for the project. A small data set was used because a larger data set would take too long.
- **Challenge:** As the data set is small, there is a risk of overfitting, especially with Harris Corner Detection.
- **Solution:** The data set is divided into training (80%) and test data (20%) to ensure a balanced evaluation.
### Harris Corner Detection
**Steps for feature extraction:**

 1. the images are converted to grayscale, since the Harris Corner Detector only works with grayscale.
 2. the algorithm detects corners in the image by analyzing local intensity changes in two directions.
 3. the strongest 100 corners are selected and their x- and y-coordinates are stored in a feature vector.
 4. if less than 100 corners are detected, the vector is padded with zeros to ensure a uniform input size.
 
**Classification:** The extracted features are fed into a Support Vector Machine (SVM) model that is trained to distinguish cat and dog images.
### EfficientNet-B0
**Steps for feature extraction:**

1. the images are scaled to size 224x224 to fit the requirements of the pre-trained model.
2. the GlobAvgPool layer is used to extract the global features of the images. This layer summarizes the information of the previous layers and outputs a compact representation of the image.
3. the extracted features are stored in a feature vector, which serves as input for the classification.

**Classification:** As with Harris Corner Detection, the extracted features are fed into an SVM model that classifies the images.

## Implementation
### Preparation
1. set path: mydata refers to the folder containing the pictures of cats and dogs.
2. load images: Using imageDatastore to create a MATLAB object that loads all images in the folder and subfolders.
3. generate labels: contains(x, 'cat') checks whether the file name contains the word “cat” and assigns the label “cat” or “dog” accordingly.
4. split into training and test data: splitEachLabel splits the data set randomly (80 % training, 20 % test).
### Harris Corner Detection
![image](https://github.com/user-attachments/assets/3d45cb9e-32ef-491a-b4e5-562fe7dc6ebb)


1.	Feature extraction with Harris Corner:
-	 The detectHarrisFeatures function finds corners in the screen.
-	Before corner detection, the image is converted from an RGB image to a grayscale image using the rgb2gray function, as the Harris Corner algorithm only works with brightness information.
-	Only the strongest 100 corners are selected to optimize the calculation.
-	The x and y coordinates of these corners are saved as a feature vector.
2.	Padding: If fewer than 100 corners are found, the vector is filled with zeros to ensure a uniform length.
3.	Processing the images: processImages extracts features from all training and test images and saves them in matrices.
### EfficientNet-B0
![image](https://github.com/user-attachments/assets/dedf6c22-0e90-4f67-9987-274651db7280)


1.	Image scaling: EfficientNet-B0 expects inputs of size 224x224. The imresize function scales the images accordingly.
2.	Pre-trained model: efficientnetb0 is a pre-trained model from the MATLAB Deep Learning Toolbox.
3.	Feature extraction: The activations of the GlobAvgPool layer are used as feature vectors for the SVM.
### SMV Classification
![image](https://github.com/user-attachments/assets/0c5f714f-fff1-4487-aeeb-fe25dadc8bc4)
![image](https://github.com/user-attachments/assets/0994b553-fb1f-4fce-bc2a-4efbb2307919)

1.	fitcsvm: Creates an SVM model based on the extracted features and the associated labels.
2.	Prediction and accuracy:
 -	The test images are classified and the predictions are compared with the actual labels.
 -	The accuracy is calculated as a percentage of correctly classified images.

## Results and Analysis
### Bar charts analysis
![image](https://github.com/user-attachments/assets/eb25821f-22f5-4a6f-8719-9b24b45d0378)
![image](https://github.com/user-attachments/assets/0b12da58-5885-41d9-8d42-27eaeccba8f5)
![image](https://github.com/user-attachments/assets/1db6df18-4ffc-43e4-a3ae-f610f01c1643)

The attached bar charts provide a visual comparison of the performance:
1.	In Chart 1, HCD achieves 51.2%, while EfficientNet-B0 achieves 65.9% accuracy. 
2.	In Chart 2, the accuracy of HCD improves to 60.9%, but EfficientNet-B0 also increases to 70.7%
3.	In Chart 3, EfficientNet-B0 achieves its highest accuracy (73.2%), while HCD remains below (58.5%).
The diagrams clearly show that EfficientNet-B0 consistently delivers more precise results than HCD.
### Reasons for performance gap
**Feature representation:**
- HCD only detects local corner points, which are often not sufficient to distinguish between complex objects such as cats and dogs.
- EfficientNet-B0 uses deep convolutional layers to recognize both local and global patterns.
**Pre-trained knowledge:**
- EfficientNet-B0 was pre-trained on ImageNet, a data set with over 14 million images. This prior knowledge enables the model to generalize well even with small data sets.
- HCD has no such prior knowledge and relies exclusively on the data of the current project.
**Data set size:**
- The small data set limits HCD's ability to learn meaningful differences between classes.
- EfficientNet-B0 compensates for this disadvantage with its pre-trained weights.
## Conclusion
The results show that the pre-trained EfficientNet-B0 model performs significantly better than the Harris Corner model (50-60%) with an accuracy of 65-73%. This is because EfficientNet-B0 captures global and abstract features, while Harris Corner only analyzes local corners. Nevertheless, Harris Corner is a simple and resource-efficient method, while EfficientNet-B0 requires more powerful hardware. 
I really enjoyed the project as it gave me new insights into image processing and the differences between classical and modern approaches. It was a valuable learning experience that broadened my knowledge in this field.
### Potential Improvements and Future Work
- **Larger data set:** Using a larger and more diverse data set could improve the performance of both models, especially the Harris Corner model, as it is more sensitive to small data sets.
- **Integration into real-time applications:** A future improvement could be to optimize the system so that it can classify images in real time, e.g. for mobile apps or camera systems.
- **More efficient models:** The use of more advanced pre-trained models could further increase accuracy and efficiency.

## Sources
1. https://de.mathworks.com/matlabcentral/answers/index
2. https://chatgpt.com
3. https://de.mathworks.com/help/matlab/ref/matlab.io.datastore.imagedatastore.html
4. https://de.mathworks.com/help/vision/ref/detectharrisfeatures.html
5. https://de.mathworks.com/help/stats/fitcsvm.html
6. https://de.mathworks.com/help/deeplearning/ref/efficientnetb0.html
7. https://de.mathworks.com/help/stats/regressionlinear.predict.html
8. https://de.mathworks.com/help/deeplearning/ref/seriesnetwork.activations.html


