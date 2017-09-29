# Behaviorial Cloning Project

[//]: # (Image References)

[image1]: ./writeup_images/nvidia_architecture.png "Model Visualization"
[image2]: ./writeup_images/before.png "Before Normalization"
[image3]: ./writeup_images/after.png "After Normalization"
[image4]: ./writeup_images/left.jpg "Training images of left view"
[image5]: ./writeup_images/center.jpg "Training images of center view"
[image6]: ./writeup_images/right.jpg "Training images of right view"
[image7]: ./writeup_images/shadow_image.png "Image added with shadow"

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network

### Introduction
The main aim of this project is to use deep learning principles to teach the car on how to autonomously drive in a simulated environment. Simulator is provided by Udacity, which is used in both training and autonomous driving. During training phase, camera data of center, right, and left view along with steering angle, throttle, brake and speed are stored. `model.py` contains code for preprocessing and model generation. `model.h5` contains the saved weights from the convolution network. `drive.py` is provided by Udacity for autonomous drive of car in a simulator with the help of the model created in `model.py` and weights saved in `model.h5`.

Code to start the server for autonomous driving is
```
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. Data collection
To train a model, data/images are required. This is achieved by using the simulator provided by Udacity, and manually driving the car over the track and recording the data. The simulator stores all the images in a folder along with a csv file which maps each image along with other parameters such as steering angle, speed, throttle, and brake.

#### 2. Model architecture
The Udacity course instruction suggests to use Nvidia model for self-driving car. As described in the Nvidia papers, the architecture was implemented with one Normalization layer to start with, followed by 5 Convolutional layers, one Flatten layer and 5 Fully-connected layer. Since, there was no mention of activation layer in the paper, sigmoid activation was made use in the Fully-connected layer _[**model.py** line 45 to 69]_. Adam optimizer and loss function as Mean Square Error (MSE) were chosen _[**model.py** line 190]_. Below image shows the architecture of Nvidia model.

![alt text][image1]

#### 3. Preprocessing
The images collected through the simulator provides images of size 320x160, but the Nvidia model intakes images of size 220x66. The input images are initially cropped to remove the additional data _[**model.py** line 24]_. The cropped images are then resized to 220x66 _[**model.py** line 25]_. These images are converted to YUV image space since the Nvidia model accepts YUV format _[**model.py** line 26]_.

The images collected from the simulator provides three different images, center, left, and right view.

| Left | Center | Right |
|:-: |:-: |:-: |
|![alt text][image4]|![alt text][image5]|![alt text][image6]|

It also provides steering angle in a csv file. The steering angle has to be adjusted for right and left view, and is accomplished by adding or subtracting 0.25 radians of offset for the steering angle.

To increase the number of training data, each image is flipped to provide images in the opposite direction of test track. The steering angle is negated for these images  _[**model.py** line 175 to 177]_.

#### 4. Design Approach

With minimal amount of training data, a convolutional neural network model has to be developed to help the car drive autnomously in a simulated track. To acheive this, Nvidia architecture is used to train the model. The preprocessed images, along with the steering angle values were fed to Nvidia architecture to generate a model. This model was tested on the simulator to analyse the performance. Initally, the car drove straight on track and started to drive off track at turns.

##### 4.1 Dropout layer
While studying about convolutional neural network, I learnt about overfitting problem and how it can be reduced by addition of Dropout layers. In the Nvidia architecture, the same was implemented _[**model.py** line 60 to 66]_.

##### 4.2 Normalizing the training data
After analysis, I noticed that the number of images of straight path were much larger than the images of turning. In other words, the steering angle of 0, 0.25 and -0.25 radians were much higher than other steering angles. The training data had to be normalized before training the model.

![alt text][image2]

During normalization, additional images with steering angle of 0, 0.25 and -0.25 radians were removed _[**model.py** line 72 to 96]_.

![alt text][image3]

Normalization helped to steer the car much better during turns.

##### 4.3 Adding shadows
The car used to be on the track until the point when it encountered a shadow from a tree. There is a bridge after this, and the car used to drive off the track and into the water, or used to crash onto the bridge. Shadows at random location are added to the images before training the model _[**model.py** line 98 to 116]_. This model helped the car to maneuver during shadows, and drove autonomously on the track.

![alt text][image7]

##### 4.4 Generator
To load the dataset for training batch wise, generator functions in keras can be made use of. Two different generator functions are created, one for training and the other for validating. Batch size of 64 is chosen as default. Images and steering angle are fed as input for the generators. Code for training data Generator can be found in _**model.py** line 118 to 129_ and for validation can be found in _**model.py** line 132 to 143_. The Generator function runs continously, returning batches of image data to the model.

##### 4.5 Activation layer
After reading about different types of activations in machine learning, I found out that Sigmoid activation layer is slower in learning and has a vanishing gradient problem. To avoid these disadvantages, Relu activation layer was made use.

#### [Output video](https://github.com/AbhishekGurudutt/CarND-Behavioral-Cloning-P3/blob/master/output.mp4)

 ### Conclusion
 I enjoyed working on this project. With no background in Machine learning, I have started to learn many concepts in this course, and platforms to use. Training the car with minimal efforts was possible due to the provision of Simiulator and testing due to the provision of `drive.py` by Udacity. I have not tested on the challenge track due to time constraints and I would like to revisit and work on this in future.
