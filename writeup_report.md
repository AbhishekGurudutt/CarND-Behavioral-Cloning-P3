[//]: # (Image References)

[image1]: ./writeup_images/nvidia_architecture.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


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
To train a model,, data/images are required. This is achieved by using the simulator provided by Udacity, and manually driving the car over the track and recording the data. The simulator stores all the images in a folder along with a csv file which maps each image along with other parameters such as steering angle, speed, throttle, and brake.

#### 2. Model architecture
The Udacity course instruction suggests to use Nvidia model for self-driving car. As described in the Nvidia papers, the architecture was implemented with one Normalization layer to start with, followed by 5 Convolutional layers, one Flatten layer and 5 Fully-connected layer. Since, there was no mention of activation layer in the paper, sigmoid activation was made use in the Fully-connected layer *[**model.py** line 45 to 69]*. Adam optimizer and loss function as Mean Square Error (MSE) were chosen *[**model.py** line 190]*. Below image shows the architecture of Nvidia model.

![alt text][image1]

#### 3. Preprocessing
The images collected from the simulator provides three different images, center, left, and right view. It also provides steering angle in a csv file. The steering angle has to be adjusted for right and left view, and is accomplished by adding or subtracting 0.25 radians of offset for the steering angle.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
