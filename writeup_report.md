# **Behavioral Cloning**

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center1]: ./examples/center.jpg "Center Driving"
[left1]: ./examples/left1.jpg "Left offset image"
[right1]: ./examples/right1.jpg "Right offset image"
[center-flipped]: ./examples/center-flipped.jpg "Right offset image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* run1.mp4, run2.mp4 Videos of the trained CNN steering the car

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model can be found in model.py (lines 102 - 138) and is based on the [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) convolution neural network (CNN). The keras model code was derived from [here](https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py#L50)

My model network consists of 11 layers. The first 2 layers are for normalization (Lambda layer) and cropping (Cropping2D) irrelevant areas of the image. (119, 122)

The next 5 layers are Convolution2D layers. First 3 layers have a depth of 24, 36 & 48 respectively, with a 5x5 kernel and 2x2 stride. The last 2 layers each have a depth of 64 each with a 3x3 kernel and 1x1 stride.

Finally the output from the convolution layers is flattened and passed into dense layers of size 256, 50, and 10.

To avoid overfitting, there is a dropout layer after the Dense 256 layer and Dense 50 layer. The first Dense layer in the NVIDIA model was actually 1164, but I reduced that to 256 to reduce model size and parameters.

To introduce nonlinearity, each layer has a RELU activation.

The final output layer is a Dense(1) layer. The initial model used from github had tanh activation for the output layer, but that is problematic as we don't want capped output. That was removed and the output of the final layer is used as is.

The final model had 2,307,663 parameters
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 130, 132).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 96). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for multiple laps

#### 3. Model parameter tuning

The model used an adam optimizer for mean squared error, so the learning rate was not tuned manually (model.py line 136).

Tuning for the hyper parameters such epochs and batch size are described in the strategy section below.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the udacity supplied data along with driving of my own around track 1. I used primarily center lane driving with 3 laps around the track.

I attempted to train using recovery laps, but found it didn't have as much imapct as just augmenting the data from the left and right cameras and then adjusting the recording steering value for the center image accordingly.

Due to lack of time, I didn't gather training data on track 2 to speed up iterating on the model type and hyper paramters such as epochs and batch size.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to rely on the [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) for end to end driving based on convolution neural network (CNN). The keras model code was obtained from [here](https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py#L50)

I felt that this would be an appropriate choice because NVIDIA had demonstrated that their model worked for CNNs to steer a car purely on camera images and steering angle training data. I modified the NVIDIA model by reducing the size of the Dense layers to reduce parameter size

In order to gauge how well the model was working, I split my image and steering angle data into a 80% training and 20% validation set. I found that with a first pass, the model worked well. The training and validation loss were both equally low.

To combat the overfitting, I modified the the NVIDIA model to add dropout layers after the Dense(256) and Dense(50) layers

The final step was to run the simulator to see how well the car was driving around track one. Initially I found that the vehicle would fail to turn enough on some of the sharper turns and would fall off the track. It seemed that the turning angle was being clipped. I looked at the final layer and noticed that it had tanh activation on it. I removed that activation since it didn't make sense to have that because we don't want to dampen the output or use it to activate some sort of binary process. It's a granular steering angle and we want to use the raw output there.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. It does weave just a little bit at start of the track, but then maintains center lane behavior throughout. I think that the weaving occurs because the shadow just before the bridge could make the lane appear narrower than it is.

#### 2. Final Model Architecture

My model can be found in model.py (lines 102 - 138) and is based on the [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) convolution neural network (CNN). The keras model code was derived from [here](https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py#L50)

My model network consists of 11 layers. The first 2 layers are for normalization (Lambda layer) and cropping (Cropping2D) irrelevant areas of the image. (119, 122)

The next 5 layers are Convolution2D layers. First 3 layers have a depth of 24, 36 & 48 respectively, with a 5x5 kernel and 2x2 stride. The last 2 layers each have a depth of 64 each with a 3x3 kernel and 1x1 stride.

Finally the output from the convolution layers is flattened and passed into dense layers of size 256, 50, and 10.

To avoid overfitting, there is a dropout layer after the Dense 256 layer and Dense 50 layer. The first Dense layer in the NVIDIA model was actually 1164, but I reduced that to 256 to reduce model size and parameters.

To introduce nonlinearity, each layer has a RELU activation.

The final output layer is a Dense(1) layer. The initial model used from github had tanh activation for the output layer, but that is problematic as we don't want capped output. That was removed and the output of the final layer is used as is.

The final model had 2,307,663 parameters

#### 3. Creation of the Training Set & Training Process

I used the udacity training data set as a starting point. Additionally I did another run by driving 3 times around the track to capture mostly good center driving data. Here is an image of driving in the center of the track

![alt text][center1]

To capture training data for off-center driving, I chose to use data from the left and right cameras instead. I did try to record some recovery driving, but I found that it caused more wobbling in the driving since the recorded data always included some frames of entering a bad part of the track.

By relying on the left and right camera images, I could simulate recovery behavior just as the NVIDIA paper had. The trick was to pick a steering angle offset that would represent the left and right augmented steering angle. By observing steering values being output by the controller and seeing how much it caused the car to turn, I picked an offset of 0.05. For the left images, the steering angle was +0.05 since the left image looks like it's a bit too much on the left side and postive steering offset will bring the car back towards the center. Similarly for the right image, the offset was -0.05 to turn slightly left.

Here is some output from the left and right cameras respectively

![alt text][left1]
![alt text][right1]

I gathered data on track2 as well, but didn't end up using it because I was trying to save some time on optimizing and training the model

To augment the data sat, I also flipped images and angles to prevent bias in the dominant direction of the track

![alt text][center-flipped]

After the collection process, I had 3 additional images and angles from each training set data point. I ended up training with about 43,050 points

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 3 as at that point the validation and training loss stopped decreasing. The validation and training set both had a similar loss so I assumed that the model was not overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I finally tuned the batch size and found that 16 was a good number. It yielded good driving around the track for multiple laps. I also tested by intentionally putting the car into a bad spot near the edge of the curves and found that the model was able to recover the car.

I tried to test the car on track 2, but it didn't work out at all. This would require capturing more data on track 2 and then augmenting the data by applying shadows to the images so that the model can train for low light conditions. I could have also reduced some redundant data from the track by dropping a certain percentage of training data where the car was driving straight so that the car isn't too biased to going straight. Additionaly I think the model parameters could have been reduced by using the right convolution sizes before the dense layers.

The assigment required autonomous driving on track 1 and that worked out well as can be seen in the run1.mp4 and run2.mp4 videos.