#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./example_images/center_lane_driving.jpg "Center Lane Driving"
[image4]: ./example_images/recover_from_left.jpg "Recovery from left"
[image5]: ./example_images/recover_from_right.jpg "Recovery from right"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
My model is mostly based on NVIDIA architecture, which consists of 5 CNN layers, dropout and 4 dense layer. Among the 5 CNN layers, 3 have 5x5 filter sizes and 2 have 3x3 filter size (model.py lines 51-55). These layers includes RELU layers to introduce nonlinearity. Then a dropout layer is subsequent to reduce overfitting (model.py lines 56).

At the beginning of the model, data is normalized in the model using a Keras lambda layer and cropped by removing top 50 lines and bottom 20 lines of image data.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 62).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code 63).


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting (code 63).

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 62).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from small network as baseline and then improve it step by step.

My first step was to use a convolution neural network model similar to the Lenet I but it didn't work well both on validation set or the evaluation drive.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had both high mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was not the right choice.

To improve the model, I adopt the NVIDIA architecture which achieves better reusults than my first attempt.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track,
especially those sharp turns after the bridge. To improve the driving behavior in these cases, I added more training data specificaly for those corner cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

Final Model Architecture consists of 5 CNN layers, dropout and 4 dense layer. Among the 5 CNN layers, 3 have 5x5 filter sizes and 2 have 3x3 filter size (model.py lines 51-55). These layers includes RELU layers to introduce nonlinearity. Then a dropout layer is subsequent to reduce overfitting (model.py lines 56).

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer right and left. :

![alt text][image4]
![alt text][image5]

I didn't record data for track 2.

To augment the data sat, I also flipped images and angles thinking that this would help, but it didn't help that much. 


After the collection process, I had 16072 number of data points. I then preprocessed this data by dividing pixel intesities with 255 and croping the images


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.


