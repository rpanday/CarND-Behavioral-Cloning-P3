# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia.png "Model Visualization"
[image2]: ./examples/left_2016_12_01_13_34_15_145.jpg "left"
[image3]: ./examples/center_2016_12_01_13_34_15_145.jpg "center"
[image4]: ./examples/right_2016_12_01_13_34_15_145.jpg "right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Required Files

#### 1. Are all required files submitted?

My project includes the following files:
* [model.py](https://github.com/rpanday/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/rpanday/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/rpanday/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [writeup_report.md](https://github.com/rpanday/CarND-Behavioral-Cloning-P3/blob/master/writeup_report.md) summarizing the results

### Qualify of Code

#### 1. Is the code functional?
Using the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim) and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 2. Is the code usable and readable?

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments & self explanatory function names to explain how the code works. I experimented with LENET5 and Nvidia model but finally used the Nvidia model. More details in the Model Architecture section.

### Model Architecture and Training Strategy

#### 1. Has an appropriate model architecture been employed for the task?

My final trained model.h5 is based on Nvidia model architecture stack from [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). This model was introduced in Chapter 15 of lesson Project Behavior cloning. This can be found in code model.py at lines 67 to 102.
```
def nvidia_model(input_shape):
```
The network architecture consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. But my model actually has 19 layers in terms of keras layers because it has cropping and dropouts  as well. More details are provided in next sections.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.
```
 model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
 ``` 
I also added a cropping layer on top of all this.
```
model.add(Cropping2D(((70, 25), (0, 0))))
```

#### 2. Has an attempt been made to reduce overfitting of the model?

The model contains 7 dropout layers in order to reduce overfitting (model.py lines 75 to 95). 
```
model.add(Dropout(0.2))
```
I used 80% data for training and 20% data for validation (model.py line 127).
```
test_size=0.2
train_samples, validation_samples = train_test_split(samples, test_size=test_size)
 ```
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
I had faced problem of overfitting with the basic LENET5 model in my model.py because test loss was decreasing but validation loss was fluctuating from low to high. Later I switched to Nvidia model and added dropout layers to get decreasing validation loss and better overall fit.

#### 3. Have the model parameters been tuned appropriately?

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).
```
model.compile(optimizer='adam', loss='mse')
```
The model was trained for 3 epochs with a batch size of 32.

#### 4. Is the training data chosen appropriately?

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and steering correction using multiple camera data for recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Is the solution design documented?

I initially used the classic LENET5 model on just the centered images. There were no droput layers added. I trained the model and tested on the simulator. The car was driving but had difficulty on the turns. 
```
def lenet5_model(input_shape): #model.py:105
```
Later I collected more data for an opposite run (clockwise) around the track (it is anticlockwise predominantly). The idea was to add a correction factor around the turns but it did not improve the situation much. I could also see overfitting although I stopped the model training at earlier epochs till the validation loss was decreasing.

Later I switched to Nvidia model and trained on same input as earlier. I still had the same problem of difficultly in negotiating turn after the bridge.
```
def nvidia_model(input_shape): #model.py:67
```

This time I added dropout layers to reduce overfitting. I also augmented data or increased the number of samples by using multiple camera images with steering correction factor of 0.2.
```
#model.py:40
correction = 0.2
measurement = float(line[3])
measurements.append(measurement)
measurements.append(measurement + correction)
measurements.append(measurement - correction)
```
I also switched to the data provided in the workspace already instead of using my own data because it was getting difficult to keep uploading data to workspace and adding more. I used `23372 images` to train the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. This can be seen in file `video.mp4`

#### 2. Is the model architecture documented?

My final trained model.h5 is based on Nvidia model architecture stack from [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). This model was introduced in Chapter 15 of lesson Project Behavior cloning. This can be found in code model.py at lines 67 to 102.
More details are already discussed in section 1 of *Model Architecture and Training Strategy* above.
Base architecture image is here but my model uses 19 layers due to addition of dropout layers.

![alt text][image1]


#### 3. Is the creation of the training dataset and training process documented?

I have already documented the process in section 1 of this part. I finally used the data provided in the workspace. I used camera images from all 3 left/right/center cameras and added correction factor of 0.2 to steering measurement.

Here is an exmaple of image from all 3 cameras.

![alt text][image2]
![alt text][image3]
![alt text][image4]

I ended up with `23372 images` for my training and validation. I used a 80:20 split between training and validation.

This data was further subjected to normalization to reduce the scale of error and cropping to speed up the model by cutting out redundant details.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model for 3 epochs on a batch size of 32. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 because subsequent epochs did not give me decrease in validtion loss and training time was around 15-20 minutes per epoch in GPU workspace. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Simulation
#### Is the car able to navigate correctly on test data?

The car drives around track 1 in autonomus mode correctly. No tire leaves the drivable portion of the track surface. The car does not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle). The sample video is recorded using drive.py and simulator and can be seen in `video.mp4` file.
