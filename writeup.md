# **Traffic Sign lassification and Recognition**

For this project, I performed bellow steps:
---

## Step 0: Load The Data

First off, I loaded the data from the corresponding files and creates the training, validation, and test feature and label sets.


## Step 1: Data exploration
Exploring the data sets reveals that, 
Number of training examples   = 34799
Number of validation examples = 4410
Number of testing examples    = 12630
Image data shape  = (32, 32, 3)
Number of classes = 43

### Visualization of the training dataset

A sample ![./examples/visual1.png][image] of each 43 sign types is shown as the first visualization practice. 
The following ![./examples/visua2.png][histogram] demonstrated how many images of each sign type exists in the training data set.


## Step 2: Data augmentation

The first analyses showd that the data size is not big enough. Moreover, the images needed more generalization. Therefore two augmentation techniques are perforemed on training set, which increase the dataset size from 34799 to 626382. These techniques are %80 central scalling and 7 rotations within the interval of [-18, 18] degrees. 


## Step 3: Pre-Processing by grayscaling and normalization

Then the images were converted into grayscales and the image arrays normalized from [0, 255] intervals to an intandard form of [-1, 1] interval.


### Prepare data to be fed into Tensorflow functions

The array shapes of all three features data sets needed to be change to have another dimention of size 1 as the last dimention to be compatible with Tensorflow functions.


### Define Tensorflow variables

Then, I defined the variables needed in Tensorflow functions.


## Step 4: Modeling

### Model architecture

The architecture is a LeNet (introduced in the previous lectures), which consists of:

Layer 1: Convolutional
    Activation
    Pooling
Layer 2: Convolutional
    Activation
    Pooling
    Flatten
Layer 3: Fully Connected
    Activation
Layer 4: Fully Connected
    Activation
Layer 5: Fully Connected


### Model properties

To train the model, I consider the batch size of 300, due to a larger dataset size. The initial epochs are 100, however I found out that only 30 would be enough for this perpose. Adam Optimizer method is used to minimize the loss function. The leaning rate is set to 0.0001. I was inspired by the explanations of the other sudents, when completing this part of the code.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used the Lenet architecture which was introduced in the previous lectures, due to its successful functionality. 
The final model acuracy are as follows:

* validation set accuracy of 0.968 
* test set accuracy of 0.947

which proves the that the model work well.

## Step 5: Testing the model on mew images

### Load and Output the Images

Here are five German traffic signs that I saved and loaded from the web:

![./newImg/1.png][image1] ![./newImg/2.png][image2] ![./newImg/3.png][image3] 
![./newImg/4.png][image4] ![./newImg/5.png][image5]

I did the same pre-processing steps on these images. Here are the final images:

![./sample/new_images.png][new_images_after_pre_processing]


### Accuracy and top 5 softmax probabilities of new images

The prediction accuracy turened out to be 80%, predicting 4 signs correctly. The model only failed to predict the "STOP" sign. Bellow is a table showing the actual images and the predictions:

| Image									|     Prediction	   					| Probability         	| 
|:-------------------------------------:|:-------------------------------------:|:---------------------:|
| Stop Sign      						| Speed limit (70km/h) 					|9.99790251e-01			| 
| Road work    							| Road work 							|1.0         			|
| Turn right ahead						| Turn right ahead						|1.0         			|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|1.0         			|
| Speed limit (70km/h)					| Speed limit (70km/h)      			|1.0         			|



### Detailed analysis for each new image

The first image is the only image from the five that is not predicted correctly. The top five softmax probabilities for the first image are as follows, whereas the actual sign is a STOP sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99790251e-01		| Speed limit (70km/h)							| 
| 2.09811056e-04		| Speed limit (30km/h)							|
| 9.88755744e-11		| Priority road									|
| 1.24509006e-11		| Speed limit (50km/h)			 				|
| 7.72891679e-13		| Speed limit (120km/h)    						|


The rest of the predictions are 100% correct in predicting the actual signs, which are Road work, Turn right ahead, Right-of-way at the next intersection, and Speed limit (70km/h), respectively.



