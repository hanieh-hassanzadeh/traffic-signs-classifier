# **Traffic Sign classification and Recognition**

For this project, I performed bellow steps:

## Step 0: Load The Data

First off, I load the data from the corresponding files, and create the training, validation, and test feature and label sets.


## Step 1: Data exploration
Exploring the data sets reveals that, 

Number of training examples   = 34799

Number of validation examples = 4410

Number of testing examples    = 12630

Image data shape  = (32, 32, 3)

Number of classes = 43

### Visualization of the training dataset

A sample image of each 43 sign types is shown as the first visualization practice:

![image](https://github.com/hanieh-hassanzadeh/traffic-signs-classifier/blob/master/examples/visul1.png)

The following histogram demonstrates how many images of each sign type exists in the training data set.

![histogram](https://github.com/hanieh-hassanzadeh/traffic-signs-classifier/blob/master/examples/visul2.png)


## Step 2: Data augmentation

The first analyses shows that the data size is not big enough. Moreover, the images needed more generalization. Therefore, two augmentation techniques are performed on training set, which increase the dataset size from 34799 to 626382. These techniques are %80 central scaling and 7 rotations within the interval of [-18, 18] degrees. 


## Step 3: Pre-Processing by gray-scaling and normalization

Then the images are converted into gray scales and the image arrays are normalized from [0, 255] intervals to the standard form of [-1, 1] interval.


### Prepare data to be fed into Tensorflow functions

The array shapes of all three features datasets needed to be changed to have another dimension of size 1 as the last dimension (to be compatible with Tensorflow functions).


### Define Tensorflow variables

Then, I defin the variables needed in Tensorflow functions.


## Step 4: Modeling

### Model architecture

I use LeNEt architecture which consists of:

Layer 1: Convolutional - Input: 32x32x1, Output: 28x28x6
    Activation
    Pooling - Input: 28x28x6, Output: 14x14x6
    
Layer 2: Convolutional - Output: 10x10x16
    Activation
    Pooling - Input: 10x10x16, Output: 5x5x16
    Flatten - Input: 5x5x16, Output: 400
    
Layer 3: Fully Connected - Input: 400, Output: 120
    Activation
    
Layer 4: Fully Connected - Input: 120, Output: 84
    Activation
    
Layer 5: Fully Connected - Input: 84, Output: 43


### Model properties

To train the model, I consider the batch size of 300, due to a larger dataset size. The initial epochs are 100, however I found out that only 30 would be enough for this purpose. Adam Optimizer method is used to minimize the loss function. The leaning rate is set to 0.0001. 

#### 4. Approach towards finding a solution

In the beginning of each epoch, I shuffled the data to avoid any bias owing to the same data order. Then, I divided the training set into the batches. 
For the training purpose, first, I extracted logits by running a Lenet architecture. Second, I calculated cross entropy using softmax_cross_entropy_with_logits function. Third, the loss is defined by the mean of cross entropy function. Fourth, I Used Adam Optimizer function to minimize the loss. 
At the end of each epoch, I calculate the accuracy of the predictions  on validation set. The model is saved frequently throughout the training to save time when a disruption occurs. 
When the number of epochs is satisfied, I calculate the accuracy on the test set. 


It's worth noting that in the beginning, the accuracy was lower and the model runtime was slower. By looking at the histogram (shown above), I realized that for some sings the number of examples in the training set is very low, therefore I increased the the examples by augmentation methods. At the same time augmentation helped to generalize the data set. Also, I increased the batch size to make the model runtime faster.

The final model accuracies are as follows:

* validation set accuracy of 0.968 
* test set accuracy of 0.947

which proves the that the model work well.


## Step 5: Testing the model on mew images

### Load and Output the Images

Here are five German traffic signs that I saved and loaded from the web:

<img src="https://github.com/hanieh-hassanzadeh/traffic-signs-classifier/blob/master/newImg/1.png" height=100>      <img src="https://github.com/hanieh-hassanzadeh/traffic-signs-classifier/blob/master/newImg/2.png" height=100>      <img src="https://github.com/hanieh-hassanzadeh/traffic-signs-classifier/blob/master/newImg/3.png" height=100>      <img src="https://github.com/hanieh-hassanzadeh/traffic-signs-classifier/blob/master/newImg/4.png" height=100>      <img src="https://github.com/hanieh-hassanzadeh/traffic-signs-classifier/blob/master/newImg/5.png" height=100>          

I performed the same pre-processing steps on these images. Here are the final images after pre-processing:

![new_images_after_pre_processing](https://github.com/hanieh-hassanzadeh/traffic-signs-classifier/blob/master/examples/new_images.png)


There are some conditions that may make the prediction of the new images a bit challenging. 
- Most of the images that I found on the web are brighter than the dataset images.
- The new images could have different orientations/rotation than the images in the dataset, due to misplacements of the signs on the streets.
- The new images were sometimes very zoomed in or zoomed out.
- The new signs were only a small part of a bigger image.
_ Some images on the web are distorted, due to the angle of the camera.


To overcome these challenges, usually the data augmentation methods (such as scaling, rotation, etc.) can help to generalize the training set and lead towards more accurate predictions. 


### Accuracy and top 5 softmax probabilities of new images

The prediction accuracy turned out to be 80%, predicting 4 signs correctly. The model only failed to predict the "STOP" sign. Bellow is a table showing the actual images and the predictions:

| Image									|     Prediction	   					| Probability         	| 
|:-------------------------------------:|:-------------------------------------:|:---------------------:|
| Stop Sign      						| Speed limit (70km/h) 					|9.99790251e-01			| 
| Road work    							| Road work 							|1.0         			|
| Turn right ahead						| Turn right ahead						|1.0         			|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|1.0         			|
| Speed limit (70km/h)					| Speed limit (70km/h)      			|1.0         			|

Although the test set accuracy was 0.947, here we have only 0.80 of accuracy on the new images. The reason lays on the very small new dataset size (of 5). That is, even one wrong prediction can lower the accuracy 20% down.   



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
