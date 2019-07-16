# FlipkartGRid-ObjLocalization
Flipkart Object Localization Challenge: Where we trained our data to learn localizing an object in an image


Flipkart Object Localization Challenge:


Initially, seeing the problem statement we thought of using traditional object detection techniques like YOLO or FAST-RCNN, but then seeing the training data we assumed that it would suffice to develop our own network, using some simple intuition.


Our Idea & why it works:
We trained the whole Resnet-50 CNN changing the final fc layer from 1000 classes to four classes on the given dataset. As resnet requires 224*224 as the image size, we have resized all the images to the required size.


The final layer outputs four values as mentioned in the problem statement. As they are continuous values, we used a different loss function which is Mean Square Loss function for this problem.


As this is a subset of the detections that an object detector can do, here we have to localize the object, if there are multiple objects all of them have to be localized and should have only one box around them. Hence we used this techquine, object detector might have failed in this case.

Implementation process:
We used pytorch-Deep learning libraries for implementation. We did our training on Google Colab system.


Firstly, the images have been separated based on names in training.csv and test.csv from the given dataset and converted them from ( 640 x 480 ) to ( 224 x 224 )( in preprocessing.py ) and scaling has been done accordingly for the coordinates of the bounding box.


Afterwards, we have divided the dataset into training set and validation set( for setting parameters). We trained the VGG-16 network(randomly initialized) on the training set and checked on validation set( train_validation.py). Finally after setting the parameters we trained the network on entire available dataset( train + validation) ( final.py).

Have Achieved a score of 0.89 IoU.
