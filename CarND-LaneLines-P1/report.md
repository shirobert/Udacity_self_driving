
**Finding Lane Lines on the Road**
In the first project of the udacity self-driving car course we had to find lane markings in pictures and videos of streets.

### Reflection

### 1.There are three steps in the pipeline:

1. Use a gaussian kernel to filter the image
    * Getting rid of noisy parts of the image 
2. Perform canny edge detection
    * detects the edges in the image with the help of the image gradient and hysteris.
3. Use hough transformation to find lines from the edges
    * Transforms each point to a line in hough space where the intersection of these lines shows the presence of a line in image space
    
[//]: # (Original)

[image1]: ./test_images/solidWhiteCurve.jpg
[image2]: ./test_images_out/solidWhiteCurve.jpg
Original
![Original][image1]
Result
![result][image2]
More results and examples can be seen in the iphone notebook.

For drawline function, I set the range of the slope and put the small parts into a list as preprocessing.
Find the average slope of the beginning part and end part of the line. Then based on the slope and begining part and end part to draw a continuesly line to avoid jitters. 


### 2. Identify potential shortcomings with your current pipeline


1.There are some parameters are manual tunned which may not cover all the cases.
2.If the backgound is too noisy, the algorithm may fail. The current pipeline is for the easiest situation.

### 3. Suggest possible improvements to your pipeline

1. Use automatic way to propose the mask region
