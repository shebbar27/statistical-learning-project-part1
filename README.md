# cse569-fsl-project-phase1
A simple project implementing Feature Extraction, Density Estimation and Bayesian Classification on a dataset which is subset of images (with modifications) from the MNIST dataset.

The original MNIST dataset (http://yann.lecun.com/exdb/mnist/) contains
70,000 images of handwritten digits, divided into 60,000 training images and 10,000
testing images. Images for only digit “3” and digit “7” is used in this project. The data is
stored in “.mat” files.

Following are the statistics for the dataset:
Number of samples in the training set: "3": 5713; "7": 5835
Number of samples in the testing set : "3": 1428; "7": 1458

The following three tasks are implmented in this project:

Task 1. Feature extraction and normalization
In the .mat file, each image is stored as a 28x28 array of pixels. For each image i, compute
two features: the mean m i and the standard deviation s i of the 784 pixels. We further
normalize the features in the following way, before starting any subsequent tasks. Using
the feature representations of all the training images from both classes (each image i is
now viewed as a 2-d vector, X i = [m i , s i ] t , as explained), compute the mean M j and
standard deviation S j , j=1,2, for the first and the second feature, respectively. M j and S j
will be used to normalize all the feature vectors (both training and testing): for each feature
vector X i representing image i, a normalized feature vector Y i will be computed as
Y i = [ y 1i , y 2i ] t = [ (m i – M 1 )/S 1 , (s i – M 2 )/S 2 ] t
This Y i is the final feature representation for image i, and will be used for subsequent
steps.

Task 2. Density estimation
We assume in the 2-d feature space of Y i defined above, samples from each class follow
a normal distribution. Using the MLE method, you will need to estimate the parameters
for the 2-d normal distribution for each class/digit, using the respective training data
for that class/digit. Note: You will have two distributions, one for each digit

Task 3. Bayesian Decision Theory for optimal classification
Use the estimated distributions for doing minimum-error-rate classification, for the
following two cases respectively:
Case 1: Assume that the prior probabilities are the same (i.e., P(3) = P(7) =0.5).
Case 2: Assume that the prior probabilities are: P(3) = 0.3, P(7) = 0.7.
For both cases, report the probability of error of the optimal classifier, for the training set
and the testing set respectively.
