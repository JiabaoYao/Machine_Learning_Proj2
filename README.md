# Machine_Learning_Proj2
* mnist all.mat : original dataset from MNIST. In this file, there are 10 matrices for testing set and 10 matrices for training set, which corresponding to 10 digits. You will have to split the training data into training and validation data.
* face all.pickle: sample of face images from the CelebA data set. In this file there is one data matrix and one corresponding labels vector. The preprocess routines in the script files will split the data into training and testing data.
* nnScript.py: Python script for this programming project. Contains function definitions -
– preprocess(): performs some preprocess tasks, and output the preprocessed train, validation and
test data with their corresponding labels. You need to make changes to this function.
* facennScript.py: Python script for running your neural network implementation on the CelebA dataset. This function will call your implementations of the functions sigmoid(), nnObjFunc() and nnPredict() that you will have to copy from your nnScript.py. You need to make changes to this function.
