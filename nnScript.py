import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import numpy as np
import pickle
import time

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return (1 / (1 + np.exp(-z)))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    # mat = loadmat('/Users/jiabaoyao/Study Abroad/Projects/Machine Learning/Proj_2/Machine_Learning_Proj2/mnist_all.mat')  # loads the MAT object as a Dictionary
    mat = loadmat('D:\Courses\Machine learning\Machine_Learning_Proj2\mnist_all.mat')

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    diff_cols = np.diff(train_data, axis = 0) # calculate the difference between adjacent pixels
    diff_select = np.any(diff_cols, axis = 0) # select the columns with difference (boolean array)

    train_data = train_data[:, diff_select] # selects only the columns (features) in train_data that have variation
    selected_indices = np.where(diff_select)[0] # stores the indices of the selected features, which is then used to filter validation_data and test_data to ensure consistency
    validation_data = validation_data[:, selected_indices]
    test_data = test_data[:, selected_indices]

    # Save selected indices
    # np.savetxt('selected_indices.txt', selected_indices, fmt = '%d')

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    n = training_data.shape[0] # n examples

    # Add bias to training_data
    training_data = np.hstack((training_data, np.ones((n, 1))))

    # Forward propagation and add bias to hidden layer
    z = sigmoid(np.dot(training_data, w1.T))
    z = np.hstack((z, np.ones((z.shape[0], 1))))

    # Output layer activation
    o = sigmoid(np.dot(z, w2.T))

    # Training label encoding by one-hot
    y = np.zeros((n, n_class))
    y[np.arange(n), training_label.astype(int)] = 1 # shape is (n examples, n_class)

    # Likelihood error function with regularization
    log_likelihood = -np.sum(y * np.log(o) + (1 - y) * np.log(1 - o)) / n
    regularization = (lambdaval / (2 * n)) * (np.sum(w1**2) + np.sum(w2**2))
    # Compute the regularized objective function
    obj_val = log_likelihood + regularization

    # Backprogation
    delta = o - y # error term for output layer
    grad_w2 = (np.dot(delta.T, z) / n) + (lambdaval / n) * w2

    # Error term for hidden layer
    hidden_error = (1 - z) * z * np.dot(delta, w2)
    hidden_error = hidden_error[:, :-1] # remove bias term

    grad_w1 = (np.dot(hidden_error.T, training_data) / n) + (lambdaval / n) * w1

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])

    # Add bias to input layer
    data = np.hstack((data, np.ones((data.shape[0], 1))))

    # Forward propagation to hidden layer
    z = sigmoid(np.dot(data, w1.T))
    z = np.hstack((z, np.ones((z.shape[0], 1)))) # add bias

    # Forward propagation to output layer
    o = sigmoid(np.dot(z, w2.T))

    labels = np.argmax(o, axis = 1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in output unit
n_class = 10

# set the number of nodes in hidden unit (not including bias unit)
# n_hidden_range = [i for i in range(0, 60, 4)]
n_hidden_range = [i for i in range(40, 56, 2)]


# set the regularization hyper-parameter
# lambdaval_range = [i for i in range(0, 60, 10)]
lambdaval_range = [i for i in range(0, 30, 5)]

best_n_hidden, best_lambdaval = 0, 0
best_accuracy = 0
best_w1, best_w2 = None, None
training_times = {n: [] for n in n_hidden_range}
training_accuracy_list = []

for n_hidden in n_hidden_range:
    for lambdaval in lambdaval_range:
        print(f"Training with n_hidden = {n_hidden} and lambdaval = {lambdaval}")

        start_time = time.time()
        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)


        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
        opts = {'maxiter': 50}  # Preferred value.
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        # Test the computed parameters
        predicted_label = nnPredict(w1, w2, train_data)
        # find the accuracy on Training Dataset
        print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

        predicted_label = nnPredict(w1, w2, validation_data)
        # find the accuracy on Validation Dataset
        accuracy = 100 * np.mean((predicted_label == validation_label).astype(float))
        print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

        training_accuracy_list.append([n_hidden, lambdaval, accuracy])

        # End time
        end_time = time.time()
        training_time = end_time - start_time
        training_times[n_hidden].append((lambdaval, training_time))
        print(f"Traing time for n_hidden = {n_hidden} and lambdaval = {lambdaval}: {training_time}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n_hidden = n_hidden
            best_lambdaval = lambdaval
            best_w1 = w1
            best_w2 = w2

print(f"Best parameters: n_hidden = {best_n_hidden}, lambdaval = {best_lambdaval}")
print(f"Best validation accuracy: {best_accuracy}")

predicted_label = nnPredict(best_w1, best_w2, test_data)
# find the accuracy on Validation Dataset
print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

# Save the training times in a list format compatible with np.savetxt
training_time_list = []
for n_hidden, times in training_times.items():  # Use .items() to iterate over dictionary items
    for lambdaval, training_time in times:
        training_time_list.append([n_hidden, lambdaval, training_time])

# Save the data to text files
np.savetxt('nn_training_time_record1.txt', training_time_list, fmt='%d, %d, %.4f', header='n_hidden, lambdaval, training_time')
np.savetxt('nn_training_accuracy_record1.txt', training_accuracy_list, fmt='%d, %d, %.4f', header='n_hidden, lambdaval, accuracy')