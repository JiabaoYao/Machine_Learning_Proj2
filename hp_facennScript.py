'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import time
import numpy as np
import pickle
from math import sqrt
from scipy.optimize import minimize

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return (1 / (1 + np.exp(-z)))
# Replace this with your nnObjFunction implementation
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
    regularization = (lambdaval / (2 * n)) * (np.square(w1).sum() + np.square(w2).sum())
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
    
# Replace this with your nnPredict implementation
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
# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)

n_hidden_range = [i for i in range(10, 200, 10)]


# set the regularization hyper-parameter
# lambdaval_range = [i for i in range(0, 60, 10)]
lambdaval_range = [i for i in range(0, 100, 10)]


best_n_hidden, best_lambdaval = 0, 0
best_accuracy = 0
best_w1, best_w2 = None, None
training_times = {n: [] for n in n_hidden_range}
training_accuracy_list = []

for n_hidden in n_hidden_range:
    for lambdaval in lambdaval_range:


        # set the number of nodes in output unit
        n_class = 2

        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden);
        initial_w2 = initializeWeights(n_hidden, n_class);
        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
        # set the regularization hyper-parameter
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        #Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
        opts = {'maxiter' :50}    # Preferred value.

        start_time = time.time()
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
        params = nn_params.get('x')
        #Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
        w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        #Test the computed parameters
        predicted_label = nnPredict(w1,w2,train_data)
        #find the accuracy on Training Dataset
        training_accuracy = 100*np.mean((predicted_label == train_label).astype(float))
        print('\n Training set Accuracy:' + str(training_accuracy) + '%')
        predicted_label = nnPredict(w1,w2,validation_data)
        #find the accuracy on Validation Dataset
        accuracy = 100*np.mean((predicted_label == validation_label).astype(float))
        print('\n Validation set Accuracy:' + str(accuracy) + '%')
            
        training_accuracy_list.append([n_hidden, lambdaval, accuracy, training_accuracy])
        end_time = time.time()
        training_time = end_time - start_time
        training_times[n_hidden].append((lambdaval, training_time))
       
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n_hidden = n_hidden
            best_lambdaval = lambdaval
            best_w1 = w1
            best_w2 = w2
        predicted_label = nnPredict(w1,w2,test_data)
        #find the accuracy on Validation Dataset
        print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

predicted_label = nnPredict(best_w1, best_w2, test_data)
# find the accuracy on Validation Dataset
print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

# Save the training times in a list format compatible with np.savetxt
training_time_list = []
for n_hidden, times in training_times.items():  # Use .items() to iterate over dictionary items
    for lambdaval, training_time in times:
        training_time_list.append([n_hidden, lambdaval, training_time])

# Save the data to text files
np.savetxt('face_nn_training_time_record_bigrange.txt', training_time_list, fmt='%d, %d, %.4f', header='n_hidden, lambdaval, training_time')
np.savetxt('face_nn_training_accuracy_record_bigrange.txt', training_accuracy_list, fmt='%d, %d, %.4f', header='n_hidden, lambdaval, accuracy')