################################################################################
#
# LOGISTICS
#
#    Name: Dinakar Kondru
#    UTD ID: 2021495823
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#    1. A summary of my nn.py code:
#
#       <Overall Summary: 
#        nn.py has 3 main sections: 1. Header, 2. XNN Class and 3. Main function
#        1. Header: It handles imports, data download, populate numpy arrays
#        2. XNN class: It has definitions for these functions-initialize,forward_pass,compute_loss,backprop,compute_acc and plot_acc
#        3. Main function calls the functions in XNN class as per project instructions.
#       >
#       <Forward path: 
#        Function      : xnn.forward_pass() 
#        Input         : 1 vectorized training image (TRAIN_BATCH_SIZE=TEST_BATCH_SIZE=1). Size = 1x784
#        Output        : Probability distribution of output class predictions(Size = 1x10) and saved parameters
#        Functionality : 1 vectorized image is taken as the input and sent through series of 
#                        Matrix-Multiply(linear), Addition(linear) and ReLU(non-linear) layers.
#                        The final layer is the softmax layer which normalizes the final prediction
#                        vector into corresponding probability distribution.
#       >
#       <Error code:
#        Function      : xnn.compute_loss()
#        Input         : Output of the softmax layer from forward path and Corresponding training label converted to one-hot encoding
#        Output        : Error
#        Functionality : The function compares the predicted and expected labels and computes loss or 
#                        error using categorical cross-entropy i,e -expected*log(predicted)
#       >
#       <Backward path:
#        Function      : xnn.backprop() 
#        Input         : Saved parameters
#        Output        : Saved derivatives of error with respect to weights and biases for each layer
#        Functionality : Derivative of the error w.r.t final output of forward path is first computed. Using the 
#                        fact that X(n) = H(n-1)*X(n-1) and applying chain rule of derivatives, and working backwards, 
#                        compuute the derivatives of error w.r.t weights and biases for each layer. 
#       >
#       <Weight update:
#        Function      : xnn.update_params()
#        Input         : Learning rate and saved derivatives from backprop function
#        Output        : Nothing returned but globally saved params are updated
#        Functionality : The function accesses the exiting parameters and updates them using gradient descent method:
#                        H = H - lr * de/dH
#       >
#
#    2. Accuracy display
#
#       ====================================================================================
#       Epoch    Execution time(s)   Training Loss   Training Accuracy(%)   Test Accuracy(%)
#       ====================================================================================
#        1             399.7              0.014              89.97             93.71
#        2             359.0              0.008              95.08             95.44
#        3             343.9              0.004              96.56             96.27
#        4             334.2              0.002              97.50             96.89
#        5             323.1              0.001              98.05             97.26
#        6             319.5              0.001              98.42             97.43
#        7             316.1              0.000              98.74             97.59
#        8             311.3              0.000              99.03             97.72
#        9             311.1              0.000              99.26             97.89
#        10            314.4              0.000              99.45             97.87
#       ===================================================================================
#       Final test accuracy =  97.87 percent
#       ===================================================================================
#
#    3. Performance display
#
#       ================================================================================
#       Total Execution time =  55.6 minutes  
#       ================================================================================
#       ================================================================================
#       Layer                  Input size         Output size       Parameter size     
#       ================================================================================
#       Data                   60000x28x28        1x28x28              0             
#       Vectorization          1x28x28            1x784                0             
#       Matrix Mult-1          1x784              1x1000               784x1000        
#       Addition-1             1x1000             1x1000               1x1000          
#       ReLu-1                 1x1000             1x1000               0             
#       Matrix Mult-2          1x1000             1x100                1000x100        
#       Addition-2             1x100              1x100                1x100           
#       ReLu-2                 1x100              1x100                0             
#       Matrix Mult-3          1x100              1x10                 100x10          
#       Addition-3             1x10               1x10                 1x10            
#       Softmax                1x10               1x10                 0              
#       ================================================================================
#
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

#
# you should not need any import beyond the below
# PyTorch, TensorFlow, ... is not allowed
#
import os.path
import urllib.request
import gzip
import math
import numpy             as np
import matplotlib.pyplot as plt
import time
################################################################################
#
# PARAMETERS
#
################################################################################

#
# add other hyper parameters here with some logical organization
#

# To compute code execution time 
code_start = time.time()

# data

DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'
NUM_EPOCHS             = 10
TRAIN_BATCH_SIZE       = 1
TEST_BATCH_SIZE        = 1
NUM_TRAIN_BATCHES      = DATA_NUM_TRAIN//TRAIN_BATCH_SIZE
NUM_TEST_BATCHES       = DATA_NUM_TEST//TEST_BATCH_SIZE

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################


# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
   urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
   urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
   urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
   urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

# Class that has all the functions defined 
class XNN:

    # constructor
    def __init__(self, lyr_sizes):
        self.lyr_sizes        = lyr_sizes
        self.params           = {}
        self.num_lyrs         = len(self.lyr_sizes)
        self.num_samples      = 0
        self.train_accuracies = []
        self.test_accuracies  = []
        self.losses           = []

    # initialize function initializes parameters and other data structures
    def initialize(self, X_shape):
        self.num_samples = X_shape[0]
        self.lyr_sizes.insert(0, X_shape[1])
        # Initialize all trainable params - weights and biases
        for lyr in range(1, len(self.lyr_sizes)):
            # Shape of H is output layer sizexinput layer size. Randomize this shape div-by square-root of size of input layer
            self.params["H" + str(lyr)] = np.random.randn(self.lyr_sizes[lyr], self.lyr_sizes[lyr - 1]) / np.sqrt(self.lyr_sizes[lyr - 1])
            self.params["v" + str(lyr)] = np.zeros((self.lyr_sizes[lyr], 1))
        # Initilize Plot at 0% accuracy and 100% Loss
        self.losses.append(100)
        self.train_accuracies.append(0)
        self.test_accuracies.append(0)
 
    # forward_pass function takes in input and sends it through all the network layers
    def forward_pass(self, X):
        mem = {}
        # Loop(Matrix-multiply + Bias-Addition + ReLU)
        for lyr in range(self.num_lyrs - 1):
            # Matrix mutliplication layer: Z(LAYER_OUTPUT_SIZE x TRAIN_BATCH_SIZE) = H(LAYER_OUTPUT_SIZE x 784).X(784 x TRAIN_BATCH_SIZE)
            Z = self.params["H" + str(lyr + 1)].dot(X) 
            # Bias addition layer
            Z = Z + self.params["v" + str(lyr + 1)]
            # ReLU Layer
            X = np.maximum(Z, 0)
            # Save X, H, Z for each layer
            mem["X" + str(lyr + 1)] = X
            mem["H" + str(lyr + 1)] = self.params["H" + str(lyr + 1)]
            mem["Z" + str(lyr + 1)] = Z
     
        # Classification layer matrix multiply
        Z = self.params["H" + str(self.num_lyrs)].dot(X) 
        # Classification layer bias addition
        Z = Z + self.params["v" + str(self.num_lyrs)]
        
        # Softmax layer
        X = (np.exp(Z - np.max(Z)))/(np.exp(Z - np.max(Z))).sum(axis=0, keepdims=True)
        
        # Save X, H, Z for classification layer
        mem["X" + str(self.num_lyrs)] = X
        mem["H" + str(self.num_lyrs)] = self.params["H" + str(self.num_lyrs)]
        mem["Z" + str(self.num_lyrs)] = Z
        # Return final output of forward pass and the saved parameters
        return X, mem

    # backprop function computes the derivatives required to update the paramaters of trainable layers, going backwards
    def backprop(self, X, Y, mem):
        derivs = {}
        
        # compute and save the derivates for classification layer
        mem["X0"] = X
        X = mem["X" + str(self.num_lyrs)]
        dZ = X - Y
        # Based on calculus cookbook, derivative of error wrt weights = (x.dz)/n
        # and derivative of error wrt biases = sum(dz)/n
        dH = dZ.dot(mem["X" + str(self.num_lyrs - 1)].T) / self.num_samples
        dv = np.sum(dZ, axis=1, keepdims=True) / self.num_samples
        dXPrev = mem["H" + str(self.num_lyrs)].T.dot(dZ)
        derivs["dH" + str(self.num_lyrs)] = dH
        derivs["dv" + str(self.num_lyrs)] = dv

        # compute and save derivative of trainable layers going backwards
        for lyr in range(self.num_lyrs - 1, 0, -1):
            ReLU_derivative = mem["Z" + str(lyr)]
            ReLU_derivative[ReLU_derivative<=0] = 0
            ReLU_derivative[ReLU_derivative>0]  = 1
            dZ = dXPrev * ReLU_derivative
            # Based on calculus cookbook, derivative of error wrt weights = (x.dz)/n
            # and derivative of error wrt biases = sum(dz)/n
            dH = (1.0 / self.num_samples) * dZ.dot(mem["X" + str(lyr - 1)].T)
            dv = (1.0 / self.num_samples) * np.sum(dZ, axis=1, keepdims=True)
            if lyr > 1: # no need to backpropagate beyond 1st layer
                dXPrev = mem["H" + str(lyr)].T.dot(dZ)
            derivs["dH" + str(lyr)] = dH
            derivs["dv" + str(lyr)] = dv

        return derivs

    # compute_loss function computes the loss for predicted vs. expected output  
    def compute_loss(self, Y_pred, Y_expect):
        # add small number to avoid log(0)
        loss = -np.mean(Y_expect * np.log(Y_pred.T + 0.0000000001))
        return loss

    # update_params function updates the parameters of trainable layers based on backprop derivatives
    def update_params(self,lr,derivs):
        for lyr in range(1, self.num_lyrs + 1):
            self.params["H" + str(lyr)] = self.params["H" + str(lyr)] - lr * derivs["dH" + str(lyr)]
            self.params["v" + str(lyr)] = self.params["v" + str(lyr)] - lr * derivs["dv" + str(lyr)]

    # compute_acc function computes training and test accuracy
    def compute_acc(self,X,Y):
        y_pred = np.argmax(X, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_pred == Y).mean()
        return accuracy

    # plot_acc function plots training accuracy, test accuracy, loss vs. epochs
    def plot_acc(self):
        plt.figure(figsize=(20, 10), dpi=70)
        plt.plot(np.arange(len(self.train_accuracies)), self.train_accuracies, "-b", label='Training Accuracy')
        plt.plot(np.arange(len(self.test_accuracies)), self.test_accuracies, "-g", label='Test Accuracy')
        plt.plot(np.arange(len(self.losses)), self.losses, "-r", label='Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy and Loss")
        plt.legend(loc="lower right")
        plt.show()

if __name__ == '__main__':

   print("====================================================================================")
   print("Epoch    Execution time(s)   Training Loss   Training Accuracy(%)   Test Accuracy(%)")
   print("====================================================================================")

   # Convert lables to one-hot encoding for math convenience
   train_y = np.zeros((train_labels.size, train_labels.max()+1))
   train_y[np.arange(train_labels.size),train_labels] = 1
   test_y = np.zeros((test_labels.size, test_labels.max()+1))
   test_y[np.arange(test_labels.size),test_labels] = 1

   # Division by 255.0 layer
   train_x     = train_data / 255.0
   test_x      = test_data / 255.0
    
   # Vectorization layer
   train_x  = train_x.reshape(DATA_NUM_TRAIN, DATA_ROWS*DATA_COLS)
   test_x   = test_x.reshape(DATA_NUM_TEST, DATA_ROWS*DATA_COLS)
   
   # Create matrix multiplication layers, ReLU layers and Classification layer
   # Pass dims of layers with trainable parameters and classification layer 
   # Layer1 = 1x1000, Layer2 = 1x100, Classification Layer = 1xDATA_CLASSES
   xnn = XNN([1000, 100, DATA_CLASSES]) 
   
   # Initialize params and other data structures
   xnn.initialize(train_x[0:TRAIN_BATCH_SIZE].shape)

   # Cycle through epochs
   for epoch in range(NUM_EPOCHS):
  
     # To compute execution time per epoch
     epoch_start = time.time()
   
     # set learning rate proportional to the batch size
     lr = 0.001

     #initialize train and test accuracy to zero
     train_acc = 0
     test_acc  = 0
     
     #for each epoch cycle through the training data
     for batch in range(NUM_TRAIN_BATCHES):
       #print("train_x=", train_x)
       #slice training data into batches
       batch_start_index = batch*TRAIN_BATCH_SIZE
       batch_end_index   = batch_start_index + TRAIN_BATCH_SIZE
       train_x_slice     = train_x[batch_start_index:batch_end_index]
       train_y_slice     = train_y[batch_start_index:batch_end_index]

       # Forward Pass
       X, mem        = xnn.forward_pass(train_x_slice.T)
       
       # Compute loss 
       loss          = xnn.compute_loss(X,train_y_slice)
       
       # Backprop
       derivs       = xnn.backprop(train_x_slice.T, train_y_slice.T, mem)
       
       # Weight update
       update_params = xnn.update_params(lr,derivs)
       
       # Compute training accuracy
       train_acc     = train_acc + xnn.compute_acc(X,train_y_slice)
   
     # For each epoch cycle through the testing data
     for batch in range(NUM_TEST_BATCHES):
       
       # For each batch get the corresponding slice of test data
       batch_start_index = batch*TEST_BATCH_SIZE
       batch_end_index   = batch_start_index + TEST_BATCH_SIZE
       test_x_slice      = test_x[batch_start_index:batch_end_index]
       test_y_slice      = test_y[batch_start_index:batch_end_index]

       # Do a forward pass of test data      
       X, mem_test       = xnn.forward_pass(test_x_slice.T)
 
       # Compute test accuracy
       test_acc          = test_acc + xnn.compute_acc(X,test_y_slice)
  
     # Save loss, training and test accuracies per epoch for plotting
     xnn.losses.append(loss)
     xnn.train_accuracies.append(train_acc*100/NUM_TRAIN_BATCHES)
     xnn.test_accuracies.append(test_acc*100/NUM_TEST_BATCHES)
     
     # To compute execution time per epoch
     epoch_end = time.time()
     
     # For each epoch display loss and accuracy information
     print("  %d             %.1f              %.3f              %.2f             %.2f" % 
     (epoch+1, (epoch_end-epoch_start), loss, train_acc*100/NUM_TRAIN_BATCHES, test_acc*100/NUM_TEST_BATCHES))

   # Accuracy display:
   
   # Final value 
   print("================================================================================")
   print(" Final test accuracy =  %.2f percent" % (test_acc*100/NUM_TEST_BATCHES))
   print("================================================================================")
   
   # Plot of accuracy vs. epoch
   print("================================================================================")
   print(" Plot of Accuracy vs. Epoch")
   print("================================================================================")
   xnn.plot_acc()

   # Performance display:

   # To compute code execution time
   code_end = time.time()
   
   # Total time
   print("================================================================================")
   print(" Total Execution time =  %.1f minutes  " % ((code_end-code_start)/60))
   print("================================================================================")
  
   # Per layer info - type, input size, output size, parameter size, MACs...
   print("================================================================================")
   print(" Layer                  Input size         Output size       Parameter size     ")
   print("================================================================================")
   print(" Data                   60000x28x28        1x28x28                0             ")
   print(" Vectorization          1x28x28            1x784                  0             ")
   print(" Matrix Mult-1          1x784              1x1000               784x1000        ")
   print(" Addition-1             1x1000             1x1000               1x1000          ")
   print(" ReLu-1                 1x1000             1x1000                 0             ")
   print(" Matrix Mult-2          1x1000             1x100                1000x100        ")
   print(" Addition-2             1x100              1x100                1x100           ")
   print(" ReLu-2                 1x100              1x100                  0             ")
   print(" Matrix Mult-3          1x100              1x10                 100x10          ")
   print(" Addition-3             1x10               1x10                 1x10            ")
   print(" Softmax                1x10               1x10                   0             ")

