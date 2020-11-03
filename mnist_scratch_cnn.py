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
#    1. A summary of my cnn.py code:
#
#       <Overall Summary: 
#        cnn.py has 4 main sections: 1. Header 2. CNN_LAYER class 3. MAXPOOL_LAYER Class  4. XNN Class  5. Main function
#        1. Header: It handles imports, data download, populate numpy arrays
#        2. CNN_LAYER class: It handles CNN2dConv, Addition and ReLU and has the functions-initialize, forward_pass, backprop and update_params.
#        3. MAXPOOL_LAYER Class: It handles maxpool operation and has the functions-initialize, forward_pass, backprop.
#        4. XNN Class: It handles the fully connected layer's Matrix-multiply,Addition,ReLU,Softmax and has the functions-initialize, forward_pass, compute_loss, backprop, update_params, compute_acc 
#        5. Main function calls the functions in CNN_LAYER, MAXPOOL_LAYER and XNN classes, following the project instructions.
#       >
#       <Forward path: 
#        Functions     : cnn.forward_pass, maxpool.forward_pass, xnn.forward_pass 
#        Functionality : 1x28x28 image is fed to cnn.forward_pass which perform cnn2dConv + bias + relu. The output of cnn.forward_pass (16x28x28) is fed maxpool.forward_pass 
#                        which reduces the size to 16x14x14. This loop (increasing output channels and decreasing heightxwidth) is repeated until the image size is 64x7x7.
#                        At this point, the image is vectorized to 1x3136. The vectorized image is fed to xnn.forward_pass and goes through a couple of matrix-mulitply+Bias+ReLu
#                        loops. The final layer is the softmax layer which normalizes the final prediction vector into corresponding probability distribution.
#       >
#       <Error code:
#        Function      : xnn.compute_loss
#        Input         : Output of the softmax layer from forward path and Corresponding training label converted to one-hot encoding
#        Output        : Error
#        Functionality : The function compares the predicted and expected labels and computes loss or 
#                        error using categorical cross-entropy i,e -expected*log(predicted)
#       >
#       <Backward path:
#        Function      : xnn.backprop, maxpool.backprop, cnn.backprop
#        Input         : Saved parameters
#        Output        : Saved derivatives of error with respect to weights and biases for each layer
#        Functionality : Derivative of the error w.r.t final output of forward path is first computed. Using the 
#                        fact that X(n) = H(n-1)*X(n-1) and applying chain rule of derivatives, and working backwards, 
#                        derivatives of error w.r.t weights and biases for each layer until it reaches the first CNN layer.
#       >
#       <Weight update:
#        Function      : cnn.update_params, xnn.update_params
#        Input         : Learning rate and saved derivatives from backprop functions
#        Output        : Nothing returned but globally saved params are updated
#        Functionality : The function accesses the exiting parameters and updates them in CNN and XNN layers, using gradient descent method:
#                        H = H - lr * de/dH
#       >
#
#    2. Accuracy display
#
#       * For some reason it runs too long and the accuracy dropping to 94% in 2nd epoch.
#       * Did not see this behavior in the original submit. Will debug and fix for personal satisfaction.
#       ====================================================================================
#       Epoch    Execution time(s)   Training Loss   Training Accuracy(%)   Test Accuracy(%)
#       ====================================================================================
#         1             1713.6              0.000              94.21             96.27
#       ===================================================================================
#       Final test accuracy =  96.27 percent
#       ===================================================================================
#
#    3. Performance display
#
#       ================================================================================
#       Total Execution time =  29 minutes  
#       ================================================================================
#       ================================================================================
#       Layer                  Input size         Output size       Parameter size     
#       ================================================================================
#       Data                    60000x28x28        1x28x28           0          
#       Division by 255.0       1x28x28            1x28x28           0          
#       3x3/1 0pad CNN2dConv-1  1x28x28            16x28x28          1x16x28x28          
#       Addition-1              16x28x28           16x28x28          16x28x28    
#       ReLu-1                  16x28x28           16x28x28          0          
#       3x3/2 0pad maxPool-1    16x28x28           16x14x14          0       
#       3x3/1 0pad CNN2dConv-2  16x14x14           32x14x14          16x32x14x14          
#       Addition-2              32x14x14           32x14x14          32x14x14    
#       ReLu-2                  32x14x14           32x14x14          0          
#       3x3/2 0pad maxPool-2    32x14x14           32x7x7            0       
#       3x3/1 0pad CNN2dConv-3  32x7x7             64x7x7            32x64x7x7          
#       Addition-3              64x7x7             64x7x7            64x7x7    
#       ReLu-3                  64x7x7             64x7x7            0          
#       Vectorization           64x7x7             1x3136            0          
#       Matrix Mult-1           1x3136             1x100             3136x100     
#       Addition-4              1x100              1x100             1x100    
#       ReLu-4                  1x100              1x100             0          
#       Matrix Mult-2           1x100              1x10              100x10  
#       Addition-5              1x10               1x10              1x10    
#       Softmax                 1x10               1x10              0        
#       ================================================================================
#
#
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
NUM_EPOCHS             = 5
TRAIN_BATCH_SIZE       = 1
TEST_BATCH_SIZE        = 1
NUM_TRAIN_BATCHES      = DATA_NUM_TRAIN//TRAIN_BATCH_SIZE
NUM_TEST_BATCHES       = DATA_NUM_TEST//TEST_BATCH_SIZE
DENSE_LAYERS_INFO      = [100, DATA_CLASSES]
CNN_FILTER_SIZE        = 3
CNN_PADDING            = 1
CNN_STRIDE             = 1
MAXPOOL_FILTER_SIZE    = 3
MAXPOOL_PADDING        = 1
MAXPOOL_STRIDE         = 2
INPUT_CHANNELS         = [1,16,32] 
OUTPUT_CHANNELS        = [16,32,64]
IMG_SIZE               = [28,14,7] 

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

# Class for Fully connected layers 
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
            #if lyr > 1: # need to open this up for CNN
            #    dXPrev = mem["H" + str(lyr)].T.dot(dZ)
            dXPrev = mem["H" + str(lyr)].T.dot(dZ)
            derivs["dH" + str(lyr)] = dH
            derivs["dv" + str(lyr)] = dv

        return dXPrev, derivs

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

#global utility function to convert image receptive field to vectors and back
def get_img2vec(x_shape, filter_h, filter_w, pad, stride):
  x_n, x_c, x_h, x_w = x_shape
  out_h = ((x_h + 2 * pad - filter_h) // stride) + 1
  out_w = ((x_w + 2 * pad - filter_w) // stride) + 1
  i0 = np.repeat(np.arange(filter_h), filter_w)
  i0 = np.tile(i0, x_c)
  i1 = stride * np.repeat(np.arange(out_h), out_w)
  j0 = np.tile(np.arange(filter_w), filter_h * x_c)
  j1 = stride * np.tile(np.arange(out_w), out_h)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  k = np.repeat(np.arange(x_c), filter_h * filter_w).reshape(-1, 1)
  return (k, i, j)

#global utility function to convert image receptive field to vectors
def img2vec(x, filter_h, filter_w, pad, stride):
  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
  k, i, j = get_img2vec(x.shape, filter_h, filter_w, pad, stride)
  cols = x_pad[:, k, i, j]
  x_c = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(filter_h * filter_w * x_c, -1)
  return cols

#global utility function to convert vectors back into image receptive fields
def vec2img(cols, x_shape, filter_h, filter_w, pad,stride):
  x_n, x_c, x_h, x_w = x_shape
  x_h_pad, x_w_pad = x_h + 2 * pad, x_w + 2 * pad
  x_pad = np.zeros((x_n, x_c, x_h_pad, x_w_pad), dtype=cols.dtype)
  k, i, j = get_img2vec(x_shape, filter_h, filter_w, pad, stride)
  cols_reshaped = cols.reshape(x_c * filter_h * filter_w, -1, x_n)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_pad, (slice(None), k, i, j), cols_reshaped)
  if pad == 0:
    return x_pad
  return x_pad[:, :, pad:-pad, pad:-pad]


class CNN_LAYER():

    def __init__(self):
        self.cnn_params = {}
        self.cnn_derivs = {}
    # function to initialize parameters
    def initialize(self, X_shape, filter_n, filter_h, filter_w, stride, pad,lyr):
        X_c, X_h, X_w = X_shape
        
        #initialize parameters
        self.cnn_params["H" + str(lyr)] = np.random.randn(filter_n, X_c, filter_h, filter_w) / np.sqrt(filter_n / 2.0)
        self.cnn_params["v" + str(lyr)] = np.zeros((filter_n, 1))

        out_h = (X_h - filter_h + 2 * pad) / stride + 1
        out_w = (X_w - filter_w + 2 * pad) / stride + 1

        out_h, out_w = int(out_h), int(out_w)
        out_shape = (filter_n, out_h, out_w)
        
        self.cnn_params["out_h" + str(lyr)] = out_h
        self.cnn_params["out_w" + str(lyr)] = out_w
        self.cnn_params["out_shape" + str(lyr)] = out_shape

    # function for one CNN layer forward pass 
    def forward_pass(self, X, filter_n, filter_h, filter_w, stride, pad, lyr):

        X_n, X_c, X_h, X_w  = X.shape
        self.cnn_params["X_n" + str(lyr)] = X_n
        self.cnn_params["X_c" + str(lyr)] = X_c
        self.cnn_params["X_h" + str(lyr)] = X_h
        self.cnn_params["X_w" + str(lyr)] = X_w

        #convert 2-d image into vectors based on filter size.
        X_col = img2vec(X, filter_h, filter_w, stride, pad)
        W_row = self.cnn_params["H" + str(lyr)].reshape(filter_n, -1)
        self.cnn_params["X_colT" + str(lyr)] = X_col.T
        
        #CNN matrix multiplication layer
        out = W_row @ X_col 
        #Bias layer
        out = out + self.cnn_params["v" + str(lyr)]
        #ReLU
        out = np.maximum(out, 0)

        out = out.reshape(filter_n, self.cnn_params["out_h" + str(lyr)], self.cnn_params["out_w" + str(lyr)], X_n)
        out = out.transpose(3, 0, 1, 2)
        return out

    # function for one CNN layer backprop 
    def backprop(self,dout,filter_n,filter_h,filter_w,pad,stride,lyr):

        dout_flat = dout.transpose(1, 2, 3, 0).reshape(filter_n, -1)

        # derivative of error wrt weights = (dout.xT)
        dH = dout_flat @ self.cnn_params["X_colT" + str(lyr)]
        dH = dH.reshape(self.cnn_params["H" + str(lyr)].shape)

         # derivative of error wrt biases = sum(dout)
        dv = np.sum(dout, axis=(0, 2, 3)).reshape(filter_n, -1)
        H_flat = self.cnn_params["H" + str(lyr)].reshape(filter_n, -1)

        #prepare vectorized receptive field
        dX_col = H_flat.T @ dout_flat
        shape = (self.cnn_params["X_n" + str(lyr)], self.cnn_params["X_c" + str(lyr)], self.cnn_params["X_h" + str(lyr)], self.cnn_params["X_w" + str(lyr)])
        
        #convert vector back into image dims based on filter size
        dX = vec2img(dX_col, shape, filter_h, filter_w, pad, stride)
        self.cnn_derivs["dH" + str(lyr)] = dH
        self.cnn_derivs["dv" + str(lyr)] = dv
        return dX, self.cnn_derivs

    # function to update trainable parameters in one CNN layer
    def update_params(self,lr,lyr):
          self.cnn_params["H" + str(lyr)] = self.cnn_params["H" + str(lyr)] - lr * self.cnn_derivs["dH" + str(lyr)]
          self.cnn_params["v" + str(lyr)] = self.cnn_params["v" + str(lyr)] - lr * self.cnn_derivs["dv" + str(lyr)]


class MAXPOOL_LAYER():

    def __init__(self):
        self.maxpool_params = {}

    # function to initialize parameters
    def initialize(self, X_shape, size, pad, stride, lyr):

        X_c, X_h, X_w = X_shape
        out_h = (X_h - size + 2 * pad) // stride + 1
        out_w = (X_w - size + 2 * pad) // stride + 1
        out_h, out_w = int(out_h), int(out_w)
        out_shape = (X_c, out_h, out_w)
        self.maxpool_params["out_h" + str(lyr)] = out_h
        self.maxpool_params["out_w" + str(lyr)] = out_w
        self.maxpool_params["out_shape" + str(lyr)] = out_shape

    # function for one Maxpool layer forward pass 
    def forward_pass(self, X, size, pad, stride, lyr):
        X_n, X_c, X_h, X_w       = X.shape
        self.maxpool_params["X_n" + str(lyr)] = X_n
        self.maxpool_params["X_c" + str(lyr)] = X_c
        self.maxpool_params["X_h" + str(lyr)] = X_h
        self.maxpool_params["X_w" + str(lyr)] = X_w
        X_reshaped = X.reshape(X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[3])
        
        #convert 2-d image into vectors based on filter size.
        X_col = img2vec(X_reshaped, size, size, pad, stride)
        
        #get the max value after vectorization of receptive field
        max_indexes = np.argmax(X_col, axis=0)
        out = X_col[max_indexes, range(max_indexes.size)]

        out = out.reshape(self.maxpool_params["out_h" + str(lyr)], self.maxpool_params["out_w" + str(lyr)], X_n, X_c).transpose(2, 3, 0, 1)
        self.maxpool_params["X_col" + str(lyr)] = X_col
        self.maxpool_params["max_indexes" + str(lyr)] = max_indexes
       
        return out
    # function for one Maxpool layer backprop
    def backprop(self,dout,size,pad,stride,lyr):
        # prepare the vectorized receptive field
        dX_col = np.zeros_like(self.maxpool_params["X_col" + str(lyr)])
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        dX_col[self.maxpool_params["max_indexes" + str(lyr)], range(self.maxpool_params["max_indexes" + str(lyr)].size)] = dout_flat
        shape = (self.maxpool_params["X_n" + str(lyr)] * self.maxpool_params["X_c" + str(lyr)], 1, self.maxpool_params["X_h" + str(lyr)], self.maxpool_params["X_w" + str(lyr)])

        #convert vector back into image dims based on filter size
        dX = vec2img(dX_col, shape, size, size, pad, stride)
        dX = dX.reshape(self.maxpool_params["X_n" + str(lyr)] * self.maxpool_params["X_c" + str(lyr)], self.maxpool_params["X_h" + str(lyr)], self.maxpool_params["X_w" + str(lyr)])
        return dX


if __name__ == '__main__':

   print("====================================================================================")
   print("Epoch    Execution time(s)   Training Loss   Training Accuracy(%)   Test Accuracy(%)")
   print("====================================================================================")

   # Create matrix multiplication layers, ReLU layers and Classification layer
   # Pass dims of layers with trainable parameters and classification layer 
   # Dense Layer 1x100, Classification Layer = 1xDATA_CLASSES
   xnn      = XNN(DENSE_LAYERS_INFO) 
   # Create CNN and Maxpool layers
   cnn      = CNN_LAYER()
   maxpool  = MAXPOOL_LAYER()
   
  # Convert lables to one-hot encoding for math convenience
   train_y = np.zeros((train_labels.size, train_labels.max()+1))
   train_y[np.arange(train_labels.size),train_labels] = 1
   test_y = np.zeros((test_labels.size, test_labels.max()+1))
   test_y[np.arange(test_labels.size),test_labels] = 1

   # Division by 255.0 layer
   train_x     = train_data / 255.0
   test_x      = test_data / 255.0
   
   train_x_cnn  = train_x.reshape(DATA_NUM_TRAIN, DATA_ROWS , DATA_COLS, 1)
   test_x_cnn   = test_x.reshape(DATA_NUM_TEST, DATA_ROWS , DATA_COLS, 1)
  
   # Intitalize cnn layers
   for lyr in range(len(IMG_SIZE)):
     cnn.initialize((INPUT_CHANNELS[lyr],IMG_SIZE[lyr],IMG_SIZE[lyr]),OUTPUT_CHANNELS[lyr],CNN_FILTER_SIZE,CNN_FILTER_SIZE,CNN_PADDING,CNN_STRIDE,lyr+1)
   # Initialize maxpool layers
   for lyr in range(len(IMG_SIZE)-1):
     maxpool.initialize((OUTPUT_CHANNELS[lyr],IMG_SIZE[lyr],IMG_SIZE[lyr]),MAXPOOL_FILTER_SIZE,MAXPOOL_PADDING,MAXPOOL_STRIDE,lyr+1)
   # Initialize dense layers
   xnn.initialize((TRAIN_BATCH_SIZE,IMG_SIZE[len(IMG_SIZE)-1]*IMG_SIZE[len(IMG_SIZE)-1]*OUTPUT_CHANNELS[len(IMG_SIZE)-1]))


   # Cycle through epochs
   for epoch in range(NUM_EPOCHS):
  
     # To compute execution time per epoch
     epoch_start = time.time()
   
     # set learning rate proportional to the batch size
     if (TRAIN_BATCH_SIZE < 100):
       lr = 0.001
     elif (TRAIN_BATCH_SIZE < 1000):
       lr = 0.01
     else:
       lr = 0.1
     
     #initialize train and test accuracy to zero
     train_acc = 0
     test_acc  = 0
     
     #for each epoch cycle through the training data
     for batch in range(NUM_TRAIN_BATCHES):
 
       batch_start = time.time()

       #slice training data into batches
       batch_start_index = batch*TRAIN_BATCH_SIZE
       batch_end_index   = batch_start_index + TRAIN_BATCH_SIZE
       train_x_cnn       = train_x_cnn.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS , DATA_COLS)
       train_x_cnn_slice = train_x_cnn[batch_start_index:batch_end_index]
       train_y_slice     = train_y[batch_start_index:batch_end_index]
       
       # Forward_pass: cnn2d + bias + relu + maxpool
       cnn_start = time.time()
       out       = train_x_cnn_slice
       for lyr in range(len(IMG_SIZE)-1):
         out1     = cnn.forward_pass(out, OUTPUT_CHANNELS[lyr], CNN_FILTER_SIZE, CNN_FILTER_SIZE, CNN_PADDING, CNN_STRIDE, lyr+1)
         out      = maxpool.forward_pass(out1,MAXPOOL_FILTER_SIZE,MAXPOOL_PADDING,MAXPOOL_STRIDE,lyr+1)
       lyr = len(IMG_SIZE)-1
       last_out = cnn.forward_pass(out, OUTPUT_CHANNELS[lyr], CNN_FILTER_SIZE, CNN_FILTER_SIZE, CNN_PADDING, CNN_STRIDE, lyr+1)
       cnn_end = time.time()

       # Vectorization layer
       train_x_cnn_out_slice = last_out.reshape(TRAIN_BATCH_SIZE, OUTPUT_CHANNELS[lyr]*IMG_SIZE[lyr]*IMG_SIZE[lyr])
 
       # Dense Layers Forward Pass
       X, mem        = xnn.forward_pass(train_x_cnn_out_slice.T)
      
       # Compute loss 
       loss          = xnn.compute_loss(X,train_y_slice)
       
       # Dense Layers Backprop
       dout, derivs  = xnn.backprop(train_x_cnn_out_slice.T, train_y_slice.T, mem)
       
       # CNN layers backprop
       for lyr in range(len(IMG_SIZE)-1, 0, -1): 
          dout1                 = dout.reshape(TRAIN_BATCH_SIZE, OUTPUT_CHANNELS[lyr], IMG_SIZE[lyr] , IMG_SIZE[lyr])
          dout1, cnn_derivs1    = cnn.backprop(dout1,OUTPUT_CHANNELS[lyr],CNN_FILTER_SIZE,CNN_FILTER_SIZE,CNN_PADDING,CNN_STRIDE,lyr+1)
          dout                  = maxpool.backprop(dout1,MAXPOOL_FILTER_SIZE,MAXPOOL_PADDING,MAXPOOL_STRIDE,lyr)
       lyr = 0
       dout_last                     = dout.reshape(TRAIN_BATCH_SIZE, OUTPUT_CHANNELS[lyr], IMG_SIZE[lyr] , IMG_SIZE[lyr])
       dout_last, cnn_derivs_last    = cnn.backprop(dout_last,OUTPUT_CHANNELS[lyr],CNN_FILTER_SIZE,CNN_FILTER_SIZE,CNN_PADDING,CNN_STRIDE,lyr+1)
       cnn_end = time.time()

       # Dense Weight update
       update_params = xnn.update_params(lr,derivs)
       
       # CNN weight update
       for lyr in range(len(IMG_SIZE)-1):
          cnn.update_params(lr,lyr+1)

       # Compute training accuracy
       train_acc = train_acc + xnn.compute_acc(X,train_y_slice)
       batch_end = time.time()
  
     # For each epoch cycle through the testing data
     for batch in range(NUM_TEST_BATCHES):
       
       batch_start = time.time()

       #slice training data into batches
       batch_start_index = batch*TEST_BATCH_SIZE
       batch_end_index   = batch_start_index + TEST_BATCH_SIZE
       test_x_cnn        = test_x_cnn.reshape(DATA_NUM_TEST, 1, DATA_ROWS , DATA_COLS)
       test_x_cnn_slice  = test_x_cnn[batch_start_index:batch_end_index]
       test_y_slice      = test_y[batch_start_index:batch_end_index]

       # Forward_pass: cnn2d + bias + relu + maxpool
       out       = test_x_cnn_slice
       for lyr in range(len(IMG_SIZE)-1):
         out1     = cnn.forward_pass(out, OUTPUT_CHANNELS[lyr], CNN_FILTER_SIZE, CNN_FILTER_SIZE, CNN_PADDING, CNN_STRIDE, lyr+1)
         out      = maxpool.forward_pass(out1,MAXPOOL_FILTER_SIZE,MAXPOOL_PADDING,MAXPOOL_STRIDE,lyr+1)
       lyr = len(IMG_SIZE)-1
       last_out = cnn.forward_pass(out, OUTPUT_CHANNELS[lyr], CNN_FILTER_SIZE, CNN_FILTER_SIZE, CNN_PADDING, CNN_STRIDE, lyr+1)

       # Vectorization layer
       test_x_cnn_out_slice = last_out.reshape(TEST_BATCH_SIZE, OUTPUT_CHANNELS[lyr]*IMG_SIZE[lyr]*IMG_SIZE[lyr])
 
       # Dense Forward Pass
       X, mem_test        = xnn.forward_pass(test_x_cnn_out_slice.T)

       batch_end = time.time()

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