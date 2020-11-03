################################################################################
#
# LOGISTICS
#
#    Name: Dinakar Kondru
#    UTD ID: 2021495823
#
# DESCRIPTION
#
#    Image classification in PyTorch for ImageNet reduced to 100 classes and
#    down sampled such that the short side is 64 pixels and the long side is
#    >= 64 pixels
#
#    This script achieved a best accuracy of 71.55% on epoch 59 with a learning
#    rate at that point of 0.000100 and time required for each epoch of ~62 s(colab pro)
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    0. For a mapping of category names to directory names see:
#       https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
#
#    1. The original 2012 ImageNet images are down sampled such that their short
#       side is 64 pixels (the other side is >= 64 pixels) and only 100 of the
#       original 1000 classes are kept.
#
#    2. Build and train a RegNetX image classifier modified as follows:
#
#       - Set stride = 1 (instead of stride = 2) in the stem
#       - Replace the first stride = 2 down sampling building block in the
#         original network by a stride = 1 normal building block
#       - The fully connected layer in the decoder outputs 100 classes instead
#         of 1000 classes
#
#       The original RegNetX takes in 3x224x224 input images and generates Nx7x7
#       feature maps before the decoder, this modified RegNetX will take in
#       3x56x56 input images and generate Nx7x7 feature maps before the decoder.
#       For reference, an implementation of this network took ~ 112 s per epoch
#       for training, validation and checkpoint saving on Sep 27, 2020 using a
#       free GPU runtime in Google Colab.
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

# torch
import torch
import torch.nn       as     nn
import torch.optim    as     optim
from   torch.autograd import Function

# torch utils
import torchvision
import torchvision.transforms as transforms

# additional libraries
import os
import urllib.request
import zipfile
import time
import math
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_DIR_1        = 'data'
DATA_DIR_2        = 'data/imagenet64'
DATA_DIR_TRAIN    = 'data/imagenet64/train'
DATA_DIR_TEST     = 'data/imagenet64/val'
DATA_FILE_TRAIN_1 = 'Train1.zip'
DATA_FILE_TRAIN_2 = 'Train2.zip'
DATA_FILE_TRAIN_3 = 'Train3.zip'
DATA_FILE_TRAIN_4 = 'Train4.zip'
DATA_FILE_TRAIN_5 = 'Train5.zip'
DATA_FILE_TEST_1  = 'Val1.zip'
DATA_URL_TRAIN_1  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train1.zip'
DATA_URL_TRAIN_2  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train2.zip'
DATA_URL_TRAIN_3  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train3.zip'
DATA_URL_TRAIN_4  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train4.zip'
DATA_URL_TRAIN_5  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train5.zip'
DATA_URL_TEST_1   = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Val1.zip'
DATA_BATCH_SIZE   = 512
DATA_NUM_WORKERS  = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 100
DATA_RESIZE       = 64
DATA_CROP         = 56
DATA_MEAN         = (0.485, 0.456, 0.406)
DATA_STD_DEV      = (0.229, 0.224, 0.225)

# model: 200MF parameters
MODEL_TAIL_OUT_CHANNELS  = 24
MODEL_BLOCKS             = [1,1,4,7]
MODEL_CHANNELS           = [24,56,152,368]
MODEL_GROUP_WIDTH        = 8
MODEL_FILTER_SIZE        = 3

# training
TRAINING_LR_MAX          = 0.01
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 5
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 55
TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

# file
FILE_NAME = 'cnn_project2.pt'
FILE_SAVE = 1
FILE_LOAD = 0

################################################################################
#
# DATA
#
################################################################################
code_start = time.time()

# create a local directory structure for data storage
if (os.path.exists(DATA_DIR_1) == False):
    os.mkdir(DATA_DIR_1)
if (os.path.exists(DATA_DIR_2) == False):
    os.mkdir(DATA_DIR_2)
if (os.path.exists(DATA_DIR_TRAIN) == False):
    os.mkdir(DATA_DIR_TRAIN)
if (os.path.exists(DATA_DIR_TEST) == False):
    os.mkdir(DATA_DIR_TEST)

# download data
if (os.path.exists(DATA_FILE_TRAIN_1) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_1, DATA_FILE_TRAIN_1)
if (os.path.exists(DATA_FILE_TRAIN_2) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_2, DATA_FILE_TRAIN_2)
if (os.path.exists(DATA_FILE_TRAIN_3) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_3, DATA_FILE_TRAIN_3)
if (os.path.exists(DATA_FILE_TRAIN_4) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_4, DATA_FILE_TRAIN_4)
if (os.path.exists(DATA_FILE_TRAIN_5) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_5, DATA_FILE_TRAIN_5)
if (os.path.exists(DATA_FILE_TEST_1) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_1, DATA_FILE_TEST_1)

# extract data
with zipfile.ZipFile(DATA_FILE_TRAIN_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_2, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_3, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_4, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_5, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TEST_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TEST)

# transforms
transform_train = transforms.Compose([transforms.RandomResizedCrop(DATA_CROP), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
transform_test  = transforms.Compose([transforms.Resize(DATA_RESIZE), transforms.CenterCrop(DATA_CROP), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])

# data sets
dataset_train = torchvision.datasets.ImageFolder(DATA_DIR_TRAIN, transform=transform_train)
dataset_test  = torchvision.datasets.ImageFolder(DATA_DIR_TEST,  transform=transform_test)

# data loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True,  num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)

################################################################################
#
# NETWORK BUILDING BLOCK
#
################################################################################

# X block
class XBlock(nn.Module):

    # initialization
    def __init__(self, Ni, No, Fr, Fc, Sr, Sc, G):

        # parent initialization
        super(XBlock, self).__init__()

        # identity part
        if ((Ni != No) or (Sr > 1) or (Sc >1)):
            self.conv0_present = True
            self.conv0         = nn.Conv2d(Ni, No, (1, 1), stride=(Sr, Sc), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        else:
            self.conv0_present = False

        # residual part
        self.bn1   = nn.BatchNorm2d(Ni, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(Ni, No, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.bn2   = nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(No, No, (Fr, Fc), stride=(Sr, Sc), padding=(1, 1), dilation=(1, 1), groups=No//G, bias=False, padding_mode='zeros')
        self.bn3   = nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(No, No, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')

    # forward path
    def forward(self, x):

        # residual path
        z = self.bn1(x)
        z = self.relu1(z)
        z = self.conv1(z)
        z = self.bn2(z)
        z = self.relu2(z)
        z = self.conv2(z)
        z = self.bn3(z)
        z = self.relu3(z)
        z = self.conv3(z)

        # identity path
        if (self.conv0_present == True):
            x = self.conv0(x)

        # addition
        x = x + z
        
        y = x

        # return
        return y

################################################################################
#
# NETWORK
#
################################################################################

# define
class Model(nn.Module):

    # initialization
    def __init__(self, 
                 data_num_channels,
                 model_blocks, 
                 model_tail_out_channels, 
                 model_channels,
                 data_num_classes):

        # parent initialization
        super(Model, self).__init__()
        
        # Stage-independent constant Model parameters 
        self.num_stages        = len(model_channels)
        self.tail_filter_size  = MODEL_FILTER_SIZE
        self.tail_stride       = 1 
        self.model_Sr1         = 1
        self.model_Sc1         = self.model_Sr1
        self.model_Fr          = MODEL_FILTER_SIZE
        self.model_Fc          = self.model_Fr
        self.model_G           = MODEL_GROUP_WIDTH

        # encoder tail/stem
        self.enc_tail = nn.ModuleList()
        self.enc_tail.append(nn.Conv2d(data_num_channels, model_tail_out_channels, (self.tail_filter_size, self.tail_filter_size), 
                             stride=(self.tail_stride, self.tail_stride), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))

        # encoder body
        self.enc_body = nn.ModuleList()
        # loop through stages
        for stage in range(self.num_stages):
          self.model_Ni          = model_tail_out_channels if (stage==0) else model_channels[stage-1]
          self.model_No          = model_channels[stage]
          self.model_Sr2         = 1 if (stage==0) else 2
          self.model_Sc2         = self.model_Sr2
          # downsampling x_block
          self.enc_body.append(XBlock(self.model_Ni, self.model_No, self.model_Fr, self.model_Fc, self.model_Sr2, self.model_Sc2, self.model_G))
          # loop through stride=1 repeat x_blocks
          for n in range(model_blocks[stage] - 1):
            self.enc_body.append(XBlock(self.model_No, self.model_No, self.model_Fr, self.model_Fc, self.model_Sr1, self.model_Sc1, self.model_G))

        # encoder body last stage batchnorm and relu
        self.enc_body.append(nn.BatchNorm2d(model_channels[self.num_stages-1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_body.append(nn.ReLU())

        # decoder
        self.dec = nn.ModuleList()
        self.dec.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.dec.append(nn.Flatten())
        self.dec.append(nn.Linear(model_channels[self.num_stages-1], data_num_classes, bias=True))

    # forward path
    def forward(self, x):

        # encoder tail
        for layer in self.enc_tail:
            x = layer(x)
        # encoder body - all stages
        for layer in self.enc_body:
            x = layer(x)
        # decoder
        for layer in self.dec:
            x = layer(x)
        y = x
        # return
        return y

# create
model = Model(DATA_NUM_CHANNELS, 
              MODEL_BLOCKS, 
              MODEL_TAIL_OUT_CHANNELS,
              MODEL_CHANNELS,
              DATA_NUM_CLASSES )

# enable data parallelization for multi GPU systems
if (torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)

# Print number of GPUs and type of GPU
print('Using {0:d} GPU(s)'.format(torch.cuda.device_count()), flush=True)
print('GPU device:')
print(torch.cuda.get_device_name(0))

################################################################################
#
# GLOBAL UTILITY FUNCTIONS
#
################################################################################

# learning rate scheduler function
def lr_schedule(epoch):

    # linear warmup followed by cosine decay
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL

    return lr

# function to plot train loss vs epochs
def plot_loss():
    plt.figure(figsize=(20, 10), dpi=70)
    plt.plot(np.arange(len(train_loss)), train_loss, "-r", label='Train Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")
    plt.legend(loc="upper right")
    plt.show()

# function to plot test accuracy vs epochs
def plot_acc():
    plt.figure(figsize=(20, 10), dpi=70)
    plt.plot(np.arange(len(test_accuracy)), test_accuracy, "-g", label='Test Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.legend(loc="lower right")
    plt.show()

################################################################################
#
# ERROR AND OPTIMIZER
#
################################################################################

# error (softmax cross entropy)
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# specify the device as the GPU if present else default to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transfer the network to the device
model.to(device)

# model loading
if FILE_LOAD == 1:
    checkpoint = torch.load(FILE_NAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

print("=========================================================================================")
print("Epoch       LR            Training Loss           Test Accuracy(%)        Execution time(s) ")
print("=========================================================================================")

################################################################################
#
# TRAINING
#
################################################################################

# initialize global training statistics
start_epoch = 0
train_loss     = []
test_accuracy  = []

# cycle through epochs
for epoch in range(start_epoch, TRAINING_NUM_EPOCHS):
    epoch_start = time.time()

    # initialize train set statistics per epoch
    model.train()
    training_loss = 0.0
    num_batches   = 0

    # set the learning rate for the epoch
    for g in optimizer.param_groups:
        g['lr'] = lr_schedule(epoch)

    # cycle through the train set
    for data in dataloader_train:

        # extract a batch of data and move it to the appropriate device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass, loss, backward pass and weight update
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update training statistics
        training_loss = training_loss + loss.item()
        num_batches   = num_batches + 1

    # initialize test set statistics
    model.eval()
    test_correct = 0
    test_total   = 0

    # skip weight update/gradient for test set
    with torch.no_grad():

        # cycle through the test set
        for data in dataloader_test:

            # extract batch of data and move it to the device
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass and prediction
            outputs      = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # update test set statistics
            test_total   = test_total + labels.size(0)
            test_correct = test_correct + (predicted == labels).sum().item()

    # epoch statistics
    epoch_end = time.time()
    train_loss.append((training_loss/num_batches)/DATA_BATCH_SIZE)
    test_accuracy.append(100.0*test_correct/test_total)

    # save model for checkpoint
    if FILE_SAVE == 1:
         torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict()
         }, FILE_NAME)
 
    # print statistics per epoch
    epoch_time = epoch_end-epoch_start
    print('{0:2d}        {1:8.6f}         {2:8.6f}            {3:5.2f}              {4:4.1f}'.format(
    epoch, lr_schedule(epoch), (training_loss/num_batches)/DATA_BATCH_SIZE, (100.0*test_correct/test_total),epoch_time))


################################################################################
#
# TEST
#
################################################################################

# initialize test set statistics
model.eval()
test_correct = 0
test_total   = 0

# skip weight update/gradient for test set
with torch.no_grad():

    # cycle through the test set
    for data in dataloader_test:

        # extract a batch of data and move it to the appropriate device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass and prediction
        outputs      = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # update test set statistics
        test_total   = test_total + labels.size(0)
        test_correct = test_correct + (predicted == labels).sum().item()

# print test accuracy
print('Test Accuracy = {0:5.2f}'.format((100.0*test_correct/test_total)))

################################################################################
#
# DISPLAY
#
################################################################################

# Plot training loss vs. Epoch and test accuracy vs. Epoch
plot_loss()
plot_acc()

code_end = time.time()

# Total time
print("================================================================================")
print(" Total Execution time =  %.1f minutes  " % ((code_end-code_start)/60))
print("================================================================================")
