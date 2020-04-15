import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = (self.key_pts_frame.iloc[idx, 1:]).to_numpy()
        print(key_pts.shape)
        key_pts = key_pts.reshape((int(len(key_pts)/2), 2))
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}


import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #input image = 200*200
        #output size = (200-5)/1 + 1 = 196
        #so output dimensions after conv1 = (32,196, 196)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        #xavier initialization
        I.xavier_uniform(self.conv1.weight)

        #after max pooling = 196/2 = 98
        #dims = (32, 98, 98)
        self.max_pool = nn.MaxPool2d(2,2)
        
        #after conv2 = (98-5)/1 + 1 = 94
        #dims = (15, 94, 94)
        self.conv2 = nn.Conv2d(32, 15, 5)
        
        #xavier initialization
        I.xavier_uniform(self.conv2.weight)

        #after max pool2 = 94/2 = 47
        #dims = (15, 47, 47)
        
        #after fc1 = 10,000
        #dims = (10000,1)
        self.fc1 = nn.Linear(15*47*47, 10000)
        
        #xavier init
        I.xavier_uniform(self.fc1.weight)

        #defining dropout layer
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(10000, 5000)
        

        #xavier init
        I.xavier_uniform(self.fc2.weight)

        #Defining final linear activation layer with 68(no of keypoints)*2 output neurons
        self.fc3 = nn.Linear(5000, 68*2)
        
        #xavier
        I.xavier_uniform(self.fc3.weight)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0),-1)
        
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

# import the usual resources
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0")

net = Net().to(device)
print(net)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
#from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor


## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(250), RandomCrop(200), Normalize(), ToTensor()])

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                             root_dir='./data/training/',
                                             transform=data_transform)


print('Number of images: ', len(transformed_dataset))


# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())


# load training data in batches
batch_size = 5

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=1)

# load in the test data, using the dataset class
# AND apply the data_transform you defined above

# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                             root_dir='./data/test/',
                                             transform=data_transform)


# load test data in batches
batch_size = 10

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=1)

import io
from contextlib import redirect_stdout

# test the model on a batch of test images
def net_sample_output():
      # iterate through the test dataset
      trap = io.StringIO()
      with redirect_stdout(trap):
        for i, sample in enumerate(test_loader):
            
            # get sample data: images and ground truth keypoints
            images = sample['image'].to(device)
            key_pts = sample['keypoints'].to(device)

            # convert images to FloatTensors
            images = images.type(torch.cuda.FloatTensor)

            # forward pass to get net output
            output_pts = net(images)
            
            #print(output_pts.shape)
            # reshape to batch_size x 68 x 2 pts
            output_pts = output_pts.view(output_pts.size()[0], 68, -1)
            
            # break after first image is tested
            
            if i == 0:
                
                return images, output_pts, key_pts      
            
# call the above function
# returns: test images, test predicted keypoints, test ground truth keypoints

test_images, test_outputs, gt_pts = net_sample_output()
# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


#For suppressing unnecessay outputs
import sys, os

class HandlePrint():

    def __init__(self):
        self.initial_stout = sys.stdout

    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def resetPrint(self):
        sys.stdout = self.initial_stout


## TODO: Define the loss and optimization
import torch.optim as optim

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr = 0.001)


def train_net_gpu(n_epochs):
    batch_step_loss = np.array([])
    epoch_step_loss = np.array([])
    # prepare the net for training
    net.train()
    total_batches = len(train_loader)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        epoch_loss = 0.0
        # train on batches of data, assumes you already have train_loader
        handle = HandlePrint()
        handle.blockPrint()
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels

            handle.blockPrint()

            images = data['image'].to(device)
            key_pts = data['keypoints'].to(device)
            
            handle.resetPrint()

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()
            
            #Delete all the variables from the gpu
            del images
            del key_pts
            torch.cuda.empty_cache()

            # print loss statistics
            
            epoch_loss += loss.item()
            running_loss += loss.item()
            
            if batch_i % 10 == 9:    # print every 10 batches
                handle.resetPrint()
                print('Epoch: {}, Batch: {}/{}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, total_batches, running_loss/10))
                batch_step_loss = np.append(batch_step_loss, running_loss)
                running_loss = 0.0    
                
        epoch_step_loss = np.append(epoch_step_loss, epoch_loss)        
        epoch_loss = 0.0        
    handle.resetPrint()
    print('Finished Training')
    return batch_step_loss, epoch_step_loss

# train your network
n_epochs = 10# start small, and increase when you've decided on your model structure and hyperparams
L_batch, L_epoch = train_net_gpu(n_epochs) 


model_dir = 'saved_models/'
model_name = 'keypoints_model_2.pt'

torch.save(net.state_dict(), model_dir + model_name)

np.save(model_dir + 'L_batch.npy', L_batch)

np.save(model_dir + 'L_epoch.npy', L_epoch)


print("All files saved Successfully!")


