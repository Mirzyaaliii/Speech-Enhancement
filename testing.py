
# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse

from torch.autograd import Variable
from os import listdir, makedirs
from os.path import join, exists

from scipy.io import loadmat
from scipy.io import savemat


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parse script arguments
ap = argparse.ArgumentParser()

ap.add_argument("-sm", "--save_model", required=True, help="Give path of saved model")
ap.add_argument("-tst", "--test_path", required=True, help="path of testing data")
ap.add_argument("-pfp", "--predicted_file_path", required=True, help="path to save predicted features")

args = vars(ap.parse_args())


# -------------------- Network ------------------------------------------------------------

# Generator

class generator(nn.Module):
    
    def __init__(self, G_in, G_out):
        
        super(generator, self).__init__()
        
        self.fc1 = nn.Linear(G_in, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features//2)
        self.fc5 = nn.Linear(self.fc4.out_features, G_out)
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        out = self.fc5(x)
        prob = F.sigmoid(out)
        return out, prob


# Discriminator

class discriminator(nn.Module):
    
    def __init__(self, D_in, D_out):
        super(discriminator, self).__init__()
        
        self.fc1 = nn.Linear(D_in, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features//2)
        self.fc5 = nn.Linear(self.fc4.out_features, D_out)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        out = F.sigmoid(self.fc5(x))
        return out


# -------------------- Testing ------------------------------------------------------------

# Path of saved models
save_folder = args["save_model"]


# Path to save enhanced files
if not os.path.exists(args["predicted_file_path"]):
    os.mkdirs(args["predicted_file_path"])
save_predicted_files = args['predicted_file_path']


# Path of testing data
testing_path = args['test_path']
testing_files = listdir(testing_path)


# Load trained model
Gnet = torch.load(join(save_folder,"Gnet_Ep_200.pth")).to(device)


# Testing
for file1 in testing_files:
    print(file1)
    d = loadmat(join(testing_path, file1))

    feat = torch.from_numpy(d['Feat'])
    nc = torch.from_numpy(d['noisy_cent'])
    feat = Variable(feat.squeeze(0)).type(torch.FloatTensor).to(device)
    nc = Variable(nc.squeeze(0).type(torch.FloatTensor)).to(device)

    G_out, G_prob = Gnet(feat)
    predicted_cc = torch.log(torch.mul(G_prob, nc))

    # save predicted file
    savemat(join(save_predicted_files, file1), mdict={'PRED_IRM': G_prob.cpu().data.numpy(), 'Clean_Pred': predicted_cc.cpu().data.numpy()}) 

