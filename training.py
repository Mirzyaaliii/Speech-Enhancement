
# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.autograd as autograd
import os
import visdom
import argparse

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from os import listdir, makedirs
from os.path import join, exists

from scipy.io import loadmat
from scipy.io import savemat


viz = visdom.Visdom()	


# Check GPU availability

cuda = True if torch.cuda.is_available() else False
print("\n\n**************** Cuda Available :",cuda,"******************\n\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parse script arguments
ap = argparse.ArgumentParser()

ap.add_argument("-sm", "--save_model", required=True, help="path to save model")
ap.add_argument("-trn", "--train_path", required=True, help="path of training data")

args = vars(ap.parse_args())


# -------------------- Data ------------------------------------------------------------

# Class to generate dataset
class speech_data(Dataset):

	def __init__(self, folder_path):
		self.path = folder_path
		self.files = listdir(folder_path)
		self.length =  len(self.files)

	def __getitem__(self, index):
		d = loadmat(join(self.path, self.files[int(index)]))
		return np.array(d['noisy_cent']), np.log(d['clean_cent']), np.array(d['Feat'])

	def __len__(self):
		return self.length


# Train dataloader
training_path = speech_data(folder_path=args['train_path'])
training_data = DataLoader(dataset=training_path, batch_size=1, shuffle=True, num_workers=3)

# Path to save model
if not os.path.exists(args["save_model"]):
    os.mkdirs(args["save_model"])
save_folder = args["save_model"]


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


# -------------------- Inintializing Parameters ------------------------------------------------------------

# Netowrk parameters
G_in = 51
G_out = 51
D_in = 51
D_out = 1

# Initialize network
Gnet = generator(G_in, G_out).to(device)
Dnet = discriminator(D_in, D_out).to(device)

# Loss functions
loss_bce = nn.BCELoss()
loss_mse = nn.MSELoss()

# Initialize optimizers
Goptim = optim.Adam(Gnet.parameters(), lr = 0.0001)
Doptim = optim.Adam(Dnet.parameters(), lr = 0.0001)


# -------------------- Training ------------------------------------------------------------

def training(training_data, epoch_no):

	Gnet.train()
	Dnet.train()

	lossG = 0
	lossD = 0

	for en, (nc, cc, feat) in enumerate(training_data):

		nc = Variable(nc.squeeze(0).type(torch.FloatTensor)).to(device)
		cc = Variable(cc.squeeze(0).type(torch.FloatTensor)).to(device)
		feat = Variable(feat.squeeze(0).type(torch.FloatTensor)).to(device)

		valid = Variable(torch.ones(cc.size(0), 1), requires_grad=False).to(device)
		fake = Variable(torch.zeros(nc.size(0), 1), requires_grad=False).to(device)

		# ----- Generator Training --------------------------

		Goptim.zero_grad()

		G_out, G_prob = Gnet(feat)
		predicted_cc = torch.log(torch.mul(G_prob, nc))
		G_loss = loss_bce(Dnet(predicted_cc), valid) + loss_mse(predicted_cc, cc)
		lossG += G_loss.item()

		G_loss.backward()
		Goptim.step()

		# ----- Discriminator Training ----------------------

		Doptim.zero_grad()

		real_loss = loss_bce(Dnet(cc), valid)
		fake_loss = loss_bce(Dnet(predicted_cc.detach()), fake)
		D_loss = (real_loss + fake_loss)/2
		lossD += D_loss.item()

		D_loss.backward()
		Doptim.step()

		# -------------------------------------------

		print("[{}/{}] -> {} Generator : {} | Discriminator : {}".format(epoch_no, epoch, en+1, G_loss.item(), D_loss.item()))

	return lossG/(en+1), lossD/(en+1)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------

epoch = 200
gl_arr = []
dl_arr = []

for ep in range(1, epoch+1):

    lossG, lossD = training(training_data, ep)

    if (ep%1==0):
        torch.save(Gnet, join(save_folder+"Gnet_Ep_{}.pth".format(ep)))
        # torch.save(Dnet, join(save_folder+"Dnet_Ep_{}.pth".format(ep)))

    gl_arr.append(lossG)
    dl_arr.append(lossD)

    print("-------------------------------------------------------------------")
    print("Loss after epoch-{} : ".format(ep))
    print("[{}/{}] - Loss_G : {} | Loss_D : {}".format(ep, epoch, lossG, lossD))

    # Plot graphs on Visdom
    if (ep == 1):
        g_plot = viz.line(Y=np.array([lossG]), X=np.array([ep]), opts=dict(title='Generator'))
        d_plot = viz.line(Y=np.array([lossD]), X=np.array([ep]), opts=dict(title='Discriminator'))
    else:
        viz.line(Y=np.array([lossG]), X=np.array([ep]), win=g_plot, update='append')
        viz.line(Y=np.array([lossD]), X=np.array([ep]), win=d_plot, update='append')

# Save graphs
plt.figure(1)
plt.plot(gl_arr)
plt.savefig(save_folder+'/generator_loss.png')
plt.figure(2)
plt.plot(dl_arr)
plt.savefig(save_folder+'/discriminator_loss.png')
