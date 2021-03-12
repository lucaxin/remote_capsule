'''
Graduation Project baselineCNN by luca lau

Thanks for project https://github.com/pytorch/tnt/blob/master/example/mnist_with_visdom.py

NOTICE:
ubuntu 18.04 | cuda 10.2 | GTX 1050Ti
torch 1.1.0 and torchvision 0.3.0

Run MNIST example and log to visdom
    Notes:
        - Visdom must be installed (pip works)
        - the Visdom server must be running at start!
    Example:
        $ python -m visdom.server -port 8097 &
        $ python mnist_with_visdom.py
'''

import sys
# this line is to increase python stack length limitation
sys.setrecursionlimit(15000)


import torch
import torchvision
from visdom import Visdom
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def augmentation(x, max_shift=2):
    '''
    图像数据增强功能：图像的平移 max_shift = 2的尺度
    :param x:
    :param max_shift:
    :return:
    '''
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()

"Conv1 -> RELU -> Conv2 -> RELU -> Conv3 -> RELu -> FC1 -> RELU -> FC2 -> SOFTMAX"
class BaseLineNet(nn.Module):
    def __init__(self):
        super(BaseLineNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, stride=1)        # in [1,28,28]   -> out [256,24,24]
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1)      # in [256,24,24] -> out [256,20,20]
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1)      # in [256,20,20] -> out [128,16,16]
        self.fc1 = nn.Linear(128*16*16, 328)
        self.fc2 = nn.Linear(328, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = x.view(-1, 128*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return (F.log_softmax(x))

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_data = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=T)
mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=100)

model = BaseLineNet()
cec_loss = nn.CrossEntropyLoss()
params = model.parameters()
#  8215568
# 13277970
# keep free aboubt the parameters the baseline CNN has.
print("# parameters:", sum(param.numel() for param in model.parameters()))
optimizer = optim.Adam(params=params)

n_epoch = 300
n_iterations = 0

vis = Visdom(use_incoming_socket=False)
vis_window = vis.line(np.array([0]), np.array([0]))

for e in range(n_epoch):
    for i,(images,labels) in enumerate(mnist_dataloader):
        images = Variable(images)
        labels = Variable(labels)
        output = model(images)

        model.zero_grad()
        loss = cec_loss(output,labels)
        loss.backward()

        optimizer.step()

        n_iterations+=1

        vis.line(np.array([loss.item()]),np.array([n_iterations]),win=vis_window, update='append')
        break
