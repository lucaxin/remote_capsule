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
import torch.nn.functional as F
from torch import nn
import numpy as np

# some Hyperparameters
BATCH_SIZE = 100              # BATCH_SIZE could train lots of imgs at the same time
NUM_CLASSES = 10                # the datasets' label classes
NUM_EPOCHS = 500                # the train iterations

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
        return (F.log_softmax(x, dim=1))

#   值得注意的是：网络的的输出10个维度代表了不同的概率，
if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm
    import torchnet as tnt

    model = BaseLineNet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    # 使用Adam优化器
    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()                                  # 平均损失
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)                   # 预测误差 accuracy=True 返回1-error
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    # visdom可视化数据
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    baseline_loss = nn.CrossEntropyLoss()

    # 加载数据集 dataloader
    def get_iterator(mode):
        # mode is True or False
        dataset = MNIST(root='./data', download=True, train=mode)
        # getattr() 函数用于返回一个对象属性值
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        # BATCH_SIZE 100
        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)

    def reset_meters():
        # 重置模型的一些参数，在开始训练前
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()

    def processor(sample):
        data, labels, training = sample
        # 图像增强操作augmentation
        data = augmentation(data.unsqueeze(1).float() / 255.0)
        labels = torch.LongTensor(labels)

        # labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            outputs = model(data)
        else:
            outputs = model(data)

        loss = baseline_loss(outputs, labels)

        return loss, outputs

    def on_sample(state):
        # 从dataloader取样
        state['sample'].append(state['train'])

    def on_forward(state):
        #
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())

    def on_start_epoch(state):
        # 每次epoch开始时 重置参数 模型评估
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        # epoch结束后的操作
        # 主要是输入log信息，计算误差
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()

        engine.test(processor, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), 'baseline_model/epoch_%d.pt' % state['epoch'])

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)




# T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# mnist_data = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=T)
# mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=100)
#
# cec_loss = nn.CrossEntropyLoss()
# params = model.parameters()
# #  8215568
# # 13277970
# # keep free aboubt the parameters the baseline CNN has.
# print("# parameters:", sum(param.numel() for param in model.parameters()))
# optimizer = optim.Adam(params=params)
#
# n_epoch = 300
# n_iterations = 0
#
# vis = Visdom(use_incoming_socket=False)
# vis_window = vis.line(np.array([0]), np.array([0]))
#
# for e in range(n_epoch):
#     for i,(images,labels) in enumerate(mnist_dataloader):
#         images = Variable(images)
#         labels = Variable(labels)
#         output = model(images)
#
#         model.zero_grad()
#         loss = cec_loss(output,labels)
#         loss.backward()
#
#         optimizer.step()
#
#         n_iterations+=1
#
#         vis.line(np.array([loss.item()]),np.array([n_iterations]),win=vis_window, update='append')
#         break
