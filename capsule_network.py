"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.

the Graduation Project by Luca lau 2021.3.1
Thanks for Github Project https://github.com/gram-ai/capsule-networks

NOTICE:
ubuntu 18.04 | cuda 10.2 | GTX 1050Ti
torch 1.1.0 and torchvision 0.3.0
this project is depend on visdom so when you run it
frist you must run commmand python -m visdom.server in your Terminal
and your 8097 portis not busy
"""
import sys
# this line is to increase python stack length limitation
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

# some Hyperparameters
BATCH_SIZE = 100                # BATCH_SIZE could train lots of imgs at the same time
NUM_CLASSES = 10                # the datasets' label classes
NUM_EPOCHS = 500                # the train iterations
NUM_ROUTING_ITERATIONS = 3      # the times of dynamic routing algorithm

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

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

class CapsuleLayer(nn.Module):
    # customer function capsule layer
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        '''
        capsule layer
        conv1 ==> primary_capsules ==> digit_capsules
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        '''
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        # digit capsule is different from the primary capsule because the parameters they have are different
        # digit capsule's parameters are not learning but a variable parameter which we hope to change in order to enhance the net
        # so we use nn.parameter to change it into variable and put it into the module
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        # capsule's activate function
        # dim = -1 ?
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                # IndexError: Dimension out of range (expected to be in range of [-2, 1] but got 2)
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs

class CapsuleNet(nn.Module):
    # the architecture of capsuleNet
    # the architecture is shallow with only two convolutional layers and one fully connected layer
    # Conv1 is 259, 9*9 kernels with stride of 1 and ReLU activation
    # the Conv1 is used to converts pixel intensities to the activities of local feature detectors which used as inputs of primary capsule
    # the primary capsule is convolutional capsule layer with 32 channels of convolutional 8D capsules
    # 8D capsules refer to capsule contains 8 convolutional units with 9*9 kernel and a stride of 2
    # 整个模型的输出为 classes：10 的向量，模长即为整个该数字的概率。 reconstructions： 将输出通过3层FC后得到的28*28的图片重构
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions

    #-----------------------------------------------  写累了，下了个搜狗拼音，下面打中文注释了，顺便吐槽下这里的输入法真难用  ----------------------------------------------#

class CapsuleLoss(nn.Module):
    # 胶囊网络的损失函数： separate Margin loss
    # we would like the top-level capsule for digit class k to have a long instantiation vector if and only if that digit is present in the image.
    # 计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0。
    # 输出：自定义损失值
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        # 注意这个几个参数的意思？
        # 原文中写的是向量的模
        left = F.relu(0.9 - classes, inplace=True) ** 2         # inplace 覆盖运算
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)

if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm
    import torchnet as tnt

    model = CapsuleNet()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()
    # 8215568
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    # 使用Adam优化器
    optimizer = Adam(model.parameters())

    # 使用Ignite训练网络 对于训练过程中的for循环，精简代码，提供度量，提前终止，保存模型，提供基于 visdom 和 tensorBoardX 的训练可视化。
    # Ignite中的主要类为Engine
    # 使用 Engine时，训练网络的时间轴为：
    # Engine是个高层封通过hook中的不同的state来链接各个模块，从而组建完整的训练过程。
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

    # 添加capsule损失
    capsule_loss = CapsuleLoss()

    # ------------------------------------------------- 通过重写state状态，来达到自定义的训练效果 -----------------------------------------------------
    # 使用egine的训练流程为：
    # 加载数据集 dataloader
    def get_iterator(mode):
        dataset = MNIST(root='./data', download=True, train=mode)
        # dataset = MNIST(root='./data', download=False, train=mode)
        # getattr() 函数用于返回一个对象属性值
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        # BATCH_SIZE 100
        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)

    def processor(sample):
        data, labels, training = sample
        # 图像增强操作augmentation
        data = augmentation(data.unsqueeze(1).float() / 255.0)
        labels = torch.LongTensor(labels)

        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes

    def reset_meters():
        # 重置模型的一些参数，在开始训练前
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()

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

        # torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

        # Reconstruction visualization.
        test_sample = next(iter(get_iterator(False)))

        ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data

        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
        reconstruction_logger.log(
            make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
