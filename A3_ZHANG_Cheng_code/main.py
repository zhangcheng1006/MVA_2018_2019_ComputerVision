import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models
from torch.autograd import Variable

import copy
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default="bird_dataset", metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default="experiment", metavar='E',
                    help='folder where experiment outputs are located.')


parser.add_argument('--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule_step', type=int, default=7,
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
args.batch_size = 64
args.epochs = 2
args.lr = 0.001
args.momentum = 0.9
args.gamma = 0.1
args.dropout = 0.3
args.weight_decay = 1e-3
print(args)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms, val_transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=val_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)


# # use following function to compute mean and std for normalization
# # with transforms.Normalize turned off then change it to the printed results
# def mean_std(temp):
#     totalMean = []
#     totalStd = []
#     meanL = 0
#     stdL = 0
    
#     for batch_id,(image,label) in enumerate(temp):
#         img = image.numpy()
#         meanL = np.mean(img,axis=(1,2))
#         stdL = np.std(img,axis=(1,2))
#         totalMean.append(meanL)
#         totalStd.append(stdL)
        
#     return totalMean,totalStd

# x, y = mean_std(train_loader.dataset)
# x_mean = np.mean(x,axis = 0)
# y_mean = np.mean(y,axis = 0)
# print(x_mean,y_mean)


def train_model(model, criterion, optimizer, scheduler):
    scheduler.step()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation(model, criterion):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        acc))

    return acc

from model import SimpleNet, TransferLearning 
# get model and replace the original fc layer with your fc layer
num_classes = 20

# # use simple network
# model_ft = SimpleNet()

# use resnet
model_ft = TransferLearning(netname="resnet")

# # use vgg
# model_ft = TransferLearning(netname="vgg")

# # use densenet
# model_ft = TransferLearning(netname="densenet")

# # use squeezenet
# model_ft = TransferLearning(netname="squeezenet")

# # use alexnet
# model_ft = TransferLearning(netname="alexnet")

# # use inception v3
# model_ft = TransferLearning(netname="inceptionv3")


if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

if use_cuda:
    model_ft = model_ft.cuda()

# define loss function
criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(list(filter(lambda p: p.requires_grad, model_ft.parameters())), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.schedule_step, gamma=args.gamma)

# train and save the best model which has the best accuracy
best_model = model_ft
best_acc = 0.0
for epoch in range(1, args.epochs + 1):
    train_model(model=model_ft,
                criterion=criterion,
                optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler)
    acc = validation(model=model_ft, criterion=criterion)
    if acc > best_acc:
        best_acc = acc
        best_model = copy.deepcopy(model_ft)

model_file = args.experiment + '/best_model' + '.pth'
torch.save(best_model.state_dict(), model_file)
print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')

