# !/usr/local/bin/python3
import os
import time
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data import CustomDataset

from net import get_model


######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data-path', default='../data', type=str, help='path to the dataset')
parser.add_argument('--model-name', default='nf_resnet50', type=str, help='name of model')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num-epoch', default=50, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=8, type=int, help='num_workers')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
parser.add_argument('--cuda', action='store_true', help='use GPU')
parser.add_argument('--device-id', default=0, type=int, help='device training cuda 0, 1, 2, ...')
args = parser.parse_args()


data_dir = args.data_path
use_gpu = args.cuda
device = "cuda:" + str(args.device_id) if use_gpu else "cpu"
model_name = args.model_name
model_dir = os.path.join('./checkpoints', model_name)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)


######################################################################
# Function
# --------
def save_network(network, epoch_label):
    save_filename = 'epoch_%s.pth'% epoch_label
    save_path = os.path.join(model_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    network.to(device)
    print('Save model to {}'.format(save_path))


######################################################################
# Draw Curve
#-----------
x_epoch = []
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(model_dir, 'train.jpg'))


######################################################################
# DataLoader
# ---------
csv_pa100k = "../data/phase1/train/train_1.csv" # 79001 images
csv_market_1501 = "../data/phase1/train/train_0.csv" # 10000 images
csv_peta = "../data/phase1/train/train_2.csv" # 8668 images
img_dir = "../data"
image_datasets = {}

image_datasets['train'] = CustomDataset(csv_path=csv_pa100k, image_dir=img_dir)
image_datasets['val'] = CustomDataset(csv_path=csv_market_1501, image_dir=img_dir)
image_datasets['test'] = CustomDataset(csv_path=csv_peta, image_dir=img_dir)
concatenated_train_dataset = torch.utils.data.ConcatDataset([image_datasets['train'], image_datasets['test']])
image_datasets['train'] = concatenated_train_dataset
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers, drop_last=True)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

######################################################################
# Model and Optimizer
# ------------------
model_name = args.model_name
num_label = 40
model = get_model(model_name, num_label)
if use_gpu:
    model = model.to(device)

# loss
criterion_bce = nn.BCELoss()
criterion_ce = nn.CrossEntropyLoss()

# optimizer
classifier_params = []
for name, param in model.named_parameters():
    if 'class_' in name:
        classifier_params.append(param)
optimizer = torch.optim.SGD([
            {'params': model.features.parameters(), 'lr': 0.002},
            {'params': classifier_params, 'lr': 0.01},
        ],
        momentum=0.9, 
        weight_decay=5e-3, nesterov=True
        )
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


######################################################################
# Training the model
# ------------------
def train_model(model, optimizer, scheduler, num_epochs):
    since = time.time()

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for count, (images, labels) in enumerate(dataloaders[phase]):
                # get the inputs
                labels = labels.float()
                if use_gpu:
                    images = images.to(device)
                    labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                pred_label = model(images)
                total_loss = criterion_bce(pred_label, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

                preds = torch.gt(pred_label, torch.ones_like(pred_label)/2 )
                # statistics
                running_loss += total_loss.item()
                running_corrects += torch.sum(preds == labels.byte()).item() / num_label
                if count % 100 == 0:
                    if not args.use_id:
                        print('step: ({}/{})  |  label loss: {:.4f}'.format(
                            count*args.batch_size, dataset_sizes[phase], total_loss.item()))
                    else:
                        print('step: ({}/{})  |  label loss: {:.4f}  |  id loss: {:.4f}'.format(
                            count*args.batch_size, dataset_sizes[phase], label_loss.item(), id_loss.item()))

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 0:
                    save_network(model, epoch)
                draw_curve(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')


######################################################################
# Main
# -----
train_model(model, optimizer, exp_lr_scheduler, num_epochs=args.num_epoch)
