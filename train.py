import argparse

import torch
import visdom
import numpy as np
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm


from datasets.VOC import pascalVOCLoader
from models.SegNet import SegNet_VGG16
from models.modules import CrossEntropyLoss2d


def train(args):
    # Setup data for training
    train_loader = pascalVOCLoader('/home/pkovacs/Documents/data/VOCdevkit/VOC2012', is_transform=True, split='train')
    train_data = data.DataLoader(train_loader, batch_size=args.batch_size, num_workers=5, shuffle=True)
    CLASSES = train_loader.n_classes

    # Setup validation data
    val_loader = pascalVOCLoader('/home/pkovacs/Documents/data/VOCdevkit/VOC2012', is_transform=True, split='val')
    val_data = data.DataLoader(val_loader, batch_size=args.batch_size, num_workers=5, shuffle=True)

    # Setup visdom for visualization
    vis = visdom.Visdom()
    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           env="loss",
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Loss',
                                     legend=['Train']))

    # Setup model
    model = SegNet_VGG16(n_classes=CLASSES, n_channels=3, pretrained=True)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, weight_decay=5e-4)

    # Loss Function
    # TODO add weights for classes
    weight = torch.ones(CLASSES)
    weight[0] = 0

    # If cuda (GPU) device available compute on it
    if torch.cuda.is_available():
        model.cuda()
        loss_fun = CrossEntropyLoss2d(weight=weight.cuda(), average=True)
    else:
        loss_fun = CrossEntropyLoss2d(weight=weight)

    for epoch in range(args.n_epoch):
        print("Epoch {}/{}:".format(epoch, args.n_epoch))
        for i, (images, labels) in enumerate(train_data):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            # Compute model output
            outputs = model(images)

            # Compute loss function
            loss = loss_fun(outputs, labels)

            # Before the backward pass, use the optimizer object to zero all of the gradients
            # for the variables it will update (which are the learnable weights of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            it = len(train_loader) * epoch + i

            # Visualise train loss function
            print(np.array([it]))
            print(np.array([loss.data[0]]))
            vis.updateTrace(
                X=np.array([it]),
                Y=np.array([loss.data[0]]),
                env="loss",
                win=loss_window,
                name='Train')

            # Log image results and validation
            if it % args.log_freq == 0:
                print("VALIDATION")
                validation_losses = []
                for images, labels in val_data:
                    if torch.cuda.is_available():
                        # volatile means that you will not compute backward pass - only inference
                        images = Variable(images.cuda(), volatile=True)
                        labels = Variable(labels.cuda(), volatile=True)
                    else:
                        images = Variable(images, volatile=True)
                        labels = Variable(labels, volatile=True)
                    outputs = model(images)
                    val_loss = loss_fun(outputs, labels)
                    validation_losses.append(val_loss.data[0])
                    break

                print(np.array([it]))
                print(np.array([np.mean(validation_losses)]))
                # Visualise val loss function
                vis.updateTrace(
                    X=np.array([it]),
                    Y=np.array([np.mean(validation_losses)]),
                    env="loss",
                    win=loss_window,
                    name='Validation')


                # TODO visualise images



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--port', type=int, default=8097,
                        help='Port for Visdom server -> result logging')
    parser.add_argument('--log_freq', type=int, default=20,
                        help='Frequency of logging of segmentation results to Visdom.')
    args = parser.parse_args()
    train(args)
