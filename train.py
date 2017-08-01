import argparse

import numpy as np
import torch
import visdom
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from datasets.VOC import pascalVOCLoader
from models.SegNet import SegNet_VGG16
from models.modules import cross_entropy2d


def train(args):
    # Setup data for training
    train_loader = pascalVOCLoader('/home/pkovacs/Documents/data/VOCdevkit/VOC2012', is_transform=True, split='train')
    train_data = data.DataLoader(train_loader, batch_size=args.batch_size, num_workers=4, shuffle=True)
    CLASSES = train_loader.n_classes

    # Setup validation data
    val_loader = pascalVOCLoader('/home/pkovacs/Documents/data/VOCdevkit/VOC2012', is_transform=True, split='val')
    val_data = data.DataLoader(val_loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # Setup visdom for visualization
    viz = visdom.Visdom()
    loss_window = viz.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Loss',
                                     legend=['Train']))

    # Setup model
    model = SegNet_VGG16(n_classes=CLASSES, n_channels=3, pretrained=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, weight_decay=5e-4)

    # Loss Function
    weight = torch.FloatTensor(train_loader.class_weights)
    # weight = torch.ones(21)
    # weight[0] = 0

    # If cuda (GPU) device available compute on it
    if torch.cuda.is_available():
        model.cuda()
        weight = weight.cuda()

    for epoch in range(args.n_epoch):
        print("Epoch {}/{}:".format(epoch, args.n_epoch))
        for i, (images, labels) in tqdm(enumerate(train_data), desc="Epoch"):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            # Compute model output
            outputs = model(images)

            # Compute loss function
            loss = cross_entropy2d(outputs, labels, weight=weight)  # loss_fun(outputs, labels)

            # Before the backward pass, use the optimizer object to zero all of the gradients
            # for the variables it will update (which are the learnable weights of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            it = (len(train_data) * epoch) + i

            # Send train and validations losses to Visdom and display them
            if it % args.log_freq == 0:

                # Visualise train loss function
                viz.updateTrace(
                    X=np.array([it]),
                    Y=np.array([loss.data[0]]),
                    win=loss_window,
                    name='Train')

                validation_losses = []
                for i2, (images, labels) in enumerate(val_data):
                    if torch.cuda.is_available():
                        # volatile means that you will not compute backward pass - only inference
                        images = Variable(images.cuda(), volatile=True)
                        labels = Variable(labels.cuda(), volatile=True)
                    else:
                        images = Variable(images, volatile=True)
                        labels = Variable(labels, volatile=True)
                    outputs = model(images)
                    val_loss = cross_entropy2d(outputs, labels, weight=weight)  # loss_fun(outputs, labels)
                    validation_losses.append(val_loss.data[0])

                    if i2 * args.batch_size > 50:
                        break

                # Visualise val loss function
                viz.updateTrace(
                    X=np.array([it]),
                    Y=np.array([np.mean(validation_losses)]),
                    win=loss_window,
                    name='Validation')

        # VISUALIZE IMAGES AFTER EACH EPOCH
        for i, (images, labels) in enumerate(val_data):
            if torch.cuda.is_available():
                images = Variable(images.cuda(), volatile=True)
                labels = Variable(labels.cuda(), volatile=True)
            else:
                images = Variable(images, volatile=True)
                labels = Variable(labels, volatile=True)
            outputs = model(images)

            if images.is_cuda:
                images = images.cpu()
                labels = labels.cpu()
            if isinstance(images, Variable):
                images = images.data
                labels = labels.data
            images = images.numpy()
            labels = labels.numpy()

            for j, img in enumerate(images):
                # Real image
                img = img.transpose(1, 2, 0)
                img *= [0.229, 0.224, 0.225]
                img += [0.485, 0.456, 0.406]
                img = img.transpose(2, 0, 1)
                viz.image(img, env='epoch_{}'.format(epoch), opts=dict(caption='Image_{}'.format(j)))

                # Ground truth segmentation
                decoded_label = np.transpose(val_loader.decode_segmap(labels[j]), [2,0,1])
                viz.image(decoded_label, env='epoch_{}'.format(epoch), opts=dict(caption='GT_{}'.format(j)))

                # Predicted segmentation
                out_img = np.transpose(train_loader.decode_segmap(outputs[j].cpu().data.numpy().argmax(0)), [2,0,1])
                viz.image(out_img, env='epoch_{}'.format(epoch), opts=dict(caption='Predicted_{}'.format(j)))

            # We dont want all validation images so stop if we have more than 10
            if i * args.batch_size >= 10:
                break

        # Save model to disk
        torch.save(model, "segnet_VOC_epoch_{}.pkl".format(epoch))


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
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Frequency of logging of segmentation results to Visdom.')
    args = parser.parse_args()
    train(args)
