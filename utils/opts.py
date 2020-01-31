from __future__ import print_function
import argparse
import os


def argparser():
    P = argparse.ArgumentParser(description='Face Expression Blendshape Prediction Network (XpressNet)')
    P.add_argument('--data', type=str, default='data/300W_LP', help='path to dataset')
    P.add_argument('--epochs', type=int, default=40, help='Number of total epochs to run')
    P.add_argument('--workers', type=int, default=4, help='number of data loader threads')
    # for a single GPU.
    P.add_argument('--train-batch', type=int, default=10, help='minibatch size')
    P.add_argument('--val-batch', type=int, default=6, help='minibatch size')
    P.add_argument('-c', '--checkpoint', type=str, default='checkpoint', help='model save path')
    P.add_argument('--resume', type=str, default='', help='resume from lastest saved FAN checkpoints')

    P.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')

    P.add_argument('--momentum', type=float, default=0.0, help='momentum')
    P.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')

    P.add_argument(
        '--schedule', type=int, nargs="+", default=[15, 30], help='adjust lr at this epoch')
    P.add_argument('--gamma', type=float, default=0.1, help='lr decay')
    P.add_argument(
        '--nFeats', type=int, default=256, help='block width (number of intermediate channels)')

    P.add_argument('--scale-factor', type=float, default=0.3, help='scaling factor')
    P.add_argument('--rot-factor', type=float, default=50, help='rotation factor(in degrees)')
    P.add_argument('-e', '--evaluation', action='store_true', help='show intermediate results')

    P.add_argument(
        '--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

    P.add_argument('--device', type=str, default='cuda:0', help='device to load and run model')
    P.add_argument('--vis-freq', type=int, default=100, help='Step Frequency to output input and mesh to Tensorboard')
    P.add_argument('--loss-freq', type=int, default=100, help='Step Frequency to output training data to Tensorboard')

    P.add_argument('--no-shuffle', action='store_false', help='Do not shuffle Dataset')

    args = P.parse_args()

    return args
