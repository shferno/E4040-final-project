"""
Command line tool for model training.
    - Usage: python fish.py [ARCH] [DATA] [--params values]
"""

# imports
import argparse
import sys
import os
from datetime import datetime
import pickle

# supress info for clearer outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tensorflow
import tensorflow as tf
from model import fishnet99, fishnet77, fishnet55

# datasets
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.datasets import mnist

# dataset generator
from imgprocess import make_dataset


def fish55(train, test, input_shape, num_cls, epochs, batch_size, lr, momentum):
    """
    Call FishNet55 Model. 
        - Arguments: 
            train: train dataset
            test: validation dataset
            input_shape
            num_cls
            epochs
            batch_size
            lr: learning rate
            momentum: momentum for SGD oprimizer
    """

    train_set, val_set = make_dataset(train, test, batch_size)
    model = fishnet55(num_cls=num_cls, input_shape=input_shape)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr, momentum=momentum), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy']
    )
    hist = model.fit(
        x=train_set, verbose=1, 
        batch_size=batch_size, epochs=epochs, 
        validation_data=val_set
    )
    return hist


def fish77(train, test, input_shape, num_cls, epochs, batch_size, lr, momentum):
    """
    Call FishNet77 Model. 
        - Arguments: 
            train: train dataset
            test: validation dataset
            input_shape
            num_cls
            epochs
            batch_size
            lr: learning rate
            momentum: momentum for SGD oprimizer
    """

    train_set, val_set = make_dataset(train, test, batch_size)
    model = fishnet77(num_cls=num_cls, input_shape=input_shape)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr, momentum=momentum), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy']
    )
    hist = model.fit(
        x=train_set, verbose=1, 
        batch_size=batch_size, epochs=epochs, 
        validation_data=val_set
    )
    return hist


def fish99(train, test, input_shape, num_cls, epochs, batch_size, lr, momentum):
    """
    Call FishNet99 Model. 
        - Arguments: 
            train: train dataset
            test: validation dataset
            input_shape
            num_cls
            epochs
            batch_size
            lr: learning rate
            momentum: momentum for SGD oprimizer
    """

    train_set, val_set = make_dataset(train, test, batch_size)
    model = fishnet99(num_cls=num_cls, input_shape=input_shape)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr, momentum=momentum), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy']
    )
    hist = model.fit(
        x=train_set, verbose=1, 
        batch_size=batch_size, epochs=epochs, 
        validation_data=val_set
    )
    return hist


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Tensorflow Fishnet Training')
    parser.add_argument('arch', metavar='ARCH', help='version of model [fishnet55|fishnet77]')
    parser.add_argument('dataset', metavar='DATA', help='name of dataset [mnist|cifar10|cifar100]')
    parser.add_argument('-e', '--epochs', default=20, type=int, metavar='N', 
                        help='total number of epochs')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='B', 
                        help='batch size (default: 32)')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, metavar='LR', 
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    args = parser.parse_args()

    # specify dataset
    if args.dataset == 'mnist':
        train, test = mnist.load_data()
        num_cls = 10
        input_shape = (56, 56, 1)
    elif args.dataset == 'cifar10':
        train, test = cifar10.load_data()
        num_cls = 10
        input_shape = (56, 56, 3)
    elif args.dataset == 'cifar100':
        train, test = cifar100.load_data()
        num_cls = 100
        input_shape = (56, 56, 3)
    else:
        print('Unsupported dataset type "{}".'.format(args.dataset))
        sys.exit()

    print('Model {} on {} (epochs={}, batch_size={}, lr={}, momentum={})'.format(
        args.arch, args.dataset, args.epochs, args.batch_size, args.learning_rate, args.momentum
    ))

    # train model
    if args.arch == 'fishnet55':
        hist = fish55(train, test, input_shape, num_cls, args.epochs, args.batch_size, 
                      args.learning_rate, args.momentum)
    elif args.arch == 'fishnet77':
        hist = fish77(train, test, input_shape, num_cls, args.epochs, args.batch_size, 
                      args.learning_rate, args.momentum)
    elif args.arch == 'fishnet99':
        hist = fish99(train, test, input_shape, num_cls, args.epochs, args.batch_size, 
                      args.learning_rate, args.momentum)
    else:
        print('Unsupported network architecture "{}".'.format(args.arch))
        sys.exit()

    # save history
    hist_path = os.path.join(
        './history',
        '{}-{}-{}'.format(args.arch, args.dataset, datetime.now().strftime('%H_%M_%S'))
    )
    with open(hist_path, 'wb') as f:
        pickle.dump(hist.history, f)
    print('History saved to {}'.format(hist_path))
