"""
Image preprocessing.
    Functions: 
        - resize(x, shape)
        - plot(x, y)
        - preprocess
        - augmentation
        - make datasets
"""


from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Data Resize and Normalization
def resize(x, shape = (224,224,3)):
    '''
    Convert data shape to the input shape for fishnet
    '''

    # RGB datasets like cifar
    if shape[2] == 3:
        x_resized = np.zeros((x.shape[0],shape[0],shape[1],shape[2])) # Create a zero matrix for resized data
        for i in range(x.shape[0]):
            img = x[i]
            img = Image.fromarray(img)
            img = np.array(img.resize((shape[0],shape[1]), Image.BICUBIC)) # Resize and convert back to array type
            x_resized[i,:,:,:] = img
    # Gray datasets like mnist
    elif shape[2] == 1:
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))
        x_resized = np.zeros((x.shape[0],shape[0],shape[1])) # Create a zero matrix for resized data
        for i in range(x.shape[0]):
            img = x[i]
            img = Image.fromarray(img, mode='L') # Convert array type to image type
            img = np.array(img.resize((shape[0],shape[1]), Image.BICUBIC)) # Resize and convert back to array type
            x_resized[i,:,:] = img
        x_resized = x_resized.reshape((x.shape[0],shape[0],shape[1],shape[2]))
    x_resized /= 255 # Normalization
    return x_resized

# Show a plot
def plot(x,y):
    """
    Plot random images for verification
    """

    fig = plt.figure(figsize=(8,8))
    plt.subplots_adjust(top=1.2)
    col = 4
    row = 4
    print(str(col*row)+" Random images and their labels")
    ri = random.sample(range(0,len(x)),col*row)
    rii = 0
    for i in range(1, col*row+1):
        img = np.copy(x[ri[rii]])
        img = np.asarray(img, dtype=np.float32)
        fig.add_subplot(row,col,i)
        x_label = 'Label : ' + str(y[ri[rii]])
        plt.xlabel(x_label)
        img = img/np.amax(img)
        img = np.clip(img, 0, 1)
        plt.imshow(img, interpolation = 'nearest')
        rii = rii + 1
    _ = plt.show()


# Resize
def preprocess(img, label):
    """preprocess images for dataset"""
    img = tf.image.resize(img, [56, 56])
    return img, label


# Augmentation
def augmentation(img, label):
    """do augmentation for dataset"""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.5, 1.5)
    if img.shape[-1] == 3:
        img = tf.image.random_hue(img, 0.2)
        img = tf.image.random_saturation(img, 0.5, 5)
    return img, label


# Make datasets, preprocess and do augmentation
def make_dataset(train, test, batch_size=32):
    """make dataset"""

    X_train, y_train = train
    X_test, y_test = test
    
    # make train
    train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_set = train_set.shuffle(X_train.shape[0])
    train_set = train_set.batch(batch_size)
    train_set = train_set.map(preprocess)
    train_set = train_set.map(augmentation)
    train_set = train_set.prefetch(1)
    
    # make validation
    val_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_set = val_set.shuffle(X_test.shape[0])
    val_set = val_set.batch(batch_size)
    val_set = val_set.map(preprocess)
    val_set = val_set.prefetch(1)
    
    return train_set, val_set