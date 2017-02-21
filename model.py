"""
  Project 3 - Model generation
  Training for controlling the steering of a car based on input images
"""
import csv
import numpy as np
from PIL import Image
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

# Global Constants
BATCH_SIZE = 16
N_EPOCHS = 5
# multiplier due to augmented data generation
# 4 implies center + mirrored + left + right images
N = 4

def generator(samples, batch_size=BATCH_SIZE):
    """
    Generator used to batch read images so that we consume reasonable
    amounts of memory reading in the images
    """
    count = 0
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for data in batch_samples:
                # load center image and generate augmented data from
                # 1. left, right images. Adding probably steering offset
                # 2. Center image flipped left to right along with -1x of steering angle
                batch_sample = data["row"]
                img_root = data["img_root"]
                name = img_root + batch_sample[0].strip()
                left_name = img_root + batch_sample[1].strip()
                right_name = img_root + batch_sample[2].strip()
                center_image = np.asarray(Image.open(name).convert('RGB'))
                center_angle = float(batch_sample[3])

                left_image = np.asarray(Image.open(left_name).convert('RGB'))
                right_image = np.asarray(Image.open(right_name).convert('RGB'))

                images.append(center_image)
                angles.append(center_angle)
                # if looks like we're too left, go right (so +ve shift)
                images.append(left_image)
                angles.append(center_angle + 0.05)

                images.append(right_image)
                angles.append(center_angle - 0.05)

                # flip center image
                images.append(np.fliplr(center_image))
                angles.append(center_angle * -1.0)
                count = count + 1

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def read_dataset(folder):
    """
        read dataset from a folder
    """
    samples = []
    csv_file = folder + '/data/driving_log.csv'
    img_root = folder + '/data/'
    with open(csv_file, 'r') as csvfile:
        next(csvfile)
        rows = csv.reader(csvfile)
        for row in rows:
            samples.append({"row": row, "img_root": img_root})
    return samples

def load_all_data():
    """
        Read training images from various sources
    """
    train_samples = []
    validation_samples = []
    data_paths = ['data/udacity', 'data/run1']
    # , 'data/track2', 'data/recovery'
    for folder in data_paths:
        print("reading folder", folder)
        # Split validation and training set for each class of data collected
        # want to make sure that the training set contains different datasets
        train, validation = train_test_split(read_dataset(folder), test_size=0.2)
        train_samples.extend(train)
        validation_samples.extend(validation)
        print("done reading folder")
    return train_samples, validation_samples

def get_model():
    """
        Get the keras model used for training
        Model based on nvidia paper
        http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

        Model source based off:
        https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py#L59

        Modifications made:
        Removed tanh activation layer at the end. It was clamping steering
        values at the output unnecessarily
        Reduced value of first Dense layer from 1184 to 256
    """
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((50, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def run():
    """
        Run the training
    """
    train_samples, validation_samples = load_all_data()
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
    print("Done reading file metadata. Training Samples", N*len(train_samples))
    model = get_model()
    model.summary()
    print("Starting training")
    model.fit_generator(train_generator, samples_per_epoch=N*len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=N*len(validation_samples),
                        nb_epoch=N_EPOCHS)
    print("Done Training")
    print("Saving Model")
    model.save('model.h5')
    print("Done")

run()











