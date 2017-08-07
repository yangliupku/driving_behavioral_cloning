import os
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split

# define the cameras and the correction angle for size cameras
cam= ['left_cam', 'center_cam', 'right_cam']
cam_correction= [0.4, 0, -0.4]


def generate_samples_from_df(df, batch_size = 128, augment=True):
    """
    sample generator used by keras. set augment = True to randomly filp the images horizontally
    """
    while True:
        df = df.sample(frac=1).reset_index(drop=True)
        for batch in np.arange(0, len(df), batch_size):
            x = np.empty([0, 160, 320, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            df_batch = df.iloc[batch: (batch + batch_size)]
            for frame in df_batch.iterrows():
                cam_rand= np.random.randint(0,3)
                img_fname = frame[1][cam[cam_rand]].split('/')[-1]
                img = cv2.imread(os.path.join('./data','IMG',img_fname))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, axis=0)
                img = img.astype(np.float32)
                # if augment:
                #     img = preprocess_img(img, np.random.uniform(-0.3,0.3))
                # else:
                #     img= preprocess_img(img,0)
                steer_angle = frame[1]['steering']+cam_correction[cam_rand]
                flip_rand = np.random.randint(0,2)
                if flip_rand == 1 and augment:
                    img = img[:,:,::-1,:]
                    steer_angle = -steer_angle
                x = np.append(x, img, axis=0)
                y = np.append(y, steer_angle)
            yield x, y

def bc_model():
    """
    define NN model 
    """
    model = Sequential()
    model.add(Cropping2D(cropping=((40, 40), (0, 0)),
         input_shape=(160, 320, 3)))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(128, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(256, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model 

if __name__=='__main__':
    # load log file and split training / test set
	df = pd.read_csv('./data/driving_log.csv', names = ['center_cam', 'left_cam', 'right_cam', 'steering',
	                                                             'throttle', 'brake', 'speed'])
	
    df_train, df_test = train_test_split(df, test_size=0.3)

    
	adam_opt=Adam(lr=5e-5)

	callbacks = [EarlyStopping(monitor='val_loss', patience=30, verbose=0),
	         ModelCheckpoint('model.hdf5', monitor='val_loss', save_best_only=True, verbose=0)]

	model = bc_model()
	model.compile(adam_opt, loss= 'mean_squared_error')

	history= model.fit_generator(generate_samples_from_df(df_train, batch_size=128),epochs=500, steps_per_epoch=int(len(df_train)/128),
	                   validation_data=generate_samples_from_df(df_test, batch_size=128, augment=False),
	                    validation_steps=int(len(df_test)/128),
	                    callbacks=callbacks)

	with open('history.p','wb') as f:
	    pickle.dump(history.history, f)