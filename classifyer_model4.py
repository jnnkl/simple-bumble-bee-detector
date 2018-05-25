#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:21:45 2018

@author: bombus
"""


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json


# the folders with the data 
train_data_dir = 'small_images/train/'
test_data_dir ='small_images/test/'



model = Sequential()
model.add(Conv2D(48, (4, 4),padding='same', input_shape=( 45, 45,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()
#%%

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#%%
batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.02,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True,
                                  vertical_flip=True)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(45, 45),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(45, 45),
        batch_size=batch_size,
        class_mode='binary')

history=model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

#save the model as json
model_json = model.to_json()
with open("classifyer/model_classifyer4_april.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("classifyer/model_classifyer4_april.h5")
print("Saved model to disk")
 

import matplotlib.pyplot as plt
history = history.history()
val_loss=history['val_loss']
val_acc=history['val_acc']
loss=history['loss']
acc=history['acc']

plt.figure()
plt.title('loss')
plt.plot(loss, label='loss')
plt.plot(val_loss,label='val_loss')
plt.legend()
plt.show()

plt.figure()
plt.title('accuracy')
plt.plot([a for a in acc], label='acc')
plt.plot([a for a in val_acc],label='val_acc')
plt.legend()
plt.show()

