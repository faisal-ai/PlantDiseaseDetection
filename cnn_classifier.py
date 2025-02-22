import keras
from keras import backend as  k
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D , MaxPooling2D , Dense , Dropout , Flatten , Activation
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import cv2
from google.colab.patches import cv2_imshow as imshow
import pickle
import os
import zipfile
import tensorflow as tf

print(keras.__version__)
print(tf.__version__)

with zipfile.ZipFile('/content/leaves_data_set.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/sample_data')

direc = [ '/content/Apple___healthy' , '/content/Apple___infected']
    
infected , healthy = [] , []

for file in direc:
  for img in os.listdir( file ):
    img_path = file +'/'+ img
    if file.split('/')[-1].split('___')[-1] == 'healthy':
      healthy.append( ( cv2.imread(img_path , 1 ) , 0) )
    if file.split('/')[-1].split('___')[-1] == 'infected':
      infected.append(  ( cv2.imread(img_path , 1) , 1) ) 

print()
print(healthy[-1])
print()
print(infected[-1])

for i in range(3):
  print( healthy[i][-1] )
  imshow( healthy[i][0] )
  print( infected[i][-1] )
  imshow( infected[i][0] )

for i in infected:
  healthy.append( i )

for i in range( 3 ):
  imshow( healthy[i+1600][0] )
  print( healthy[i+1600][-1] )

print( '\n'*2 )
print(len(healthy))

from random import shuffle
shuffle( healthy )

inf , heal = 0 , 0
for i in healthy:
  if i[-1] == 1:
    inf+=1
  if i[-1] == 0:
    heal+=1

print(inf , heal)

from sklearn.model_selection import train_test_split
images , labels = [] , []

for i in healthy:
  images.append( i[0] )
  labels.append( i[-1] )

images , labels = np.array(images) , np.array(labels)
print( images.shape , labels.shape )

x_train , x_test , y_train , y_test = train_test_split(images , labels , test_size=0.2)
print(x_train.shape , y_train.shape)

print(x_test.shape , y_test.shape)

from keras.layers.normalization import BatchNormalization
# x_train = x_train/255.0
# x_test = x_test/255.0

if k.image_data_format() == 'channels_first':
    input_shape = (3, 256, 256 ) 
else:
    input_shape = (256, 256 , 3)

print(input_shape)

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape = input_shape  ))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D(2, 2))

model.summary()

model.add( Flatten() )
model.add( Dense(32 , activation='relu') )

# model.add( Dense(64 , activation='relu') )
model.add( Dropout( 0.5 ) )

model.add( Dense(1 , activation='sigmoid') )
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(x_train , y_train , epochs=4 , validation_data=(x_test, y_test))
loss , acc = model.evaluate( x_test , y_test )

print(acc , loss)

predictions = model.predict( x_test )

error = 0
for i in range( len(predictions) ):
  if  np.round(predictions[i])[0]  != y_test[i]:
    # print('ERROR')
    error+=1

# print( np.round(predictions[i]) , y_test[i])

print( len(predictions) , error )


# pred = model.predict(x)
# print( np.round(pred) )











