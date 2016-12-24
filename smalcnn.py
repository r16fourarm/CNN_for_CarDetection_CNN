import keras.backend as K
K.set_image_dim_ordering('th')
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import scipy.io as sio
import cv2,numpy as np
import sys
from keras.utils import np_utils
#from imagenet_utils import decode_predictions
#from imagenet_utils import preprocess_input


def totgtmat(d2mat,filename):
    mat= np.argmax(d2mat,axis=1)
    np.savetxt(filename,delimiter=',',X=d2mat)
    return mat
# dimensions of our images.

img_width, img_height = 224, 224

train_data_dir = 'E:/testCNN/train'
validation_data_dir = 'E:/testCNN/val'
prediction_data_dir = 'E:/testCNN/sliding2_'
nb_train_samples = 6000
nb_validation_samples = 2280
#nb_epoch = 50

#batch_size = 40
nb_epoch =50
    
#    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#load dataset
#dataset= sio.loadmat('E:/testCNN/datasetV2a.mat')

#dataset_=dataset["datasetV2a"].T
#
##skalar isi targtet
#target=np.zeros((1200,))
#target[0:600]=[1 for i in range(600)]
#
#(X_train, y_train), (X_test, y_test) = (dataset_,target),(dataset_,target)
#
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
#
#Y_train = np_utils.to_categorical(y_train, 2)
#Y_test = np_utils.to_categorical(y_test, 2)
#
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255

#bikinstruktur
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height))) #conv layer output 32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #maxpool 2x2

model.add(Convolution2D(32, 3, 3))#ouput 32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #maxpool 2x2 

model.add(Convolution2D(64, 3, 3))#output 32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))#maxpool 2x2

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

#model.fit(X_train, Y_train,
#              batch_size=batch_size,
#              nb_epoch=nb_epoch,
#              validation_data=(X_test, Y_test),
#              shuffle=True)

#save bobot
#model.save_weights("model_fchollet2.h5")
#model.load_weights("model_fchollet.h5")#load bobot
outss=np.zeros((1,2))
model.load_weights("saved_weights\model_fcholletdup--.h5")
#print(sys.argv[1])
for  i  in range(int(sys.argv[1])):
    im = cv2.resize(cv2.imread('G:\\cnn\\CNN_FinalTask\\out\\window'+str(i+1)+'.jpg'), (224, 224)).astype(np.float32)
##    im=datap[i].astype(np.float32)
#    im=cv2.imread('C:\\Users\\user\\Downloads\\car224.jpg')
##    #cv2.imshow('image',cv2.imread('C:\\Users\\user\\Downloads\\car224.jpg'))
##    #cv2.waitKey()
##    im[:,:,0] -= 103.939
##    im[:,:,1] -= 116.779
##    im[:,:,2] -= 123.68
    im/=255
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    
##    #cv2.imshow("im",im)
##    #
    out = model.predict(im)
    outss=np.append(outss,out,axis=0)
outss=np.delete(outss,0,axis=0)
if np.size(outss) == 0:
  print("error dimensi  window")
  quit()
np.savetxt("CarProbability.csv",X=outss,delimiter=',')
print("done")
#out1 = model.predict(datap.astype(np.float32))
#print (np.argmax(out))
#yang dibawah untuk preprocessing dari gambar asli (blum dicoba)

# this is the augmentation configuration we will use for training
#train_datagen = ImageDataGenerator(
#        rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        vertical_flip=True,
#        width_shift_range=0.2,
#        height_shift_range=0.2)

# this is the augmentation configuration we will use for testing:
# only rescaling
#test_datagen = ImageDataGenerator(rescale=1./255)
##
#predict_datagen = ImageDataGenerator(rescale=1./255)
#predict_generator = test_datagen.flow_from_directory(
#       prediction_data_dir,
#        target_size=(img_width, img_height),
#        batch_size=40,
#        class_mode=None,
#        shuffle=False
#        )
#train_generator = train_datagen.flow_from_directory(
#        train_data_dir,
#        target_size=(img_width, img_height),
#        batch_size=40,
#        classes=['car','noncar'],
#        class_mode='categorical')
###
#validation_generator = test_datagen.flow_from_directory(
#        validation_data_dir,
#        target_size=(img_width, img_height),
#        batch_size=40,
#        classes=['car','noncar'],
#        class_mode='categorical')
#
#history=model.fit_generator(
#        train_generator,
#        samples_per_epoch=nb_train_samples,
#        nb_epoch=nb_epoch,
#        validation_data=validation_generator,
#        nb_val_samples=nb_validation_samples)
#
#model.save_weights("model_112cat3.h5")
#outs=model.predict_generator(predict_generator,val_samples=266)
#
