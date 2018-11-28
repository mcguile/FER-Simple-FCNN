import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt

#------------------------------
#cpu - gpu configuration
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 
keras.backend.set_session(sess)
#------------------------------
#variables
num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 100
epochs = 150
#------------------------------
#read kaggle facial expression recognition challenge dataset (fer2013.csv)
#https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

with open("fer2017-training.csv") as f1:
    train_content = f1.readlines()

with open("fer2017-testing.csv") as f2:
    test_content = f2.readlines()
    


train_lines = np.array(train_content)
test_lines = np.array(test_content)

train_instances = train_lines.size
test_instances = test_lines.size

#------------------------------
#initialize trainset and test set
x_train, y_train, x_test, y_test = [], [], [], []

#------------------------------
#transfer train and test set data
for i in range(1,train_instances):
    try:
        emotion, img = train_lines[i].split(",")
          
        val = img.split(" ")
            
        pixels = np.array(val, 'float32')
        
        emotion = keras.utils.to_categorical(emotion, num_classes)
        
        y_train.append(emotion)
        x_train.append(pixels)
    except:
        print("",end="")
        
for i in range(1,test_instances):
    try:
        emotion, img = test_lines[i].split(",")
          
        val = img.split(" ")
            
        pixels = np.array(val, 'float32')
        
        emotion = keras.utils.to_categorical(emotion, num_classes)
        
        y_test.append(emotion)
        x_test.append(pixels)
    except:
        print("",end="")

#------------------------------
#data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

#------------------------------
#construct CNN structure
def constructModel():
    model = Sequential()
    model.add(Flatten())
    #fully connected neural network
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))
    #------------------------------

    return model
   
    #------------------------------

def evaluate():
    # Overall evaluation
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss:', train_score[0])
    print('Train accuracy:', 100*train_score[1])

    test_score = model.evaluate(x_test, y_test)
    print('Test loss:', test_score[0])
    print('Test accuracy:', 100*test_score[1])
    
    
    #------------------------------

def plotCurves(history):
    # Only available when training, not loading saved model
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    # Accuracy curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    
    #------------------------------
    
#------------------------------
# Main Program #
#------------------------------
    
model = constructModel()

model.compile(loss='categorical_crossentropy'
                , optimizer=keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)
                , metrics=['accuracy']
            )
  
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test))

evaluate()
plotCurves(history)
