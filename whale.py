import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from PIL import Image
import os

def createModel(img_row, img_col, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(img_row,img_col,1)))
      
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
  
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    return model

def train(x_train, labels, img_row, img_col, num_classes, sample_size):
    feature_length = 2*(0 - img_row) 
    x_test = x_train[feature_length:]
    y_test = labels[-2:]
    x_train = x_train.reshape(sample_size, img_row,img_col,1)
    x_test = x_test.reshape(2,img_row,img_col,1)
    print(labels, num_classes)
    y_train = keras.utils.to_categorical(labels, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
          
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(img_row,img_col,1)))
      
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
  
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
  
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1) #, validation_data=(x_test, y_test)
    score = model.evaluate(x_test,y_test, verbose=0)
    print('Test loss:', score[0]) 
    print('Test Accuracy:', score[1])

def TrainIteration(feature, label, num_classes, img_row, img_col, model):
    feature = feature.reshape(1, img_row,img_col,1)
    print(feature.shape)
    label = keras.utils.to_categorical(label, num_classes)
    print(label.shape)
    model.fit(feature, label, batch_size=1, epochs=1, verbose=1) #, validation_data=(x_test, y_test)


def IterTrain(pictures, labels, img_row, img_col, num_classes, sample_size):
    #pic_arr = np.empty([120000, 60])
    os.chdir("./train")
    i = 0
    print("go fuk yourself")
    model = createModel(img_col, img_row, num_classes)
    for pic,label in zip(pictures[:sample_size],labels):
        i += 1
        image = Image.open(pic)
        image = image.resize((img_row,img_col), Image.ANTIALIAS)
        arr = np.array(image)
        arr = rgb2gray(arr)
        TrainIteration(arr,label,num_classes,img_row,img_col,model)
        image.close()
        #np.append(pic_arr, arr)
        if i % 500 == 0:
            print(i)
    return model
    #return pic_arr

def getFeatureArray(pictures, sample_size, img_row, img_col):
    pic_arr = np.empty([(sample_size * img_row), img_col])
    os.chdir("./train")
    i = 0
    print("go fuk yourself")
    for pic in pictures[:sample_size]:
        i += 1
        image = Image.open(pic)
        image = image.resize((img_row,img_col), Image.ANTIALIAS)
        arr = np.array(image)
        arr = rgb2gray(arr)
        image.close()
        np.append(pic_arr, arr)
        if i % 500 == 0:
            print(i)

    return pic_arr


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def FileToArray(file):
    with open(file, 'r') as f:  
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
    return data

def integerizeclasses(labels):
    newlabels = []
    for num in labels:
        if num[1] == "new_whale":
            newlabels.append(0)
        else:
            temp = float.fromhex(num[1][2:])
            newlabels.append(int(temp))
    return newlabels

def normalizeLabels(labels):
    newlabels = []
    temp = list(set(labels))
    temp.sort()
    for label in labels:
        newlabel = temp.index(label)
        newlabels.append(newlabel)
    print(newlabels)
    return newlabels

def createnewtrainfile(pics, labels):
    sys.stdout = open("newtrain.csv", "w")
    for pic, label in zip(pics, labels):
        print(pic+","+str(label))

def SaveModelToJSon(model): #Save and load methods copied from the internet
    model_json = model.to_json()
    with open(".\model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("./model.h5")
    print("Saved model to disk")

def LoadModelFromToJSon():
    json_file = open('.\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

def main():
    trainfile = sys.argv[1]
    #testfile = sys.argv[1]
    img_col, img_row = 60, 60
    sample_size = 2000
    blerble = FileToArray(trainfile)
    pics = [i[0] for i in blerble]
    labels = [i[1] for i in blerble]
    #labels = integerizeclasses(blerble[1:])
    labels = labels[:sample_size]
    labels = normalizeLabels(labels)
    num_classes = len(set(labels))
    #createnewtrainfile(pics, labels)
    features = getFeatureArray(pics, sample_size, img_row, img_col)
    train(features, labels, img_row, img_col, num_classes, sample_size)

if __name__ == "__main__":
    main()