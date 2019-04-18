import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from PIL import Image
import os



def train(x_train, labels):
    num_classes = len(set(labels))
    img_col, img_row = 140, 140
    x_test = x_train[-1]
    y_test = labels[-1]
    #x_train = x_train.reshape(1999, 140,140,1)
    #x_test = x_test.reshape(1,60,60,1)
    print(labels, num_classes)
    y_train = keras.utils.to_categorical(labels, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
          
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
      
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
  
  
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
  
    model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test,y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test Accuracy:', score[1])

def get_arrays(pictures):
    pic_arr = []
    os.chdir("./train")
    i = 0
    print("go fuk yourself")
    for pic in pictures:
        i += 1
        image = Image.open(pic)
        image = image.resize((140,140), Image.ANTIALIAS)
        arr = np.array(image)
        arr = rgb2gray(arr)
        image.close()
        pic_arr.append(arr)
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
        print(pic,",", label)


def main():
    file = sys.argv[1]
    blerble = FileToArray(file)
    pics = [i[0] for i in blerble]
    labels = [i[1] for i in blerble]
    #labels = integerizeclasses(blerble[1:])
    #labels = normalizeLabels(labels)
    #createnewtrainfile(pics, labels)
    features = get_arrays(pics)
    train(features, labels)

if __name__ == "__main__":
    main()
