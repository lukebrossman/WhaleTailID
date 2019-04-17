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

def get_arrays(welp):
    pic_arr = []
    os.chdir("./train")
    i = 0
    print("go fuk yourself")
    for pic in welp[1:]:
        i += 1
        image = Image.open(pic[0])
        image = image.resize((60,60), Image.ANTIALIAS)
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

def main():
    file = sys.argv[1]
    blerble = FileToArray(file)
    arrays = get_arrays(blerble)
    #print(arrays)

if __name__ == "__main__":
    main()
