import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import os


os.chdir("../../pics")
for pic in os.listdir('./'):
    print(pic)
    
        


image = Image.open('test.jpg')
arr = np.array(image)
arr[20,30]

num_classes = 10
epochs = 12

#image dimensions
img_rows, img_cols = 28, 28

#the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(arr)
plt.imshow(arr)
#plt.imshow("pics/0a0c1df99.jpg")
plt.show()
