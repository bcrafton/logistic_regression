
import numpy as np
import math
import gzip
import time
import pickle
import argparse
import keras
from keras.datasets import mnist

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--scale', type=float, default=2.0)
parser.add_argument('--low', type=float, default=1e-2)
args = parser.parse_args()

#######################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10

#######################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

#######################################

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#######################################

print (args.epochs, args.alpha, args.scale, args.low)

LAYER1 = 784
LAYER2 = 10

# remember there is a sigmoid ... so it makes things a little nicer
# maybe we should use softmax, but sigmoid seems to work fine.
# trying a bunch of learning rates also a good idea
low = args.low
high = args.low * args.scale

weights = np.random.uniform(low, high, size=(LAYER1, LAYER2))
bias = np.zeros(shape=(LAYER2))

#######################################

accs = []

for epoch in range(args.epochs):
    print ("epoch: " + str(epoch + 1) + "/" + str(args.epochs))
    
    for ex in range(TRAIN_EXAMPLES):
        A1 = x_train[ex]
        Z2 = np.dot(A1, weights) + bias
        A2 = softmax(Z2)
        
        ANS = y_train[ex]
        E = A2 - ANS
        
        DW = np.dot(A1.reshape(LAYER1, 1), E.reshape(1, LAYER2))
        DB = E
        
        weights -= args.alpha * DW
        bias -= args.alpha * DB
        
        weights = np.clip(weights, low, high)
        
    correct = 0
    for ex in range(TEST_EXAMPLES):
        A1 = x_test[ex]
        Z2 = np.dot(A1, weights) + bias
        A2 = sigmoid(Z2) 
        
        if (np.argmax(A2) == np.argmax(y_test[ex])):
            correct += 1
        
    acc = 1.0 * correct / TEST_EXAMPLES
    print ("accuracy: " + str(acc))
    accs.append(acc)
    print (np.min(weights), np.max(weights), np.average(weights), np.std(weights))

name = "./results/epochs_%d_alpha_%f_scale_%f_low_%f.npy"% (args.epochs, args.alpha, args.scale, args.low)
np.save(name, accs)

    
    
    
    
