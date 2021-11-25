## MY FIRST NEURAL NETWORK !!! WITH ACCURACY 93%
## This NN can tell whether an image is a triangle or not

# increase the shapes' sizes => increase accuracy (easier to recognize)
# introducing noise (50,20,20) => decrease accuracy by ~3%
# using tanh as activation function instead of sigmoid => the result converge faster
# increase nepoch => increase accuracy
# still 2 hidden layers, 1 layer has more neurons => increase accuracy, slower training speed
# decrease isize => faster training speed
# => use average pooling to increase the training speed!!! (IMPLEMENT THIS NOW)
# ReLU seems to have much careful steps, but the convergence is not as fast as tanh
# using (tanh, tanh, sigmoid) can achieve the error rate of 0.0005 in 300 epochs :) and error rate of 0.0 after 400 epochs on training set :)

##import os
##os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"
##import cupy as cp

import numpy as np
import random
import time
from winsound import Beep

# generate a random 64x64 image of a shape: square, triangle

def genSquare(size=64):
    ssize = random.randint(size//3,size) # square size
    image = np.zeros((size,size))
    sx, sy = [random.randint(0, size-ssize) for _ in range(2)]
    image[sx:sx+ssize, sy:sy+ssize] = 1
    return image

def genTriangle(size=64): # only odd length base
    tsize = random.randint(size>>2,(size+1)>>1) # the height
    image = np.zeros((size,size))
    sx = random.randint(0, size-tsize)
    sy = random.randint(0, size - (tsize << 1) + 1)
    for i in range(tsize):
        image[sx+i, sy+tsize-i-1:sy+tsize+i] = 1
    return image

def genNoise(size=64):
    return np.random.choice([0., 1.], (size,size))

def genLabeledDataset(n, size=64): # 0.5 triangle, 0.25 square, 0.25 noise
    data, labels = [], []
    for _ in range(n):
        c = random.getrandbits(2)
        if c >> 1:
            data.append(genTriangle(size))
            labels.append(1)
        else:
            if c & 1:
                data.append(genSquare(size))
            else:
                data.append(genNoise(size))
            labels.append(0)
    return np.array(data), np.array(labels)[:, np.newaxis, np.newaxis] # this choice is because we are doing binary classification, last layer only have 1 neuron

def draw(shape):
    image = '\n'.join(map(lambda row : ''.join(map(lambda pixel : '▓' if pixel else '░', row)), shape))
    print(image)

def sigmoid(x, deriv=False):
    sigmoid_ = np.reciprocal(1 + np.exp(-x))
    if deriv:
        return sigmoid_ * (1 - sigmoid_)
    return sigmoid_

def tanh(x, deriv=False):
    if deriv:
        return np.reciprocal(np.cosh(x)**2)
    return np.tanh(x)

def ReLU(x, deriv=False):
    if deriv:
        return x > 0
    return x * (x > 0)

def main():
    startTime = time.time()
    
    print('--- Initializing ---')
    dsize, tsize, isize = 10000, 100, 10 # train & test may overlap
    nepoch = 5
    step = 1 # length of the step to take in gradient descent
    print('Training set size: ' + str(dsize) + ' images')
    print('Testing set size: ' + str(tsize) + ' images')
    print('Image size: ' + str(isize) + 'x' + str(isize))
    print('#epochs:', nepoch)
    print('Step:', step)

    # we use a NN that have 4 layers with config (isize**2, 32, 32, 1), each number represent the number of neurons in each layer
    n1, n2 = 64, 64
    print('--- Initialize the neural network (' + str(n1) + ',' + str(n2) + ',1) ---')
    w1 = np.random.randn(n1, isize**2)
    b1 = np.random.randn(n1, 1)
    w2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)
    w3 = np.random.randn(1, n2)
    b3 = np.random.randn(1, 1)

    print('--- Generating dataset ---')
    images, labels = genLabeledDataset(n=dsize, size=isize)

    # show an example from the generated dataset
    print('--- Example ---')
    exImage, exLabel = random.choice(list(zip(images,labels)))
    print('Label:', exLabel)
    print('Image:')
    draw(exImage)

    # preprocessing
    print('--- Preprocessing images ---')
    data = images.reshape((dsize, -1, 1)) # data in the first layer l0

    # the activation function
    activate = tanh
    activateLast = sigmoid

    # actual learning process - each epoch we forward `dsize` images, then backprop 1 time to update the NN
    print('\n--- Training ---')
    error = 1

    while True:
        for epoch in range(nepoch):
            print('Epoch #' + str(epoch) + '/' + str(nepoch))
            
            # forward propagation - the prediction
            a0s = data # feeding `dsize` images at the same time!
            a1s = np.array([activate(w1 @ a0 + b1) for a0 in a0s])
            a2s = np.array([activate(w2 @ a1 + b2) for a1 in a1s])
            a3s = np.array([activateLast(w3 @ a2 + b3) for a2 in a2s])

            # the errors
            oldError, error = error, (np.array([round(a3) for a3 in a3s.reshape(-1)]) ^ labels.reshape(-1)).sum() / dsize
            print('Error rate:', error)
            if error > oldError:
                step *= 0.5
                print('Step changed:', step)

            # back propagation function - to update weigths and biases to do the gradient descent
            # lấy tổng trước rồi mới normalize
            db3s = np.array([2 * (a3 - y) * activateLast(w3 @ a2 + b3, deriv=True) for a2,a3,y in zip(a2s,a3s,labels)])
            dw3s = np.array([db3 * a2.T for a2,db3 in zip(a2s,db3s)])
            da2s = np.array([w3.T @ db3 for db3 in db3s])

            db2s = np.array([da2 * activate(w2 @ a1 + b2, deriv=True) for a1,da2 in zip(a1s,da2s)])
            dw2s = np.array([db2 * a1.T for a1,db2 in zip(a1s,db2s)])
            da1s = np.array([w2.T @ db2 for db2 in db2s])

            db1s = np.array([da1 * activate(w1 @ a0 + b1, deriv=True) for a0,da1 in zip(a0s,da1s)])
            dw1s = np.array([db1 * a0.T for a0,db1 in zip(a0s,db1s)])

            # sum all the opinions of the dataset, then normalize the coordinates to take the given `step`
            dw1, db1 = dw1s.sum(axis=0), db1s.sum(axis=0)
            dw2, db2 = dw2s.sum(axis=0), db2s.sum(axis=0)
            dw3, db3 = dw3s.sum(axis=0), db3s.sum(axis=0)

            # the minus sign: minimize the cost function, since gradient maximizes it
            denom = sum([(dx**2).sum() for dx in [dw1,db1,dw2,db2,dw3,db3]]) ** 0.5 + 1e-300 # divide by 0
            dw1 *= -step / denom
            db1 *= -step / denom
            dw2 *= -step / denom
            db2 *= -step / denom
            dw3 *= -step / denom
            db3 *= -step / denom

            # gradient descent
            w1 += dw1
            b1 += db1
            w2 += dw2
            b2 += db2
            w3 += dw3
            b3 += db3

        print('\nTraining completed!')
        print('Time elapsed: ' + str(time.time() - startTime) + ' seconds.')
        #Beep(440, 10000)
        
        cont = input('Continue to learn? (y?) ')
        if cont != 'y':
            break

    # 1: triangle, 0: not triangle
    def isTriangle(image):
        # layer 0
        x = image.reshape(-1, 1)
        # layer 1
        x = activate(w1 @ x + b1)
        # layer 2
        x = activate(w2 @ x + b2)
        # layer 3
        x = activateLast(w3 @ x + b3)

        return round(x.item())

    print('\n--- Testing ---')
    while True:
        terror = 0
        print('Testing set size: ' + str(tsize))
        for test in range(tsize):
            shapeID = random.getrandbits(2)
            if shapeID >> 1:
                shape = genTriangle(size=isize)
            else:
                if shapeID & 1:
                    shape = genSquare(size=isize)
                else:
                    shape = genNoise(size=isize)
            error_ = (shapeID >> 1) ^ isTriangle(shape)
            terror += error_

            if error_:
                print('===== Test #' + str(test) + ' =====')
                print('Label:', shapeID >> 1)
                print('Predicted:', (shapeID >> 1) ^ 1)
                draw(shape)
                
        print('Error rate: ' + str(terror / tsize))

        cont = input('Continue to test? (y?) ')
        if cont != 'y':
            break

    print('--- See you later! ---')
        
if __name__ == '__main__':
    main()
