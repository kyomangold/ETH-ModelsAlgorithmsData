import matplotlib.pyplot as plt
import numpy as np

def plotSinusTest(x, y, y_hat, path):
    #plot data
    plt.figure(0)
    plt.xlabel('x')
    plt.plot(x,y_hat,'go', label='sin(x)')
    plt.plot(x,y,'bo', label='NN')
    plt.legend(loc=1)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot(x, y, path):
    #plot data
    plt.figure(0)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.plot(x,y,'bo')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plotLosses(train, val, path):
    #plot data
    plt.figure(0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(len(train)),train,'b--', label='train')
    plt.plot(np.arange(len(val)),val,'g--', label='validation')
    plt.legend(loc=1)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def getBatch(data, i, batch_size):
    return data[i*batch_size:(i+1)*batch_size, :]

def processLabels(labels):
    # function to transform the mnist classes to two classes, even [1,0] and odd [0,1] numbers.
    targets = []
    for label in labels:
        idx = np.where(label == 1)[0][0]
        if idx % 2 == 0:
            # Even number detected
            targets.append([1, 0])
        else:
            # Odd number detected
            targets.append([0, 1])
    return np.array(targets) 


def displayImageEvenOdd(vector, even_odd):
    # reshaping the data vector to an image
    n = int(np.sqrt(vector.shape[0]))
    image = np.reshape(vector, (n, n))
    plt.imshow(image, cmap='gray')
    idx = np.where(even_odd>0.5)[0][0]
    if idx==0:
        titl = 'Even number'
    else:
        titl = 'Odd number'
    plt.title(titl)
    plt.show()
    return 0

def displayImageMNIST(vector, digit):
    # reshaping the data vector to an image
    n = int(np.sqrt(vector.shape[0]))
    image = np.reshape(vector, (n, n))
    plt.imshow(image, cmap='gray')
    idx = np.where(digit>0.9)[0][0]
    plt.title('Number {:d}'.format(idx))
    plt.show()
    return 0
