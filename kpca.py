from operator import le
import numpy as np
from numpy.core.fromnumeric import argsort, shape, trace
from scipy import linalg
import matplotlib.pyplot as plt

def main(name, sigma):
  with open(name) as f:
    content = [line.strip().split(',') for line in f] 
  content = np.array(content)
  np.random.seed(1)
  np.random.shuffle(content)
  content = content.astype(np.float)
  n = len(content)
  d = 2
  numTrain = round(.7 * n)
  yTrain = content[:numTrain,2]
  yTest = content[numTrain:,2]
  X = content[:,0:2]
  mean = np.mean(X, 0)
  std = np.std(X, 0)
  for i in range(len(X)):
    X[i] = (X[i] - mean) / std
  xTest = X[numTrain:]
  xTrain = X[:numTrain]

  sizeY = len(yTrain)
  l = np.zeros((sizeY, sizeY))
  for i in range(sizeY):
    for j in range(sizeY):
      if yTrain[i] == yTrain[j]:
        l[i][j] = 1
  
  size1 = len(xTrain)
  k = np.zeros((size1, size1))
  for i in range(size1):
    for j in range(size1):
      k[i][j] = np.exp((-1 * (np.linalg.norm(xTrain[i] - xTrain[j]))**2) / (2 * sigma**2))

  size1 = len(xTrain)
  size2 = len(xTest)
  kTest = np.zeros((size1, size2))
  for i in range(size1):
    for j in range(size2):
      kTest[i][j] = np.exp((-1 * np.linalg.norm(xTrain[i] - xTest[j])**2) / (2 * sigma**2))

  I = np.identity(numTrain, np.float)
  e = np.ones((numTrain,)).T
  H = I - ((numTrain**-1)*(e @ e.T))

  Q = k @ H @ l @ H @ k
  w, v = linalg.eigh(Q, k)
  beta = v[:,len(w) - d:]
  Z = (beta.T @ k).T
  z = (beta.T @ kTest).T

  plot(name, xTrain, yTrain, xTest, yTest, Z, z)

def plot(name, xTrain, yTrain, xTest, yTest, Z, z):
  if name == 'Concentric_rectangles.txt':
    class0Train = []
    class1Train = []
    class2Train = []
    class0Z = []
    class1Z = []
    class2Z = []
    for i in range(len(xTrain)):
      if yTrain[i] == 1:
        class0Train.append(xTrain[i])
        class0Z.append(Z[i])
      elif yTrain[i] == 2:
        class1Train.append(xTrain[i])
        class1Z.append(Z[i])
      else:
        class2Train.append(xTrain[i])
        class2Z.append(Z[i])

    class0Test = []
    class1Test = []
    class2Test = []
    class0z = []
    class1z = []
    class2z = []
    for i in range(len(xTest)):
      if yTest[i] == 1:
        class0Test.append(xTest[i])
        class0z.append(z[i])
      elif yTest[i] == 2:
        class1Test.append(xTest[i])
        class1z.append(z[i])
      else:
        class2Test.append(xTest[i])
        class2z.append(z[i])
    plt.scatter(*zip(*class0Train), color = "#ff4d4d", label="class1, train")   # y = 0, train
    plt.scatter(*zip(*class0Test), color = "#ff8080", label="class1, test", marker="^")  # y = 0, test
    plt.scatter(*zip(*class1Train), color = "#00ace6", label="class2, train")  # y = 1, train
    plt.scatter(*zip(*class1Test), color = "#80dfff", label="class2, test", marker="^") # y = 1, test  
    plt.scatter(*zip(*class2Train), color = "#009900", label="class3, train")  # y = 2, train
    plt.scatter(*zip(*class2Test), color = "#4dff4d", label="class3, test", marker="^") # y = 2, test  
    plt.legend(loc='upper left') 
    plt.title(name + " original space")
    plt.show()

    plt.scatter(*zip(*class0Z), color = "#ff4d4d", label="class1, train")   # y = 0, train
    plt.scatter(*zip(*class0z), color = "#ff8080", label="class1, test", marker="^")  # y = 0, test
    plt.scatter(*zip(*class1Z), color = "#00ace6", label="class2, train")  # y = 1, train
    plt.scatter(*zip(*class1z), color = "#80dfff", label="class2, test", marker="^") # y = 1, test  
    plt.scatter(*zip(*class2Z), color = "#009900", label="class3, train")  # y = 2, train
    plt.scatter(*zip(*class2z), color = "#4dff4d", label="class3, test", marker="^") # y = 2, test   
    plt.legend(loc='upper left') 
    plt.title(name + " after projection space")
    plt.show()

  else:
    class0Train = []
    class1Train = []
    class0Z = []
    class1Z = []
    for i in range(len(xTrain)):
      if yTrain[i] == 1:
        class0Train.append(xTrain[i])
        class0Z.append(Z[i])
      else:
        class1Train.append(xTrain[i])
        class1Z.append(Z[i])

    class0Test = []
    class1Test = []
    class0z = []
    class1z = []
    for i in range(len(xTest)):
      if yTest[i] == 1:
        class0Test.append(xTest[i])
        class0z.append(z[i])
      else:
        class1Test.append(xTest[i])
        class1z.append(z[i])

    plt.scatter(*zip(*class0Train), color = "#ff4d4d", label="class1, train")   # y = 0, train
    plt.scatter(*zip(*class0Test), color = "#ff8080", label="class1, test", marker="^")  # y = 0, test
    plt.scatter(*zip(*class1Train), color = "#00ace6", label="class2, train")  # y = 1, train
    plt.scatter(*zip(*class1Test), color = "#80dfff", label="class2, test", marker="^") # y = 1, test  
    plt.legend(loc='upper left') 
    plt.title(name+ " original space")
    plt.show()

    plt.scatter(*zip(*class0Z), color = "#ff4d4d", label="class1, train")   # y = 0, train
    plt.scatter(*zip(*class0z), color = "#ff8080", label="class1, test", marker="^")  # y = 0, test
    plt.scatter(*zip(*class1Z), color = "#00ace6", label="class2, train")  # y = 1, train
    plt.scatter(*zip(*class1z), color = "#80dfff", label="class2, test", marker="^") # y = 1, test   
    plt.legend(loc='upper left') 
    plt.title(name+ " after projection space")
    plt.show()

main("Twomoons.txt", sigma=.1)
main("Binary_XOR.txt", sigma=.1)
main("Concentric_rings.txt", sigma=.1)
main("Concentric_rectangles.txt", sigma=.1)