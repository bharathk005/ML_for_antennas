# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:53:28 2017

@author: bharathk
"""

import math
import random
import matplotlib.pyplot as plt
import time

import numpy as np
from sklearn.svm import LinearSVC
import cpw_plot
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, Perceptron


class NN:
    def __init__(self, NI, NH, NO):
        # number of nodes in layers
        self.ni = NI + 1  # +1 for bias
        self.nh = NH
        self.no = NO

        # initialize node-activations
        self.ai, self.ah, self.ao = [], [], []
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create node weight matrices
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # initialize node weights to random vals
        randomizeMatrix(self.wi, -0.2, 0.2)
        randomizeMatrix(self.wo, -2.0, 2.0)
        # create last change in weights matrices for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def runNN(self, inputs):
        if len(inputs) != self.ni - 1:
            print('incorrect number of inputs')

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += (self.ai[i] * self.wi[i][j])
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += (self.ah[j] * self.wo[j][k])
            self.ao[k] = sigmoid(sum)

        return self.ao

    def backPropagate(self, targets, N, M):

        # calc output deltas
        # we want to find the instantaneous rate of change of ( error with respect to weight from node j to node k)
        # output_delta is defined as an attribute of each ouput node. It is not the final rate we need.
        # To get the final rate we must multiply the delta by the activation of the hidden layer node in question.
        # This multiplication is done according to the chain rule as we are taking the derivative of the activation function
        # of the ouput node.
        # dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * dsigmoid(self.ao[k])

            # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                # output_deltas[k] * self.ah[j] is the full derivative of dError/dweight[j][k]
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + M * self.co[j][k]
                self.co[j][k] = change

        # calc hidden deltas
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self.ah[j])

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                # print 'activation',self.ai[i],'synapse',i,j,'change',change
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # calc combined error
        # 1/2 for differential convenience & **2 for modulus
        error = 0.0
        for k in range(len(targets)):
            error = 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])
        print('')

    def test(self, patterns):
        for p in patterns:
            inputs = p[0]
            self.runNN(inputs)
            # print ('Inputs:', p[0], '-->', self.runNN(inputs), ' Target', p[1])

    def train(self, patterns, max_iterations=1000, N=0.5, M=0.1):
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.runNN(inputs)
                error = self.backPropagate(targets, N, M)
                # if i % 50 == 0:
                #   print ('Combined error', error)
        self.test(patterns)

    def fun(self, x):
        return self.runNN(x)


def sigmoid(x):
    return math.tanh(x)


# the derivative of the sigmoid function in terms of output
# proof here:
# http://www.math10.com/en/algebra/hyperbolic-functions/hyperbolic-functions.html
def dsigmoid(y):
    return 1 - y ** 2


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


def randomizeMatrix(matrix, a, b):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)






def backProp(dat_backProp, inp_start, inp_end, inc, out_max):


 #   start = time.time()

    myNN2 = NN(1, 5, 1)
    myNN2.train(dat_backProp)
    print(' ')

    inp2 = []
    out2 = []

    x1 = np.arange(inp_start, inp_end, inc)
    x = x1.reshape(-1, 1).tolist()

    for i in x:
        inp2.append(i)
        k =[]
        k.append( i[0]/inp_end )
        y = myNN2.fun(k)
        for j in range(len(y)):
            y[j] *= out_max
        out2.append(list(y))

 #   print(time.time() - start)

    plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', x1, cpw_plot.f2(x1,6), 'go')
    plt.ylabel('Impedance')
    plt.xlabel('a/b')
    plt.title('cpw using BackPropagation      b/h=0.1')
    plt.show()

    errorplot = abs(np.asarray(out2).T - cpw_plot.f2(x1,6)) / cpw_plot.f2(x1,6) * 100

    max_error = max(errorplot[0])
    print('BACK_PROP: ')
    print("max error: ",max_error)

    avg_error = 0
    for i in errorplot[0]:
        avg_error += i
    avg_error = avg_error/len(errorplot[0])

    print("average erro: ",avg_error)

    plt.plot(x1, errorplot[0], 'r--')
    plt.ylabel('absolute error')
    plt.xlabel('a/b ')
    plt.title('%error using BackPropagation')
    plt.show()


    x = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9,0.95]
    for i in x:
        ou = myNN2.fun([i / 0.99])
        print(ou[0] * 105.1679)


def inBuilt(inp_dat,out_dat, inp_start, inp_end, inc, out_max):

    inp = []
    out = []
    for n in inp_dat:
        inp.append([int(n[0]*10)])
    for n in out_dat:
        out.append(int(n*100))



    classifiers = [
        ("LINEAR: ", linear_model.LinearRegression()),
        ('LOG-LBFGS: ', LogisticRegression(solver='lbfgs', max_iter=2000)),
        ('LOG-NEWTON: ', LogisticRegression(solver='newton-cg', max_iter=2000)),

        ('MLPCLAS-ADAM: ', MLPClassifier(solver='adam', max_iter=5000)),
        ('SGDREG: ', MLPRegressor(solver='lbfgs', max_iter=2000)),

    ]
    clas = [

        ('SVC', LinearSVC(max_iter=2000)),

    ]

    for name, clf in classifiers:
        print(' ')
        clf.fit(inp,out)
        print(name,': ')
        inp2 = []
        out2 = []

        x1 = np.arange(inp_start*10, inp_end*10, inc*10)
        x = x1.reshape(-1, 1).tolist()

        for i in x:
            inp2.append(i[0]/10)
            k = []
            k.append(i[0])
            y = clf.predict(k[0])/100 + 4
           # for j in range(len(y)):
           #     y[j] *= out_max
            out2.append(list(y))

            #   print(time.time() - start)

        plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', np.asarray(inp2), cpw_plot.f2(np.asarray(inp2),6), 'go')
        plt.ylabel('Impedance')
        plt.xlabel('a/b ')
        plt.title('cpw using ' + name + '   b/h=0.1')
        plt.show()

        errorplot = abs(np.asarray(out2).T - cpw_plot.f2(np.asarray(inp2),6)) / cpw_plot.f2(np.asarray(inp2),6 ) * 100


        max_error = max(errorplot[0])
        print("max error: ", max_error)

        avg_error = 0
        for i in errorplot[0]:
            avg_error += i
        avg_error = avg_error / len(errorplot[0])

        print("average erro: ", avg_error)

        plt.plot(np.asarray(inp2), errorplot[0], 'r--')
        plt.ylabel('absolute error')

        plt.xlabel('a/b')

        plt.title('%error using ' + name)
        plt.show()

        x = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9,0.95]
        for i in x:
            ou = clf.predict(i * 10)
            print(ou[0] / 100)

def main():

    cpw_plot.plot()
    inp_dat = [
        [0.14999999999999999], [0.16], [0.17000000000000001], [0.18000000000000002], [0.19000000000000003],
        [0.20000000000000004], [0.21000000000000005], [0.22000000000000006], [0.23000000000000007],
        [0.24000000000000007], [0.25000000000000011], [0.26000000000000012], [0.27000000000000013],
        [0.28000000000000014], [0.29000000000000015], [0.30000000000000016], [0.31000000000000016],
        [0.32000000000000017], [0.33000000000000018], [0.34000000000000019], [0.3500000000000002],
        [0.36000000000000021], [0.37000000000000022], [0.38000000000000023], [0.39000000000000024],
        [0.40000000000000024], [0.41000000000000025], [0.42000000000000026], [0.43000000000000027],
        [0.44000000000000028], [0.45000000000000029], [0.4600000000000003], [0.47000000000000031],
        [0.48000000000000032], [0.49000000000000032], [0.50000000000000033], [0.51000000000000034],
        [0.52000000000000035], [0.53000000000000036], [0.54000000000000037], [0.55000000000000038],
        [0.56000000000000039], [0.5700000000000004], [0.5800000000000004], [0.59000000000000041], [0.60000000000000042],
        [0.61000000000000043], [0.62000000000000044], [0.63000000000000045], [0.64000000000000046],
        [0.65000000000000047], [0.66000000000000048], [0.67000000000000048], [0.68000000000000049],
        [0.6900000000000005], [0.70000000000000051], [0.71000000000000052], [0.72000000000000053],
        [0.73000000000000054], [0.74000000000000055], [0.75000000000000056], [0.76000000000000056],
        [0.77000000000000057], [0.78000000000000058], [0.79000000000000059], [0.8000000000000006],
        [0.81000000000000061], [0.82000000000000062], [0.83000000000000063], [0.84000000000000064],
        [0.85000000000000064], [0.86000000000000065], [0.87000000000000066], [0.88000000000000067],
        [0.89000000000000068], [0.90000000000000069], [0.9100000000000007], [0.92000000000000071],
        [0.93000000000000071], [0.94000000000000072], [0.95000000000000073], [0.96000000000000074],
        [0.97000000000000075], [0.98000000000000076], [0.99000000000000077]
    ]
    out_dat = [
        105.16795768136328, 103.07269419212219, 101.10124226680774, 99.239229825988716, 97.474614602722028,
        95.797205409014808, 94.198300316624554, 92.670409075701343, 91.207037276818227, 89.80251647197494,
        88.451868985839809, 87.150699245514346, 85.89510561798231, 84.681608275911017, 83.50708971361172,
        82.368745337178922, 81.264042144417033, 80.190683951364974, 79.146581954727168, 78.129829672540211,
        77.138681499696801, 76.171534265411211, 75.226911297156263, 74.303448587956368, 73.399882737054099,
        72.515040392254221, 71.647828968990325, 70.79722845885405, 69.962284170888736, 69.142100273848328,
        68.335834028000193, 67.542690611793077, 66.761918462515553, 65.992805061484262, 65.234673103757274,
        64.486877000209219, 63.748799666306404, 63.019849557297128, 62.299457913957461, 61.587076186644182,
        60.882173608305926, 60.18423488937669, 59.492758009177209, 58.807252079621961, 58.127235257691737,
        57.452232683289346, 56.781774418732553, 56.115393365220413, 55.452623130079097, 54.792995816367309,
        54.136039703382977, 53.481276782602762, 52.828220108395932, 52.176370916200881, 51.525215452367163,
        50.874221449052179, 50.222610968508846, 49.570246830685655, 48.916287294607123, 48.260080147086654,
        47.600927961831417, 46.938079888686993, 46.270721891965096, 45.597965037079405, 44.918831300463893,
        44.23223620674429, 43.536967358521395, 42.831657586119071, 42.114750957838751, 41.384459177192696,
        40.638704824699445, 39.87504626522886, 39.090576471698512, 38.281783862302397, 37.444356313227573,
        36.572897486363424, 35.660502840040557, 34.698101181547813, 33.673383438309628, 32.568956195957149,
        31.358914647401722, 30.001823618758245, 28.424201445675898, 26.472374699431207, 23.704187401746509
    ]
    dat_backProp = [
        [[0.15 / 0.99], [105.167957681 / 105.167957681]],
        [[0.16 / 0.99], [103.072694192 / 105.167957681]],
        [[0.17 / 0.99], [101.101242267 / 105.167957681]],
        [[0.18 / 0.99], [99.239229826 / 105.167957681]],
        [[0.19 / 0.99], [97.4746146027 / 105.167957681]],
        [[0.2 / 0.99], [95.797205409 / 105.167957681]],
        [[0.21 / 0.99], [94.1983003166 / 105.167957681]],
        [[0.22 / 0.99], [92.6704090757 / 105.167957681]],
        [[0.23 / 0.99], [91.2070372768 / 105.167957681]],
        [[0.24 / 0.99], [89.802516472 / 105.167957681]],
        [[0.25 / 0.99], [88.4518689858 / 105.167957681]],
        [[0.26 / 0.99], [87.1506992455 / 105.167957681]],
        [[0.27 / 0.99], [85.895105618 / 105.167957681]],
        [[0.28 / 0.99], [84.6816082759 / 105.167957681]],
        [[0.29 / 0.99], [83.5070897136 / 105.167957681]],
        [[0.3 / 0.99], [82.3687453372 / 105.167957681]],
        [[0.31 / 0.99], [81.2640421444 / 105.167957681]],
        [[0.32 / 0.99], [80.1906839514 / 105.167957681]],
        [[0.33 / 0.99], [79.1465819547 / 105.167957681]],
        [[0.34 / 0.99], [78.1298296725 / 105.167957681]],
        [[0.35 / 0.99], [77.1386814997 / 105.167957681]],
        [[0.36 / 0.99], [76.1715342654 / 105.167957681]],
        [[0.37 / 0.99], [75.2269112972 / 105.167957681]],
        [[0.38 / 0.99], [74.303448588 / 105.167957681]],
        [[0.39 / 0.99], [73.3998827371 / 105.167957681]],
        [[0.4 / 0.99], [72.5150403923 / 105.167957681]],
        [[0.41 / 0.99], [71.647828969 / 105.167957681]],
        [[0.42 / 0.99], [70.7972284589 / 105.167957681]],
        [[0.43 / 0.99], [69.9622841709 / 105.167957681]],
        [[0.44 / 0.99], [69.1421002738 / 105.167957681]],
        [[0.45 / 0.99], [68.335834028 / 105.167957681]],
        [[0.46 / 0.99], [67.5426906118 / 105.167957681]],
        [[0.47 / 0.99], [66.7619184625 / 105.167957681]],
        [[0.48 / 0.99], [65.9928050615 / 105.167957681]],
        [[0.49 / 0.99], [65.2346731038 / 105.167957681]],
        [[0.5 / 0.99], [64.4868770002 / 105.167957681]],
        [[0.51 / 0.99], [63.7487996663 / 105.167957681]],
        [[0.52 / 0.99], [63.0198495573 / 105.167957681]],
        [[0.53 / 0.99], [62.299457914 / 105.167957681]],
        [[0.54 / 0.99], [61.5870761866 / 105.167957681]],
        [[0.55 / 0.99], [60.8821736083 / 105.167957681]],
        [[0.56 / 0.99], [60.1842348894 / 105.167957681]],
        [[0.57 / 0.99], [59.4927580092 / 105.167957681]],
        [[0.58 / 0.99], [58.8072520796 / 105.167957681]],
        [[0.59 / 0.99], [58.1272352577 / 105.167957681]],
        [[0.6 / 0.99], [57.4522326833 / 105.167957681]],
        [[0.61 / 0.99], [56.7817744187 / 105.167957681]],
        [[0.62 / 0.99], [56.1153933652 / 105.167957681]],
        [[0.63 / 0.99], [55.4526231301 / 105.167957681]],
        [[0.64 / 0.99], [54.7929958164 / 105.167957681]],
        [[0.65 / 0.99], [54.1360397034 / 105.167957681]],
        [[0.66 / 0.99], [53.4812767826 / 105.167957681]],
        [[0.67 / 0.99], [52.8282201084 / 105.167957681]],
        [[0.68 / 0.99], [52.1763709162 / 105.167957681]],
        [[0.69 / 0.99], [51.5252154524 / 105.167957681]],
        [[0.7 / 0.99], [50.8742214491 / 105.167957681]],
        [[0.71 / 0.99], [50.2226109685 / 105.167957681]],
        [[0.72 / 0.99], [49.5702468307 / 105.167957681]],
        [[0.73 / 0.99], [48.9162872946 / 105.167957681]],
        [[0.74 / 0.99], [48.2600801471 / 105.167957681]],
        [[0.75 / 0.99], [47.6009279618 / 105.167957681]],
        [[0.76 / 0.99], [46.9380798887 / 105.167957681]],
        [[0.77 / 0.99], [46.270721892 / 105.167957681]],
        [[0.78 / 0.99], [45.5979650371 / 105.167957681]],
        [[0.79 / 0.99], [44.9188313005 / 105.167957681]],
        [[0.8 / 0.99], [44.2322362067 / 105.167957681]],
        [[0.81 / 0.99], [43.5369673585 / 105.167957681]],
        [[0.82 / 0.99], [42.8316575861 / 105.167957681]],
        [[0.83 / 0.99], [42.1147509578 / 105.167957681]],
        [[0.84 / 0.99], [41.3844591772 / 105.167957681]],
        [[0.85 / 0.99], [40.6387048247 / 105.167957681]],
        [[0.86 / 0.99], [39.8750462652 / 105.167957681]],
        [[0.87 / 0.99], [39.0905764717 / 105.167957681]],
        [[0.88 / 0.99], [38.2817838623 / 105.167957681]],
        [[0.89 / 0.99], [37.4443563132 / 105.167957681]],
        [[0.9 / 0.99], [36.5728974864 / 105.167957681]],
        [[0.91 / 0.99], [35.66050284 / 105.167957681]],
        [[0.92 / 0.99], [34.6981011815 / 105.167957681]],
        [[0.93 / 0.99], [33.6733834383 / 105.167957681]],
        [[0.94 / 0.99], [32.568956196 / 105.167957681]],
        [[0.95 / 0.99], [31.3589146474 / 105.167957681]],
        [[0.96 / 0.99], [30.0018236188 / 105.167957681]],
        [[0.97 / 0.99], [28.4242014457 / 105.167957681]],
        [[0.98 / 0.99], [26.4723746994 / 105.167957681]],
        [[0.99 / 0.99], [23.7041874017 / 105.167957681]]
    ]  ##100
    inp_start = 0.15
    inp_end = 0.99
    inc = 0.01
    out_max = 105.167957681

    backProp(dat_backProp, inp_start, inp_end, inc, out_max)

    inBuilt(inp_dat,out_dat, inp_start, inp_end, inc, out_max)




if __name__ == "__main__":
    main()
