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
import stripline_plot
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

    plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', x1, stripline_plot.f2(x1, 6), 'go')

    plt.ylabel('Impedance')
    plt.xlabel('W/a')

    plt.title('stripline using BackPropagation')
    plt.show()

    errorplot = abs(np.asarray(out2).T - stripline_plot.f2(x1, 6))/ stripline_plot.f2(x1, 6)*100

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

    plt.xlabel('W/a')

    plt.title('%error using BackPropagation')
    plt.show()


    x = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for i in x:
        ou = myNN2.fun([i / 0.99])
        print(ou[0] * 86.3722)


def inBuilt(inp_dat,out_dat, inp_start, inp_end, inc, out_max):

    inp = []
    out = []
    for n in inp_dat:
        inp.append([int(n[0]*10)])
    for n in out_dat:
        out.append(int(n*100))



    classifiers = [
        ("LINEAR: ", linear_model.LinearRegression()),
        ('LOG-LBFGS: ', LogisticRegression(solver='lbfgs', max_iter=5000)),
        ('LOG-NEWTON: ', LogisticRegression(solver='newton-cg', max_iter=2000)),

        ('MLPCLAS-ADAM: ', MLPClassifier(solver='adam', max_iter=5000)),
        ('SGDREG: ', MLPRegressor(solver='lbfgs', max_iter=1000)),

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
            y = clf.predict(k[0])/100 + 2
           # for j in range(len(y)):
           #     y[j] *= out_max
            out2.append(list(y))

            #   print(time.time() - start)

        plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', np.asarray(inp2), stripline_plot.f2(np.asarray(inp2), 6), 'go')
        plt.ylabel('Impedance')
        plt.xlabel('w/a')

        plt.title('stripline using ' + name)
        plt.show()

        errorplot = abs(np.asarray(out2).T - stripline_plot.f2(np.asarray(inp2), 6))/stripline_plot.f2(np.asarray(inp2), 6)*100

        max_error = max(errorplot[0])
        print("max error: ", max_error)

        avg_error = 0
        for i in errorplot[0]:
            avg_error += i
        avg_error = avg_error / len(errorplot[0])

        print("average erro: ", avg_error)

        plt.plot(np.asarray(inp2), errorplot[0], 'r--')
        plt.ylabel('absolute error')

        plt.xlabel('w/a')

        plt.title('%error using ' + name)
        plt.show()

        x = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        for i in x:
            ou = clf.predict(i * 10)
            print(ou[0] / 100)

def main():

    stripline_plot.plot()
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
        86.372265712631929, 84.795287069694467, 83.314427219294629, 81.918719549137478, 80.598977609945209,
        79.347429499484292, 78.157441543248623, 77.023306314831089, 75.940077815676943, 74.903441758809691,
        73.909612350039382, 72.955249325560288, 72.037390655262982, 71.153397490807905, 70.300908778536794,
        69.477803570016761, 68.682169514873181, 67.912276357563925, 67.166553513726285, 66.443570995010958,
        65.742023099742383, 65.060714401707301, 64.398547659115835, 63.754513336369882, 63.127680487185316,
        62.517188792196386, 61.922241579942614, 61.342099689011711, 60.776076052544504, 60.223530905435105,
        59.683867530238928, 59.156528470731025, 58.640992152753867, 58.136769860894681, 57.643403026959696,
        57.160460792442912, 56.687537812427465, 56.224252272784547, 55.770244096288486, 55.32517331645527,
        54.888718600636636, 54.460575906229977, 54.040457255865348, 53.628089619151787, 53.223213890052023,
        52.825583950240024, 52.434965809912704, 52.051136818497795, 51.673884938546514, 51.303008076839433,
        50.938313467382649, 50.579617101539277, 50.226743201043455, 49.879523730083115, 49.537797943029666,
        49.201411964735605, 48.870218400627863, 48.544075974096089, 48.222849188915824, 47.906408014662553,
        47.594627593264477, 47.28738796501311, 46.984573812505602, 46.686074221129189, 46.391782454823669,
        46.101595745967579, 45.815415098335372, 45.533145102162372, 45.254693760436737, 44.979972325611094,
        44.708895145993296, 44.441379521136881, 44.177345565606316, 43.916716080542372, 43.65941643249807,
        43.405374439057894, 43.154520260789411, 42.906786299112127, 42.662107099699185, 42.420419261056232,
        42.181661347948655, 41.9457738093719, 41.712698900782073, 41.482380610324391, 41.254764588815348
    ]
    dat_backProp = [
        [[0.15 / 0.99], [86.3722657126 / 86.3722657126]],
        [[0.16 / 0.99], [84.7952870697 / 86.3722657126]],
        [[0.17 / 0.99], [83.3144272193 / 86.3722657126]],
        [[0.18 / 0.99], [81.9187195491 / 86.3722657126]],
        [[0.19 / 0.99], [80.5989776099 / 86.3722657126]],
        [[0.2 / 0.99], [79.3474294995 / 86.3722657126]],
        [[0.21 / 0.99], [78.1574415432 / 86.3722657126]],
        [[0.22 / 0.99], [77.0233063148 / 86.3722657126]],
        [[0.23 / 0.99], [75.9400778157 / 86.3722657126]],
        [[0.24 / 0.99], [74.9034417588 / 86.3722657126]],
        [[0.25 / 0.99], [73.90961235 / 86.3722657126]],
        [[0.26 / 0.99], [72.9552493256 / 86.3722657126]],
        [[0.27 / 0.99], [72.0373906553 / 86.3722657126]],
        [[0.28 / 0.99], [71.1533974908 / 86.3722657126]],
        [[0.29 / 0.99], [70.3009087785 / 86.3722657126]],
        [[0.3 / 0.99], [69.47780357 / 86.3722657126]],
        [[0.31 / 0.99], [68.6821695149 / 86.3722657126]],
        [[0.32 / 0.99], [67.9122763576 / 86.3722657126]],
        [[0.33 / 0.99], [67.1665535137 / 86.3722657126]],
        [[0.34 / 0.99], [66.443570995 / 86.3722657126]],
        [[0.35 / 0.99], [65.7420230997 / 86.3722657126]],
        [[0.36 / 0.99], [65.0607144017 / 86.3722657126]],
        [[0.37 / 0.99], [64.3985476591 / 86.3722657126]],
        [[0.38 / 0.99], [63.7545133364 / 86.3722657126]],
        [[0.39 / 0.99], [63.1276804872 / 86.3722657126]],
        [[0.4 / 0.99], [62.5171887922 / 86.3722657126]],
        [[0.41 / 0.99], [61.9222415799 / 86.3722657126]],
        [[0.42 / 0.99], [61.342099689 / 86.3722657126]],
        [[0.43 / 0.99], [60.7760760525 / 86.3722657126]],
        [[0.44 / 0.99], [60.2235309054 / 86.3722657126]],
        [[0.45 / 0.99], [59.6838675302 / 86.3722657126]],
        [[0.46 / 0.99], [59.1565284707 / 86.3722657126]],
        [[0.47 / 0.99], [58.6409921528 / 86.3722657126]],
        [[0.48 / 0.99], [58.1367698609 / 86.3722657126]],
        [[0.49 / 0.99], [57.643403027 / 86.3722657126]],
        [[0.5 / 0.99], [57.1604607924 / 86.3722657126]],
        [[0.51 / 0.99], [56.6875378124 / 86.3722657126]],
        [[0.52 / 0.99], [56.2242522728 / 86.3722657126]],
        [[0.53 / 0.99], [55.7702440963 / 86.3722657126]],
        [[0.54 / 0.99], [55.3251733165 / 86.3722657126]],
        [[0.55 / 0.99], [54.8887186006 / 86.3722657126]],
        [[0.56 / 0.99], [54.4605759062 / 86.3722657126]],
        [[0.57 / 0.99], [54.0404572559 / 86.3722657126]],
        [[0.58 / 0.99], [53.6280896192 / 86.3722657126]],
        [[0.59 / 0.99], [53.2232138901 / 86.3722657126]],
        [[0.6 / 0.99], [52.8255839502 / 86.3722657126]],
        [[0.61 / 0.99], [52.4349658099 / 86.3722657126]],
        [[0.62 / 0.99], [52.0511368185 / 86.3722657126]],
        [[0.63 / 0.99], [51.6738849385 / 86.3722657126]],
        [[0.64 / 0.99], [51.3030080768 / 86.3722657126]],
        [[0.65 / 0.99], [50.9383134674 / 86.3722657126]],
        [[0.66 / 0.99], [50.5796171015 / 86.3722657126]],
        [[0.67 / 0.99], [50.226743201 / 86.3722657126]],
        [[0.68 / 0.99], [49.8795237301 / 86.3722657126]],
        [[0.69 / 0.99], [49.537797943 / 86.3722657126]],
        [[0.7 / 0.99], [49.2014119647 / 86.3722657126]],
        [[0.71 / 0.99], [48.8702184006 / 86.3722657126]],
        [[0.72 / 0.99], [48.5440759741 / 86.3722657126]],
        [[0.73 / 0.99], [48.2228491889 / 86.3722657126]],
        [[0.74 / 0.99], [47.9064080147 / 86.3722657126]],
        [[0.75 / 0.99], [47.5946275933 / 86.3722657126]],
        [[0.76 / 0.99], [47.287387965 / 86.3722657126]],
        [[0.77 / 0.99], [46.9845738125 / 86.3722657126]],
        [[0.78 / 0.99], [46.6860742211 / 86.3722657126]],
        [[0.79 / 0.99], [46.3917824548 / 86.3722657126]],
        [[0.8 / 0.99], [46.101595746 / 86.3722657126]],
        [[0.81 / 0.99], [45.8154150983 / 86.3722657126]],
        [[0.82 / 0.99], [45.5331451022 / 86.3722657126]],
        [[0.83 / 0.99], [45.2546937604 / 86.3722657126]],
        [[0.84 / 0.99], [44.9799723256 / 86.3722657126]],
        [[0.85 / 0.99], [44.708895146 / 86.3722657126]],
        [[0.86 / 0.99], [44.4413795211 / 86.3722657126]],
        [[0.87 / 0.99], [44.1773455656 / 86.3722657126]],
        [[0.88 / 0.99], [43.9167160805 / 86.3722657126]],
        [[0.89 / 0.99], [43.6594164325 / 86.3722657126]],
        [[0.9 / 0.99], [43.4053744391 / 86.3722657126]],
        [[0.91 / 0.99], [43.1545202608 / 86.3722657126]],
        [[0.92 / 0.99], [42.9067862991 / 86.3722657126]],
        [[0.93 / 0.99], [42.6621070997 / 86.3722657126]],
        [[0.94 / 0.99], [42.4204192611 / 86.3722657126]],
        [[0.95 / 0.99], [42.1816613479 / 86.3722657126]],
        [[0.96 / 0.99], [41.9457738094 / 86.3722657126]],
        [[0.97 / 0.99], [41.7126989008 / 86.3722657126]],
        [[0.98 / 0.99], [41.4823806103 / 86.3722657126]],
        [[0.99 / 0.99], [41.2547645888 / 86.3722657126]]
    ]
    inp_start = 0.15
    inp_end = 0.99
    inc = 0.01
    out_max = 86.3722657126

    backProp(dat_backProp, inp_start, inp_end, inc, out_max)

    inBuilt(inp_dat,out_dat, inp_start, inp_end, inc, out_max)




if __name__ == "__main__":
    main()
