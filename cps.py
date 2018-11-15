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
import cps_plot
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

    plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', x1, cps_plot.f2(x1, 6), 'go')

    plt.ylabel('Impedance')
    plt.xlabel('a/b')

    plt.title('cps using BackPropagation')
    plt.show()

    errorplot = abs(np.asarray(out2).T - cps_plot.f2(x1, 6))/ cps_plot.f2(x1, 6)*100

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
    plt.xlabel('a/b')
    plt.title('%error using BackPropagation')
    plt.show()


    x = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]
    for i in x:
        ou = myNN2.fun([i / 0.85])
        print(ou[0] * 271.445863004)


def inBuilt(inp_dat,out_dat, inp_start, inp_end, inc, out_max):

    inp = []
    out = []
    for n in inp_dat:
        inp.append([int(n[0]*10)])
    for n in out_dat:
        out.append(int(n*100))



    classifiers = [
        ("LINEAR: ", linear_model.LinearRegression()),
        ('LOG-LBFGS: ', LogisticRegression(solver='lbfgs', max_iter=1000)),
        ('LOG-NEWTON: ', LogisticRegression(solver='newton-cg', max_iter=1000)),

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
            y = clf.predict(k[0])/100  - 10
           # for j in range(len(y)):
           #     y[j] *= out_max
            out2.append(list(y))

            #   print(time.time() - start)

        plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', np.asarray(inp2), cps_plot.f2(np.asarray(inp2), 6), 'go')
        plt.ylabel('Impedance')
        plt.xlabel('a/b')
        plt.title('cps using ' + name)
        plt.show()
        errorplot = abs(np.asarray(out2).T - cps_plot.f2(np.asarray(inp2), 6))/cps_plot.f2(np.asarray(inp2), 6)*100

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

        x = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]
        for i in x:
            ou = clf.predict(i * 10)
            print(ou[0] / 100)

def main():
    cps_plot.plot()
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
        [0.89000000000000068]
    ]
    out_dat = [
        96.612854885044641, 98.578433042084612, 100.5022992522847, 102.3895988941463, 104.24476143250942,
        106.07163956831863, 107.87361626369143, 109.65368826006092, 111.41453211411394, 113.15855704193471,
        114.88794767804886, 116.6046990332985, 118.31064535401556, 120.00748416782838, 121.69679649802674,
        123.38006400494201, 125.05868364630393, 126.73398032316925, 128.40721788272589, 130.07960877619516,
        131.75232261359756, 133.42649381320814, 135.10322850916998, 136.78361085372012, 138.46870882921289,
        140.15957966834796, 141.85727496782138, 143.56284557030384, 145.27734628169836, 147.00184048462685,
        148.7374047047542, 150.48513318365545, 152.24614251031184, 154.0215763628897, 155.81261041314096,
        157.62045744757958, 159.44637276252811, 161.29165989427898, 163.1576767510837, 165.04584222059876,
        166.95764333501711, 168.89464308663227, 170.85848899937804, 172.85092257737679, 174.8737897702641,
        176.92905261770446, 179.01880226295054, 181.14527355859778, 183.3108615282444, 185.51813999734119,
        187.76988276737885, 190.06908778263693, 192.41900483180947, 194.82316744290719, 197.28542977545774,
        199.81000949790592, 202.40243794760764, 205.06605291683425, 207.80739528898155, 210.63273879610452,
        213.54908140919014, 216.56426434536635, 219.68711637426836, 222.92763018005692, 226.29717973785134,
        229.80879072635059, 233.47748032016133, 237.320688899385, 241.35883524537024, 245.6160402116019,
        250.12108422278322, 254.90869558665128, 260.02131704001704, 265.51158078750512, 271.4458630037459
    ]
    dat_backProp = [
        [[0.15 / 0.89], [96.612854885 / 271.445863004]],
        [[0.16 / 0.89], [98.5784330421 / 271.445863004]],
        [[0.17 / 0.89], [100.502299252 / 271.445863004]],
        [[0.18 / 0.89], [102.389598894 / 271.445863004]],
        [[0.19 / 0.89], [104.244761433 / 271.445863004]],
        [[0.2 / 0.89], [106.071639568 / 271.445863004]],
        [[0.21 / 0.89], [107.873616264 / 271.445863004]],
        [[0.22 / 0.89], [109.65368826 / 271.445863004]],
        [[0.23 / 0.89], [111.414532114 / 271.445863004]],
        [[0.24 / 0.89], [113.158557042 / 271.445863004]],
        [[0.25 / 0.89], [114.887947678 / 271.445863004]],
        [[0.26 / 0.89], [116.604699033 / 271.445863004]],
        [[0.27 / 0.89], [118.310645354 / 271.445863004]],
        [[0.28 / 0.89], [120.007484168 / 271.445863004]],
        [[0.29 / 0.89], [121.696796498 / 271.445863004]],
        [[0.3 / 0.89], [123.380064005 / 271.445863004]],
        [[0.31 / 0.89], [125.058683646 / 271.445863004]],
        [[0.32 / 0.89], [126.733980323 / 271.445863004]],
        [[0.33 / 0.89], [128.407217883 / 271.445863004]],
        [[0.34 / 0.89], [130.079608776 / 271.445863004]],
        [[0.35 / 0.89], [131.752322614 / 271.445863004]],
        [[0.36 / 0.89], [133.426493813 / 271.445863004]],
        [[0.37 / 0.89], [135.103228509 / 271.445863004]],
        [[0.38 / 0.89], [136.783610854 / 271.445863004]],
        [[0.39 / 0.89], [138.468708829 / 271.445863004]],
        [[0.4 / 0.89], [140.159579668 / 271.445863004]],
        [[0.41 / 0.89], [141.857274968 / 271.445863004]],
        [[0.42 / 0.89], [143.56284557 / 271.445863004]],
        [[0.43 / 0.89], [145.277346282 / 271.445863004]],
        [[0.44 / 0.89], [147.001840485 / 271.445863004]],
        [[0.45 / 0.89], [148.737404705 / 271.445863004]],
        [[0.46 / 0.89], [150.485133184 / 271.445863004]],
        [[0.47 / 0.89], [152.24614251 / 271.445863004]],
        [[0.48 / 0.89], [154.021576363 / 271.445863004]],
        [[0.49 / 0.89], [155.812610413 / 271.445863004]],
        [[0.5 / 0.89], [157.620457448 / 271.445863004]],
        [[0.51 / 0.89], [159.446372763 / 271.445863004]],
        [[0.52 / 0.89], [161.291659894 / 271.445863004]],
        [[0.53 / 0.89], [163.157676751 / 271.445863004]],
        [[0.54 / 0.89], [165.045842221 / 271.445863004]],
        [[0.55 / 0.89], [166.957643335 / 271.445863004]],
        [[0.56 / 0.89], [168.894643087 / 271.445863004]],
        [[0.57 / 0.89], [170.858488999 / 271.445863004]],
        [[0.58 / 0.89], [172.850922577 / 271.445863004]],
        [[0.59 / 0.89], [174.87378977 / 271.445863004]],
        [[0.6 / 0.89], [176.929052618 / 271.445863004]],
        [[0.61 / 0.89], [179.018802263 / 271.445863004]],
        [[0.62 / 0.89], [181.145273559 / 271.445863004]],
        [[0.63 / 0.89], [183.310861528 / 271.445863004]],
        [[0.64 / 0.89], [185.518139997 / 271.445863004]],
        [[0.65 / 0.89], [187.769882767 / 271.445863004]],
        [[0.66 / 0.89], [190.069087783 / 271.445863004]],
        [[0.67 / 0.89], [192.419004832 / 271.445863004]],
        [[0.68 / 0.89], [194.823167443 / 271.445863004]],
        [[0.69 / 0.89], [197.285429775 / 271.445863004]],
        [[0.7 / 0.89], [199.810009498 / 271.445863004]],
        [[0.71 / 0.89], [202.402437948 / 271.445863004]],
        [[0.72 / 0.89], [205.066052917 / 271.445863004]],
        [[0.73 / 0.89], [207.807395289 / 271.445863004]],
        [[0.74 / 0.89], [210.632738796 / 271.445863004]],
        [[0.75 / 0.89], [213.549081409 / 271.445863004]],
        [[0.76 / 0.89], [216.564264345 / 271.445863004]],
        [[0.77 / 0.89], [219.687116374 / 271.445863004]],
        [[0.78 / 0.89], [222.92763018 / 271.445863004]],
        [[0.79 / 0.89], [226.297179738 / 271.445863004]],
        [[0.8 / 0.89], [229.808790726 / 271.445863004]],
        [[0.81 / 0.89], [233.47748032 / 271.445863004]],
        [[0.82 / 0.89], [237.320688899 / 271.445863004]],
        [[0.83 / 0.89], [241.358835245 / 271.445863004]],
        [[0.84 / 0.89], [245.616040212 / 271.445863004]],
        [[0.85 / 0.89], [250.121084223 / 271.445863004]],
        [[0.86 / 0.89], [254.908695587 / 271.445863004]],
        [[0.87 / 0.89], [260.02131704 / 271.445863004]],
        [[0.88 / 0.89], [265.511580788 / 271.445863004]],
        [[0.89 / 0.89], [271.445863004 / 271.445863004]]
    ]  ##100
    inp_start = 0.15
    inp_end = 0.89
    inc = 0.01
    out_max = 271.445863004

    backProp(dat_backProp, inp_start, inp_end, inc, out_max)

    inBuilt(inp_dat,out_dat, inp_start, inp_end, inc, out_max)




if __name__ == "__main__":
    main()
