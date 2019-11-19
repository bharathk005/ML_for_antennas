# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:53:28 2017

@author: bharathk
"""

import math
import random
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
import ant_plot
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

    plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', x1, ant_plot.f2(x1, 6), 'go')

    plt.ylabel('fr in MHz')
    plt.xlabel('w/h')

    plt.title('microstrip patch antenna using BackPropagation')
    plt.show()

    errorplot = abs(np.asarray(out2).T - ant_plot.f2(x1, 6))/ ant_plot.f2(x1, 6)*100

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

    plt.xlabel('w/h')

    plt.title('%error using BackPropagation')
    plt.show()


    x = [1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8, 8.5, 9., 9.5]
    for i in x:
        ou = myNN2.fun([i/9.5])
        print(ou[0]*7710.5576782)


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
            y = clf.predict(k[0])/100
           # for j in range(len(y)):
           #     y[j] *= out_max
            out2.append(list(y))

            #   print(time.time() - start)

        plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', np.asarray(inp2), ant_plot.f2(np.asarray(inp2), 6), 'go')
        plt.ylabel('fr in MHz')
        plt.xlabel('w/h')

        plt.title('microstrip patch antenna using ' + name)
        plt.show()

        errorplot = abs(np.asarray(out2).T - ant_plot.f2(np.asarray(inp2), 6))/ant_plot.f2(np.asarray(inp2), 6)*100

        max_error = max(errorplot[0])
        print("max error: ", max_error)

        avg_error = 0
        for i in errorplot[0]:
            avg_error += i
        avg_error = avg_error / len(errorplot[0])

        print("average erro: ", avg_error)

        plt.plot(np.asarray(inp2), errorplot[0], 'r--')
        plt.ylabel('absolute error')

        plt.xlabel('w/h')

        plt.title('%error using ' + name)
        plt.show()

        x = [ 1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.,7.5,8,8.5,9.,9.5]
        for i in x:
           ou =  clf.predict(i*10)
           print(ou/100)

def main():

    ant_plot.plot()

    inp_dat = pd.read_csv('train_in.csv')
    out_dat = pd.read_csv('train_out.csv')
    dat_backProp = pd.read_csv('train_BP.csv')
    inp_start = 1.0
    inp_end = 9.9
    inc = 0.1
    out_max = 7710.5576782

    backProp(dat_backProp, inp_start, inp_end, inc, out_max)

    inBuilt(inp_dat,out_dat, inp_start, inp_end, inc, out_max)




if __name__ == "__main__":
    main()
