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
import microstrip_plot
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
    #TODO: change the file name inside the plot function
    plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', x1, microstrip_plot.f2(x1, 2), 'go')
    #TODO: change the parameter names if they are different
    plt.ylabel('Impedance')
    plt.xlabel('w/h')
 # todo change title
    plt.title('microstrip using BackPropagation')
    plt.show()
    #TODO change the file name in the below equation
    errorplot = abs(np.asarray(out2).T - microstrip_plot.f2(x1, 2))/ microstrip_plot.f2(x1, 2)*100

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
    #TODO change the parameter name if it is deiiferent
    plt.xlabel('w/h')
 # todo change title
    plt.title('%error using BackPropagation')
    plt.show()


    x = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5]
    for i in x:
        ou = myNN2.fun([i / 9.9])
        print(ou[0] * 98.525)


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
        # TODO: change the file name inside the plot function
        plt.plot(np.asarray(inp2), np.asarray(out2), 'b--', np.asarray(inp2), microstrip_plot.f2(np.asarray(inp2), 2), 'go')
        plt.ylabel('Impedance')
        plt.xlabel('w/h')
        # todo change title
        plt.title('microstrip using ' + name)
        plt.show()
        # TODO: change the file name inside the equation
        errorplot = abs(np.asarray(out2).T - microstrip_plot.f2(np.asarray(inp2), 2))/microstrip_plot.f2(np.asarray(inp2), 2)*100

        max_error = max(errorplot[0])
        print("max error: ", max_error)

        avg_error = 0
        for i in errorplot[0]:
            avg_error += i
        avg_error = avg_error / len(errorplot[0])

        print("average erro: ", avg_error)

        plt.plot(np.asarray(inp2), errorplot[0], 'r--')
        plt.ylabel('absolute error')
        # TODO: change the parameters if it is different
        plt.xlabel('w/h')
        # todo change title
        plt.title('%error using ' + name)
        plt.show()

        x = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5]
        for i in x:
            ou = clf.predict(i * 10)
            print(ou[0] / 100)

def main():
    #TODO change the below file name, all the relavant data, input, output.
    microstrip_plot.plot()
    inp_dat = [
        [1.0], [1.1000000000000001], [1.2000000000000002], [1.3000000000000003], [1.4000000000000004],
        [1.5000000000000004], [1.6000000000000005], [1.7000000000000006], [1.8000000000000007], [1.9000000000000008],
        [2.0000000000000009], [2.100000000000001], [2.2000000000000011], [2.3000000000000012], [2.4000000000000012],
        [2.5000000000000013], [2.6000000000000014], [2.7000000000000015], [2.8000000000000016], [2.9000000000000017],
        [3.0000000000000018], [3.1000000000000019], [3.200000000000002], [3.300000000000002], [3.4000000000000021],
        [3.5000000000000022], [3.6000000000000023], [3.7000000000000024], [3.8000000000000025], [3.9000000000000026],
        [4.0000000000000027], [4.1000000000000032], [4.2000000000000028], [4.3000000000000025], [4.400000000000003],
        [4.5000000000000036], [4.6000000000000032], [4.7000000000000028], [4.8000000000000034], [4.9000000000000039],
        [5.0000000000000036], [5.1000000000000032], [5.2000000000000037], [5.3000000000000043], [5.4000000000000039],
        [5.5000000000000036], [5.6000000000000041], [5.7000000000000046], [5.8000000000000043], [5.9000000000000039],
        [6.0000000000000044], [6.100000000000005], [6.2000000000000046], [6.3000000000000043], [6.4000000000000048],
        [6.5000000000000053], [6.600000000000005], [6.7000000000000046], [6.8000000000000052], [6.9000000000000057],
        [7.0000000000000053], [7.100000000000005], [7.2000000000000055], [7.300000000000006], [7.4000000000000057],
        [7.5000000000000053], [7.6000000000000059], [7.7000000000000064], [7.800000000000006], [7.9000000000000057],
        [8.0000000000000071], [8.1000000000000068], [8.2000000000000064], [8.300000000000006], [8.4000000000000057],
        [8.5000000000000071], [8.6000000000000068], [8.7000000000000064], [8.8000000000000078], [8.9000000000000075],
        [9.0000000000000071], [9.1000000000000068], [9.2000000000000064], [9.3000000000000078], [9.4000000000000075],
        [9.5000000000000071], [9.6000000000000085], [9.7000000000000082], [9.8000000000000078], [9.9000000000000075]
    ]
    out_dat = [
        98.525978582166715, 94.339383929699039, 90.519281339083705, 87.01789532782557, 83.795572015452208,
        80.819100838752576, 78.060440368060554, 75.495737946631749, 73.104566336126993, 70.869322878796268,
        68.774751901355131, 66.807561644214459, 64.956114447036541, 63.210174251291001, 61.560699345280483,
        59.999671113548693, 58.519951657447415, 57.115164731651035, 55.779595635713243, 54.508106611681342,
        53.296065000864765, 52.139281957579897, 51.033959943421969, 49.976647560627271, 48.964200548432245,
        47.993747977784665, 47.062662849247843, 46.168536435536467, 45.309155820793961, 44.482484178833083,
        43.686643406285, 42.919898787188934, 42.180645415563639, 41.467396143950666, 40.778770860409495,
        40.11348692525339, 39.470350622966883, 38.848249505063777, 38.246145516795032, 37.663068815140456,
        37.098112197855215, 36.550426073854794, 36.019213914204862, 35.503728130678965, 35.003266335460751,
        34.517167941263899, 34.044811066062593, 33.585609710884498, 33.139011182813341, 32.704493738562228,
        32.281564426780498, 31.869757109703144, 31.468630646893985, 31.077767225711732, 30.696770824778994,
        30.325265798186859, 29.962895569449969, 29.609321425358978, 29.264221400879901, 28.927289247138578,
        28.598233475317574, 28.276776469994896, 27.962653666079429, 27.655612784055723, 27.355413118749503,
        27.061824877271278, 26.774628562195328, 26.493614396389884, 26.218581786236427, 25.949338820266515,
        25.685701800504958, 25.427494804044333, 25.17454927258802, 24.926703627891687, 24.683802911206644,
        24.445698444986697, 24.212247515262806, 23.983313073219893, 23.758763454628145, 23.538472115888759,
        23.322317385551738, 23.110182230252736, 22.901954034097347, 22.697524390595639, 22.496788906317764,
        22.299647015503751, 22.10600180491765, 21.915759848288435, 21.728831049728086, 21.5451284955614
    ]
    dat_backProp = [
        [[1.0 / 9.9], [98.5259785822 / 98.5259785822]],
        [[1.1 / 9.9], [94.3393839297 / 98.5259785822]],
        [[1.2 / 9.9], [90.5192813391 / 98.5259785822]],
        [[1.3 / 9.9], [87.0178953278 / 98.5259785822]],
        [[1.4 / 9.9], [83.7955720155 / 98.5259785822]],
        [[1.5 / 9.9], [80.8191008388 / 98.5259785822]],
        [[1.6 / 9.9], [78.0604403681 / 98.5259785822]],
        [[1.7 / 9.9], [75.4957379466 / 98.5259785822]],
        [[1.8 / 9.9], [73.1045663361 / 98.5259785822]],
        [[1.9 / 9.9], [70.8693228788 / 98.5259785822]],
        [[2.0 / 9.9], [68.7747519014 / 98.5259785822]],
        [[2.1 / 9.9], [66.8075616442 / 98.5259785822]],
        [[2.2 / 9.9], [64.956114447 / 98.5259785822]],
        [[2.3 / 9.9], [63.2101742513 / 98.5259785822]],
        [[2.4 / 9.9], [61.5606993453 / 98.5259785822]],
        [[2.5 / 9.9], [59.9996711135 / 98.5259785822]],
        [[2.6 / 9.9], [58.5199516574 / 98.5259785822]],
        [[2.7 / 9.9], [57.1151647317 / 98.5259785822]],
        [[2.8 / 9.9], [55.7795956357 / 98.5259785822]],
        [[2.9 / 9.9], [54.5081066117 / 98.5259785822]],
        [[3.0 / 9.9], [53.2960650009 / 98.5259785822]],
        [[3.1 / 9.9], [52.1392819576 / 98.5259785822]],
        [[3.2 / 9.9], [51.0339599434 / 98.5259785822]],
        [[3.3 / 9.9], [49.9766475606 / 98.5259785822]],
        [[3.4 / 9.9], [48.9642005484 / 98.5259785822]],
        [[3.5 / 9.9], [47.9937479778 / 98.5259785822]],
        [[3.6 / 9.9], [47.0626628492 / 98.5259785822]],
        [[3.7 / 9.9], [46.1685364355 / 98.5259785822]],
        [[3.8 / 9.9], [45.3091558208 / 98.5259785822]],
        [[3.9 / 9.9], [44.4824841788 / 98.5259785822]],
        [[4.0 / 9.9], [43.6866434063 / 98.5259785822]],
        [[4.1 / 9.9], [42.9198987872 / 98.5259785822]],
        [[4.2 / 9.9], [42.1806454156 / 98.5259785822]],
        [[4.3 / 9.9], [41.467396144 / 98.5259785822]],
        [[4.4 / 9.9], [40.7787708604 / 98.5259785822]],
        [[4.5 / 9.9], [40.1134869253 / 98.5259785822]],
        [[4.6 / 9.9], [39.470350623 / 98.5259785822]],
        [[4.7 / 9.9], [38.8482495051 / 98.5259785822]],
        [[4.8 / 9.9], [38.2461455168 / 98.5259785822]],
        [[4.9 / 9.9], [37.6630688151 / 98.5259785822]],
        [[5.0 / 9.9], [37.0981121979 / 98.5259785822]],
        [[5.1 / 9.9], [36.5504260739 / 98.5259785822]],
        [[5.2 / 9.9], [36.0192139142 / 98.5259785822]],
        [[5.3 / 9.9], [35.5037281307 / 98.5259785822]],
        [[5.4 / 9.9], [35.0032663355 / 98.5259785822]],
        [[5.5 / 9.9], [34.5171679413 / 98.5259785822]],
        [[5.6 / 9.9], [34.0448110661 / 98.5259785822]],
        [[5.7 / 9.9], [33.5856097109 / 98.5259785822]],
        [[5.8 / 9.9], [33.1390111828 / 98.5259785822]],
        [[5.9 / 9.9], [32.7044937386 / 98.5259785822]],
        [[6.0 / 9.9], [32.2815644268 / 98.5259785822]],
        [[6.1 / 9.9], [31.8697571097 / 98.5259785822]],
        [[6.2 / 9.9], [31.4686306469 / 98.5259785822]],
        [[6.3 / 9.9], [31.0777672257 / 98.5259785822]],
        [[6.4 / 9.9], [30.6967708248 / 98.5259785822]],
        [[6.5 / 9.9], [30.3252657982 / 98.5259785822]],
        [[6.6 / 9.9], [29.9628955694 / 98.5259785822]],
        [[6.7 / 9.9], [29.6093214254 / 98.5259785822]],
        [[6.8 / 9.9], [29.2642214009 / 98.5259785822]],
        [[6.9 / 9.9], [28.9272892471 / 98.5259785822]],
        [[7.0 / 9.9], [28.5982334753 / 98.5259785822]],
        [[7.1 / 9.9], [28.27677647 / 98.5259785822]],
        [[7.2 / 9.9], [27.9626536661 / 98.5259785822]],
        [[7.3 / 9.9], [27.6556127841 / 98.5259785822]],
        [[7.4 / 9.9], [27.3554131187 / 98.5259785822]],
        [[7.5 / 9.9], [27.0618248773 / 98.5259785822]],
        [[7.6 / 9.9], [26.7746285622 / 98.5259785822]],
        [[7.7 / 9.9], [26.4936143964 / 98.5259785822]],
        [[7.8 / 9.9], [26.2185817862 / 98.5259785822]],
        [[7.9 / 9.9], [25.9493388203 / 98.5259785822]],
        [[8.0 / 9.9], [25.6857018005 / 98.5259785822]],
        [[8.1 / 9.9], [25.427494804 / 98.5259785822]],
        [[8.2 / 9.9], [25.1745492726 / 98.5259785822]],
        [[8.3 / 9.9], [24.9267036279 / 98.5259785822]],
        [[8.4 / 9.9], [24.6838029112 / 98.5259785822]],
        [[8.5 / 9.9], [24.445698445 / 98.5259785822]],
        [[8.6 / 9.9], [24.2122475153 / 98.5259785822]],
        [[8.7 / 9.9], [23.9833130732 / 98.5259785822]],
        [[8.8 / 9.9], [23.7587634546 / 98.5259785822]],
        [[8.9 / 9.9], [23.5384721159 / 98.5259785822]],
        [[9.0 / 9.9], [23.3223173856 / 98.5259785822]],
        [[9.1 / 9.9], [23.1101822303 / 98.5259785822]],
        [[9.2 / 9.9], [22.9019540341 / 98.5259785822]],
        [[9.3 / 9.9], [22.6975243906 / 98.5259785822]],
        [[9.4 / 9.9], [22.4967889063 / 98.5259785822]],
        [[9.5 / 9.9], [22.2996470155 / 98.5259785822]],
        [[9.6 / 9.9], [22.1060018049 / 98.5259785822]],
        [[9.7 / 9.9], [21.9157598483 / 98.5259785822]],
        [[9.8 / 9.9], [21.7288310497 / 98.5259785822]],
        [[9.9 / 9.9], [21.5451284956 / 98.5259785822]]
    ]  ##100
    inp_start = 1.1
    inp_end = 9.9
    inc = 0.1
    out_max = 98.5259785822

    backProp(dat_backProp, inp_start, inp_end, inc, out_max)

    inBuilt(inp_dat,out_dat, inp_start, inp_end, inc, out_max)




if __name__ == "__main__":
    main()
