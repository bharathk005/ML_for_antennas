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
        7710.5576782020507, 7682.1578746061996, 7655.6174716576961, 7630.7019034354735, 7607.2209452931811,
        7585.0178118199156, 7563.9614783489869, 7543.9411319207466, 7524.8620712292413, 7506.6426181070365,
        7489.2117511489032, 7472.5072651440669, 7456.4743201334259, 7441.0642837415498, 7426.2337973961094,
        7411.9440156597011, 7398.1599809766494, 7384.850105474884, 7371.9857382288155, 7359.5408013574552,
        7347.4914820261001, 7335.8159701976529, 7324.4942340908083, 7313.5078269223777, 7302.8397197657814,
        7292.4741563377738, 7282.3965262969532, 7272.5932542497285, 7263.0517021481355, 7253.7600831570326,
        7244.7073853861602, 7235.8833041413018, 7227.2781815606268, 7218.8829526764512, 7210.6890970866871,
        7202.6885955399266, 7194.8738908379173, 7187.2378525428257, 7179.7737450471695, 7172.4751986236515,
        7165.3361831226521, 7158.3509840280012, 7151.5141806182828, 7144.8206260123607, 7138.2654289046914,
        7131.8439368193685, 7125.5517207318335, 7119.3845609246382, 7113.3384339588356, 7107.4095006557,
        7101.5940949950673, 7095.8887138466307, 7090.2900074594454, 7084.7947706426303, 7079.399934577179,
        7074.1025592048454, 7068.8998261454226, 7063.7890320985443, 7058.767582690346, 7053.8329867290986,
        7048.9828508372921, 7044.2148744306724, 7039.5268450173844, 7034.9166337928291, 7030.382191507967,
        7025.9215445907894, 7021.5327915023772, 7017.2140993106123, 7012.9637004659653, 7008.7798897651464,
        7004.6610214894954, 7000.6055067061188, 6996.6118107207049, 6992.6784506718132, 6988.8039932572829,
        6984.9870525840479, 6981.2262881334009, 6977.5204028342669, 6973.8681412376463, 6970.2682877859042,
        6966.7196651710055, 6963.2211327762216, 6959.771585196283, 6956.3699508312275, 6953.0151905495832,
        6949.7062964168026, 6946.4422904851563, 6943.22222364151, 6940.0451745097371, 6936.9102484046216
    ]
    dat_backProp = [
        [[1.0 / 9.9], [7710.5576782 / 7710.5576782]],
        [[1.1 / 9.9], [7682.15787461 / 7710.5576782]],
        [[1.2 / 9.9], [7655.61747166 / 7710.5576782]],
        [[1.3 / 9.9], [7630.70190344 / 7710.5576782]],
        [[1.4 / 9.9], [7607.22094529 / 7710.5576782]],
        [[1.5 / 9.9], [7585.01781182 / 7710.5576782]],
        [[1.6 / 9.9], [7563.96147835 / 7710.5576782]],
        [[1.7 / 9.9], [7543.94113192 / 7710.5576782]],
        [[1.8 / 9.9], [7524.86207123 / 7710.5576782]],
        [[1.9 / 9.9], [7506.64261811 / 7710.5576782]],
        [[2.0 / 9.9], [7489.21175115 / 7710.5576782]],
        [[2.1 / 9.9], [7472.50726514 / 7710.5576782]],
        [[2.2 / 9.9], [7456.47432013 / 7710.5576782]],
        [[2.3 / 9.9], [7441.06428374 / 7710.5576782]],
        [[2.4 / 9.9], [7426.2337974 / 7710.5576782]],
        [[2.5 / 9.9], [7411.94401566 / 7710.5576782]],
        [[2.6 / 9.9], [7398.15998098 / 7710.5576782]],
        [[2.7 / 9.9], [7384.85010547 / 7710.5576782]],
        [[2.8 / 9.9], [7371.98573823 / 7710.5576782]],
        [[2.9 / 9.9], [7359.54080136 / 7710.5576782]],
        [[3.0 / 9.9], [7347.49148203 / 7710.5576782]],
        [[3.1 / 9.9], [7335.8159702 / 7710.5576782]],
        [[3.2 / 9.9], [7324.49423409 / 7710.5576782]],
        [[3.3 / 9.9], [7313.50782692 / 7710.5576782]],
        [[3.4 / 9.9], [7302.83971977 / 7710.5576782]],
        [[3.5 / 9.9], [7292.47415634 / 7710.5576782]],
        [[3.6 / 9.9], [7282.3965263 / 7710.5576782]],
        [[3.7 / 9.9], [7272.59325425 / 7710.5576782]],
        [[3.8 / 9.9], [7263.05170215 / 7710.5576782]],
        [[3.9 / 9.9], [7253.76008316 / 7710.5576782]],
        [[4.0 / 9.9], [7244.70738539 / 7710.5576782]],
        [[4.1 / 9.9], [7235.88330414 / 7710.5576782]],
        [[4.2 / 9.9], [7227.27818156 / 7710.5576782]],
        [[4.3 / 9.9], [7218.88295268 / 7710.5576782]],
        [[4.4 / 9.9], [7210.68909709 / 7710.5576782]],
        [[4.5 / 9.9], [7202.68859554 / 7710.5576782]],
        [[4.6 / 9.9], [7194.87389084 / 7710.5576782]],
        [[4.7 / 9.9], [7187.23785254 / 7710.5576782]],
        [[4.8 / 9.9], [7179.77374505 / 7710.5576782]],
        [[4.9 / 9.9], [7172.47519862 / 7710.5576782]],
        [[5.0 / 9.9], [7165.33618312 / 7710.5576782]],
        [[5.1 / 9.9], [7158.35098403 / 7710.5576782]],
        [[5.2 / 9.9], [7151.51418062 / 7710.5576782]],
        [[5.3 / 9.9], [7144.82062601 / 7710.5576782]],
        [[5.4 / 9.9], [7138.2654289 / 7710.5576782]],
        [[5.5 / 9.9], [7131.84393682 / 7710.5576782]],
        [[5.6 / 9.9], [7125.55172073 / 7710.5576782]],
        [[5.7 / 9.9], [7119.38456092 / 7710.5576782]],
        [[5.8 / 9.9], [7113.33843396 / 7710.5576782]],
        [[5.9 / 9.9], [7107.40950066 / 7710.5576782]],
        [[6.0 / 9.9], [7101.594095 / 7710.5576782]],
        [[6.1 / 9.9], [7095.88871385 / 7710.5576782]],
        [[6.2 / 9.9], [7090.29000746 / 7710.5576782]],
        [[6.3 / 9.9], [7084.79477064 / 7710.5576782]],
        [[6.4 / 9.9], [7079.39993458 / 7710.5576782]],
        [[6.5 / 9.9], [7074.1025592 / 7710.5576782]],
        [[6.6 / 9.9], [7068.89982615 / 7710.5576782]],
        [[6.7 / 9.9], [7063.7890321 / 7710.5576782]],
        [[6.8 / 9.9], [7058.76758269 / 7710.5576782]],
        [[6.9 / 9.9], [7053.83298673 / 7710.5576782]],
        [[7.0 / 9.9], [7048.98285084 / 7710.5576782]],
        [[7.1 / 9.9], [7044.21487443 / 7710.5576782]],
        [[7.2 / 9.9], [7039.52684502 / 7710.5576782]],
        [[7.3 / 9.9], [7034.91663379 / 7710.5576782]],
        [[7.4 / 9.9], [7030.38219151 / 7710.5576782]],
        [[7.5 / 9.9], [7025.92154459 / 7710.5576782]],
        [[7.6 / 9.9], [7021.5327915 / 7710.5576782]],
        [[7.7 / 9.9], [7017.21409931 / 7710.5576782]],
        [[7.8 / 9.9], [7012.96370047 / 7710.5576782]],
        [[7.9 / 9.9], [7008.77988977 / 7710.5576782]],
        [[8.0 / 9.9], [7004.66102149 / 7710.5576782]],
        [[8.1 / 9.9], [7000.60550671 / 7710.5576782]],
        [[8.2 / 9.9], [6996.61181072 / 7710.5576782]],
        [[8.3 / 9.9], [6992.67845067 / 7710.5576782]],
        [[8.4 / 9.9], [6988.80399326 / 7710.5576782]],
        [[8.5 / 9.9], [6984.98705258 / 7710.5576782]],
        [[8.6 / 9.9], [6981.22628813 / 7710.5576782]],
        [[8.7 / 9.9], [6977.52040283 / 7710.5576782]],
        [[8.8 / 9.9], [6973.86814124 / 7710.5576782]],
        [[8.9 / 9.9], [6970.26828779 / 7710.5576782]],
        [[9.0 / 9.9], [6966.71966517 / 7710.5576782]],
        [[9.1 / 9.9], [6963.22113278 / 7710.5576782]],
        [[9.2 / 9.9], [6959.7715852 / 7710.5576782]],
        [[9.3 / 9.9], [6956.36995083 / 7710.5576782]],
        [[9.4 / 9.9], [6953.01519055 / 7710.5576782]],
        [[9.5 / 9.9], [6949.70629642 / 7710.5576782]],
        [[9.6 / 9.9], [6946.44229049 / 7710.5576782]],
        [[9.7 / 9.9], [6943.22222364 / 7710.5576782]],
        [[9.8 / 9.9], [6940.04517451 / 7710.5576782]],
        [[9.9 / 9.9], [6936.9102484 / 7710.5576782]]
    ]  ##100
    inp_start = 1.0
    inp_end = 9.9
    inc = 0.1
    out_max = 7710.5576782

    backProp(dat_backProp, inp_start, inp_end, inc, out_max)

    inBuilt(inp_dat,out_dat, inp_start, inp_end, inc, out_max)




if __name__ == "__main__":
    main()
