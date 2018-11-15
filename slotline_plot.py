import numpy as np
import matplotlib.pyplot as plt


t1 = np.arange(0.2, 1, 0.05)  #  W/D
t2 = np.arange(0.2, 1.0, 0.005)




def dbylamda(er):
    return 0.001#(0.25/np.sqrt(er-1))

def f2(n, er):
    return -( 113.19 - 53.55*np.log(er) + 1.25*n*( 114.59 - 51.88*np.log(er))+ 20*(n - 0.2)*(1 - n)- (0.15 + 0.23*np.log(er)
                    + n*(-0.79 + 2.07*np.log(er)))*(np.power(10.25 - 5*np.log(er) + n*(2.1 - 1.42*np.log(er))- dbylamda(er)*np.power(10,2),2)))


def printback():
    for t in t2:
        # TODO the below print is for backProp function. Format: [[[normalisef inp list],[normalised op list]], [[],[]] ,...]
        print("[[", t, "/0.995],[", f2(t, 16), "/", 246.954278861, "]],")

def printinBuilt():
    inp = []
    out = []
    for t in t2:
        inp.append([t])
        out.append(f2(t,16))

    # todo these two print for inBuilt funtions. Format- input: [[1st input list],[2nd..], ...] output: [output list]
    print(inp)
    print(out)


def plot():

    plt.plot( t2, f2(t2, 16), 'r--', t2, f2(t2, 15), 'r--')

    plt.ylabel('Impedance')
    plt.xlabel('W/d')
    plt.show()

def table():
    print(t1)
    print(f2(t1, 16))


def main():
    #plot()
    table()

if __name__ == "__main__":
    main()

