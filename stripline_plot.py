import numpy as np
import matplotlib.pyplot as plt


t2 = np.arange(0.15, 1.0, 0.01)
t1 = np.arange(0.2,1.0,0.05)


def kk(n):
    op = []
    for i in n:
        k = np.tanh(np.pi * i / 4)
        kd = np.sqrt(1 - np.square(k))
        if (k < 0.707):
            op.append(np.pi / (np.log(2 * (1 + np.sqrt(kd)) / (1 - np.sqrt(kd)))))

        else:
            op.append(np.log(2 * (1 + np.sqrt(k)) / (1 - np.sqrt(k))) / np.pi)
    return(np.asarray(op))

def f2(n, er):
   return (30*np.pi/(kk(n)*np.sqrt(er)))


def printback():
    for t in t2:
        print("[[", t, "/0.99],[", f2([t], 6)[0], "/",86.3722657126, "]],")

def printinBuilt():
    inp = []
    out = []
    for t in t2:
        inp.append([t])
        out.append(list(f2([t],6))[0])

    print(inp)
    print(out)



def plot():

    plt.plot( t2, f2(t2, 6), 'r--')

    plt.ylabel('Impedance')
    plt.xlabel('W/a')
    plt.show()


def table():
    print(t1)
    for i in t1:
        print(f2([i],6)[0])


def main():
    #plot()
    table()
    #printinBuilt()
    #printback()


if __name__ == "__main__":
    main()