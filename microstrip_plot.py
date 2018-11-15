import numpy as np
import matplotlib.pyplot as plt

# TODO change the range according to the device
t2 = np.arange(1.0, 10.0, 0.1)
t1 = np.arange(1, 10, 0.5)

# TODO change all these functions acc to device
def e1(n,er):
    
    return (er+1)/2 + (er-1)/2*(1/(np.sqrt(1+12/n)) + 0.04*np.square(1-n) )


def f1(n,er):
    return (60*np.log(8/n + n/4)/np.sqrt(e1(n,er)))
    
def e2(n,er):
    
    return (er+1)/2 + (er-1)/2*(1/(np.sqrt(1+12/n)))

#TODO keep this function name as f2. The above 3 functions may not be necessary. f2 returns the exact output for the given input.
def f2(n,er):
    return (120*np.pi/(n+1.393+0.667*np.log(n+1.444))/np.sqrt(e2(n,er)))


def printback():
    for t in t2:
        # TODO the below print is for backProp function. Format: [[[normalisef inp list],[normalised op list]], [[],[]] ,...]
        print("[[", t, "/9.8],[", f2(t, 2), "/", 98.5259785822, "]],")

def printinBuilt():
    inp = []
    out = []
    for t in t2:
        inp.append([t])
        # todo change the function parameters. er
        out.append(f2(t,2))

    # todo these two print for inBuilt funtions. Format- input: [[1st input list],[2nd..], ...] output: [output list]
    print(inp)
    print(out)


def plot():
    # todo change the function parameters. er
    plt.plot(t1, f1(t1, 2), 'b--', t1, f1(t1, 4), 'b--', t1, f1(t1, 10), 'b--', t1, f1(t1, 15), 'b--')
    ##TODO: change the parameters
    plt.ylabel('Impedance')
    plt.xlabel('w/h')
    plt.show()

    # todo change the function parameters. er
    plt.plot(t2, f2(t2, 2), 'r--', t2, f2(t2, 4), 'r--', t2, f2(t2, 10), 'r--', t2, f2(t2, 15), 'r--')
    ##TODO: change the parameters
    plt.ylabel('Impedance')
    plt.xlabel('w/h')
    plt.show()

def table():
    print(t1)
    print(f2(t1,2))


def main():
    plot()
    table()

if __name__ == "__main__":
    main()