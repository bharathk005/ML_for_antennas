import numpy as np
import matplotlib.pyplot as plt


t2 = np.arange(1, 10, 0.1)

t1 = np.arange(1,10,0.5)

def eff(n,er):
    return(((er+1)/2) + ((er-1)/(np.sqrt(1+12/n)*2)) )

def delL(n):
    return 0

def f2(n, er):
   return (300/(2*np.sqrt(eff(n, er))*(9.5+2*delL(n)))*1000)


def printback():
    for t in t2:
        print("[[", t, "/9.9],[", f2(t, 6), "/",7710.5576782 , "]],")

def printinBuilt():
    inp = []
    out = []
    for t in t2:
        inp.append([t])
        out.append(f2(t,6))

    print(inp)
    print(out)



def table():
    print(t1,f2(t1,6))

def plot():

    plt.plot( t2, f2(t2, 6), 'r--')

    plt.ylabel('fr in MHz')
    plt.xlabel('w/h')
    plt.show()




def main():
    #plot()
    table()
    #printinBuilt()
    #printback()


if __name__ == "__main__":
    main()