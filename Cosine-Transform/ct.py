import sys
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_wave(x, path = './wave.png'):
    plt.gcf().clear()
    plt.plot(x)
    plt.xlabel('n')
    plt.ylabel('xn')
    plt.savefig(path)

def plot_ak(a, path = './freq.png'):
    plt.gcf().clear()
    # Only plot the mag of a
    a = np.abs(a)
    plt.plot(a)
    plt.xlabel('k')
    plt.ylabel('ak')
    plt.savefig(path)

def CosineTrans(x, B):
    # implement cosine transform
    inverse_B = np.linalg.inv(B)
    a = np.matmul(inverse_B,x)
    return a

def InvCosineTrans(a, B):
    # implement inverse cosine transform
    x = np.matmul(B,a)
    return x

def gen_basis(N):
    B = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(N):
            if i == 0:
                B[j][i] = 1 / math.sqrt(N)
            else:
                B[j][i] = (math.sqrt(2/N)) * math.cos((j+0.5) * i * math.pi /N)
    return B

if __name__ == '__main__':
    signal_path = sys.argv[1]
    x = np.loadtxt(signal_path).reshape(-1,1)
    B = gen_basis(1000)
    a = CosineTrans(x,B)
    plot_ak(a)

    a_f1 = [0 for i in range(len(a))]
    a_f3 = [0 for i in range(len(a))]
    a_f1[35:55] = a[35:55]
    a_f3[475:495] = a[475:495]

    f1 = InvCosineTrans(a_f1,B)
    f3 = InvCosineTrans(a_f3,B)

    np.savetxt('b08901021_f1.txt',f1)
    np.savetxt('b08901021_f3.txt',f3)


