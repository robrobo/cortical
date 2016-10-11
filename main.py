import cortical
import util
import numpy as np
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def main():
    print("hello")
    shape = 10
    sgen = util.InputGenerator(shape,np.array([[1,0,1,0,1,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,1,0,1,0]]),[0.1,0.25, 0.1], noise=0.2)
    network = cortical.CortialNN(shape)
    print(network.connections)
    for _ in range(10000):
        input = sgen.getInput()[0]
        #print("iteration")
        #print(input)
        network.tick(input)
        #print(network.previous)

    print(network.connections)

if __name__ == "__main__":
    main()