import cortical
import util
import numpy as np

def main():
    print("hello")
    sgen = util.InputGenerator((3, 3),np.array([[0,1],[1,0]]),[0.2,0.7])
    print(sgen.getInput(count=2))

if __name__ == "__main__":
    main()