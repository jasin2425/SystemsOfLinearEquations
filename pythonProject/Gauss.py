import matplotlib.pyplot as plt
import time

from linearEquation import *
class Gauss:
    def __init__(self,sysMatrix,vectorB):
        self.sysMatrix=sysMatrix
        self.vectorB=vectorB

    def calcResiduum(self,sysMatrix,vectorB,x):
        return np.dot(sysMatrix,x)-vectorB

    def solveGauss(self):
        res=[]
        n=len(self.vectorB)
        start = time.perf_counter()

        x=np.ones(n)
        Resnorm=5
        maxIter=8000
        iter=0
        L = np.tril(self.sysMatrix, k=-1)
        U = np.triu(self.sysMatrix, k=1)
        D = np.diag(np.diag(self.sysMatrix))
        T=D+L
        T_inv = np.linalg.inv(T)

        while (iter<maxIter) and (Resnorm>1e-9):
            new_x=np.dot(-T_inv,np.dot(U,x))+np.dot(T_inv,self.vectorB)
            Resnorm=np.linalg.norm(np.dot(self.sysMatrix, new_x) - self.vectorB)
            res.append(Resnorm)
            x=new_x
            iter+=1
        end = time.perf_counter()

        #self.PlotResiduum(res)
        t_total = end - start

        return x,res,iter,t_total
    def PlotResiduum(self,res):
        res_np = np.array(res)
        plt.plot(res_np)
        plt.yscale('log')
        plt.xlabel("Numer iteracji")
        plt.ylabel("Norma residuum")
        plt.title("Zbieżność metody Gaussa-Seidla")
        plt.grid(True)
        plt.savefig("Gauss-Seidel.png", dpi=300)
        plt.show()