import numpy as np

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
            resid=np.sqrt(np.sum(self.calcResiduum(self.sysMatrix,self.vectorB,new_x)**2))
            res.append(resid)
            Resnorm=resid
            x=new_x
            iter+=1
        self.PlotResiduum(res)
        return x,res,iter
    def PlotResiduum(self,res):
        res_np=np.array(res)
        plt.plot(res_np)
        plt.yscale('log')
        plt.show()