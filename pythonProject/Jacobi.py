
from linearEquation import *
class Jacobi:
    def __init__(self,sysMatrix,vectorB):
        self.sysMatrix=sysMatrix
        self.vectorB=vectorB

    def calcResiduum(self,sysMatrix,vectorB,x):
        return np.dot(sysMatrix,x)-vectorB

    def solveJacobi(self):
        res=[]
        n=len(self.vectorB)
        x=np.ones(n)
        Resnorm=5
        maxIter=8000
        iter=0
        L = np.tril(self.sysMatrix, k=-1)
        U = np.triu(self.sysMatrix, k=1)
        D = np.diag(np.diag(self.sysMatrix))
        D_inv = np.diag(1 / np.diag(D))
        while (iter<maxIter) and (Resnorm>1e-9):
            new_x=np.dot(np.dot(-1,D_inv),np.dot((L+U),x))+np.dot(D_inv,self.vectorB)
            Resnorm=np.sqrt(np.sum(self.calcResiduum(self.sysMatrix,self.vectorB,new_x)**2))
            res.append(Resnorm)
            x=new_x
            iter+=1
        self.PlotResiduum(res)
        return x,res,iter
    def PlotResiduum(self,res):
        res_np=np.array(res)
        plt.plot(res_np)
        plt.yscale('log')
        plt.show()

#1 2 0
#2 1 2
#0 2 1

# 1 0 0
# 0 1 0
# 0 0 1

# 0 2 0
# 0 0 2
# 0 0 0

# 0 0 0
# 2 0 0
# 0 0 1