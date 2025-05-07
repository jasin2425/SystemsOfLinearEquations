import matplotlib.pyplot as plt
import time
from linearEquation import *
class Jacobi:
    def __init__(self,sysMatrix,vectorB):
        self.sysMatrix=sysMatrix
        self.vectorB=vectorB

    def solveJacobi(self):
        residuum_array=[]
        #wielkosc macierzy
        n=len(self.vectorB)
        start = time.perf_counter()
        x=np.ones(n)
        resnorm=5
        maxIter=8000
        iter=0

        #macierz trojkatna dolna
        L = np.tril(self.sysMatrix, k=-1)
        #macierz trojkatna gorna
        U = np.triu(self.sysMatrix, k=1)
        #macierz diagnonalna
        D = np.diag(np.diag(self.sysMatrix))

        D_inv = np.linalg.inv(D)
        M= np.dot(-D_inv,(L+U))
        w=D_inv@self.vectorB

        while (iter<maxIter) and (resnorm>1e-9):
            new_x=M@x+w
            resnorm=np.linalg.norm(np.dot(self.sysMatrix, new_x) - self.vectorB)
            residuum_array.append(resnorm)
            x=new_x
            iter+=1
        end=time.perf_counter()
        #self.PlotResiduum(residuum_array)
        t_total = end - start
        return x,residuum_array,iter,t_total

    def PlotResiduum(self, res):
        res_np = np.array(res)
        plt.plot(res_np)
        plt.yscale('log')
        plt.xlabel("Numer iteracji")
        plt.ylabel("Norma residuum")
        plt.title("Zbieżność metody Jacobiego")
        plt.grid(True)
        plt.savefig("jacobi.png", dpi=300)
        plt.show()