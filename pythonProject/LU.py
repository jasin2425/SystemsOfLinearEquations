import matplotlib.pyplot as plt
import time

from linearEquation import *
class LU:
    def __init__(self,sysMatrix,vectorB):
        self.sysMatrix=sysMatrix
        self.vectorB=vectorB
    def sloveFirst(self,L):
        #L * z = vectorB
        n=len(self.vectorB)
        z=np.zeros(n)
        for i in range(n):
            z[i]=self.vectorB[i]-np.dot(L[i,:i],z[:i])
        return z
    def sloveSecond(self,U,z):
        #U*x=Z
        n = len(self.vectorB)
        x=np.zeros(n)
        for i in range(n-1,-1,-1):
            x[i]=(z[i]-np.dot(U[i,i+1:],x[i+1:]))/U[i,i]
        return x
    def solveLU(self):
        n=len(self.vectorB)
        start = time.perf_counter()

        L=np.eye(n)
        U=np.zeros((n,n))
        for i in range(n):
            for j in range(i, n):
                U[i,j]=self.sysMatrix[i,j]-np.dot(L[i,:i],U[:i,j])
            for j in range(i + 1, n):
                L[j, i] = (self.sysMatrix[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

        t_fact = time.perf_counter()

        z=self.sloveFirst(L)
        x=self.sloveSecond(U,z)
        t_end = time.perf_counter()

        r = np.linalg.norm(self.sysMatrix @ x - self.vectorB)
        t_factorization = t_fact - start
        t_substitution = t_end - t_fact
        t_total = t_end - start

        return x, r, t_total, t_factorization, t_substitution
