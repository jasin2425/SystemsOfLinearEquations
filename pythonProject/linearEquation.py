import numpy as np
import matplotlib.pyplot as plt
#macierz N=1293, a1=14 a2=-1 a3=-1 f=7
#wektor b_i=sin(i*7)
class LinearEquation:
    def __init__(self,n,a1,a2,a3,f):
        self.n=n
        self.a1=a1
        self.a2=a2
        self.a3=a3
        self.f=f
    def systemMatrix(self,n,a1,a2,a3):
        sysMatrix=np.zeros((n,n))
        for i in range(n):
            sysMatrix[i,i]=a1
            if i<n-2:
                sysMatrix[i,i+2]=a3
                sysMatrix[i+2,i]=a3
            if i<n-1:
                sysMatrix[i,i+1]=a2
                sysMatrix[i+1,i]=a2
        return sysMatrix
    def awakeVector(self,n,f):
        vectorB=np.zeros(n)
        for i in range(n):
            vectorB[i]=np.sin(i*(f+1))
        return vectorB
    def createEquation(self):
        sysMatrix=self.systemMatrix(self.n,self.a1,self.a2,self.a3)
        vectorB=self.awakeVector(self.n,self.f)
        return sysMatrix,vectorB

