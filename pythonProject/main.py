import pandas as pd
from linearEquation import LinearEquation
from Jacobi import Jacobi
from Gauss import  Gauss
#macierz N=1293, a1=14 a2=-1 a3=-1 f=7
#wektor b_i=sin(i*7)
def main():
    n=1293
    a1=14
    a2=-1
    a3=-1
    f=7
    eq=LinearEquation(n,a1,a2,a3,f)
    sysMatrix,vectorB =eq.createEquation()
    jacobi=Jacobi(sysMatrix,vectorB)
    jacobi.solveJacobi()
    gauss=Gauss(sysMatrix,vectorB)
    gauss.solveGauss()
main()