from linearEquation import LinearEquation
from Jacobi import Jacobi
from Gauss import Gauss
from LU import LU
import matplotlib.pyplot as plt

def main():
    a1 = 14
    a2 = -1
    a3 = -1
    f = 7

    sizes = [100, 250,500,750,1000,1250,1500,1750,2000,2500,3000]
    time_lu = []
    time_jacobi = []
    time_gauss = []

    for n in sizes:
        print(f"Rozmiar macierzy: {n}")
        eq = LinearEquation(n, a1, a2, a3, f)
        sysMatrix, vectorB = eq.createEquation()

        jacobi = Jacobi(sysMatrix, vectorB)
        _, _, _, t_jacobi = jacobi.solveJacobi()

        gauss = Gauss(sysMatrix, vectorB)
        _, _, _, t_gauss = gauss.solveGauss()

        lu = LU(sysMatrix, vectorB)
        _, r, t_lu, _, _ = lu.solveLU()

        time_jacobi.append(t_jacobi)
        time_gauss.append(t_gauss)
        time_lu.append(t_lu)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Skala liniowa
    ax1.plot(sizes, time_jacobi, label='Jacobi', marker='o')
    ax1.plot(sizes, time_gauss, label='Gauss-Seidel', marker='o')
    ax1.plot(sizes, time_lu, label='LU', marker='o')
    ax1.set_title("Czas rozwiązania vs rozmiar macierzy (skala liniowa)")
    ax1.set_xlabel("Rozmiar")
    ax1.set_ylabel("Czas rozwiązania (s)")
    ax1.grid(True)
    ax1.legend()

    # Skala logarytmiczna
    ax2.plot(sizes, time_jacobi, label='Jacobi', marker='o')
    ax2.plot(sizes, time_gauss, label='Gauss-Seidel', marker='o')
    ax2.plot(sizes, time_lu, label='LU', marker='o')
    ax2.set_yscale('log')
    ax2.set_title("Czas rozwiązania vs od rozmiaru macierzy (skala logarytmiczna)")
    ax2.set_xlabel("Rozmiar")
    ax2.set_ylabel("Czas (s)")
    ax2.grid(True, which='both')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("czas_vs_rozmiar_lin_log.png", dpi=300)
    plt.show()


main()
