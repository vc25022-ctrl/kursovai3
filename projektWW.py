import numpy as np
import matplotlib.pyplot as plt

# Метод Рунге-Кутты 4-го порядка
def rk4(f, a, b, y0, h):
    t = np.arange(a, b + h / 2, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(len(t) - 1):
        hi = t[i + 1] - t[i]

        k1 = f(t[i], y[i])
        k2 = f(t[i] + hi/2, y[i] + hi*k1/2)
        k3 = f(t[i] + hi/2, y[i] + hi*k2/2)
        k4 = f(t[i] + hi, y[i] + hi*k3)

        y[i + 1] = y[i] + hi * (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y

# 1. Тестовая задача

def test_system(t, y):
    y1, y2 = y
    return np.array([
        -y2 + t**2 + 6*t + 1,
        y1 - 3*t**2 + 3*t + 1
    ])


def exact_solution(t):
    y1 = 3*t**2 - t - 1 + np.cos(t) + np.sin(t)
    y2 = t**2 + 2 - np.cos(t) + np.sin(t)
    return np.column_stack((y1, y2))


def test_rk4():
    print("\n===== ПРОВЕРКА RK4 =====")

    t, y = rk4(test_system, 0, 3, [0, 1], 0.1)
    exact = exact_solution(t)

    error = np.max(np.abs(y - exact))

    print("h = 0.1")
    print("Максимальная ошибка =", error)

    # Сравнение численного и точного решения
    plt.figure(figsize=(8, 5))
    plt.plot(t, y[:, 0], "o", markersize=3, label="RK4")
    plt.plot(t, exact[:, 0], "-", label="Точное решение")
    plt.xlabel("t")
    plt.ylabel("y1")
    plt.title("Проверка метода RK4")
    plt.grid()
    plt.legend()

# 2. Исследование погрешности
def error_research():
    print("\n===== ИССЛЕДОВАНИЕ ПОГРЕШНОСТИ =====")
    print("h\t\te\t\t\te/h^4")

    h_values = [0.3, 0.15, 0.075, 0.0375, 0.01875]

    errors = []

    for h in h_values:
        t, y = rk4(test_system, 0, 3, [0, 1], h)
        exact = exact_solution(t)

        e = np.max(np.abs(y - exact))
        errors.append(e)

        print(f"{h:<8g}\t{e:.6e}\t{e/h**4:.6e}")

    # График ошибки
    plt.figure(figsize=(8, 5))
    plt.loglog(h_values, errors, "o-", label="Ошибка RK4")

    C = errors[-1] / h_values[-1]**4
    plt.loglog(
        h_values,
        C * np.array(h_values)**4,
        "--",
        label="C*h^4"
    )

    plt.xlabel("h")
    plt.ylabel("e")
    plt.title("Погрешность метода RK4")
    plt.grid()
    plt.legend()

# 3. Осциллятор Ван-дер-Поля

def van_der_pol(t, y, eta):
    y1, y2 = y

    return np.array([
        y2 - y1 * (y1**2 / 3 - 1),
        -eta * y1
    ])


def solve_vdp(eta, T=100, h=0.01):
    def system(t, y):
        return van_der_pol(t, y, eta)

    return rk4(system, 0, T, [2, 2], h)

# 4. Стационарная точка

def stationary_points():
    print("\n===== СТАЦИОНАРНАЯ ТОЧКА =====")
    print("Стационарная точка: (0, 0)")
    print("lambda^2 - lambda + eta = 0\n")

    eta_values = [0.001, 0.01, 0.05, 0.1, 0.2]

    for eta in eta_values:
        D = 1 - 4*eta

        if D > 0:
            l1 = (1 + np.sqrt(D)) / 2
            l2 = (1 - np.sqrt(D)) / 2

            print(
                f"eta={eta:<5} "
                f"lambda1={l1:.4f}, "
                f"lambda2={l2:.4f} "
                "-> неустойчивый узел"
            )
        else:
            real = 0.5
            imag = np.sqrt(-D) / 2

            print(
                f"eta={eta:<5} "
                f"lambda={real:.3f} +/- {imag:.3f}i "
                "-> неустойчивый фокус"
            )

# 5. Решение при eta = 0.1

def plot_solution():
    eta = 0.1

    t, y = solve_vdp(eta)

    # y1(t)
    plt.figure(figsize=(8, 5))
    plt.plot(t, y[:, 0])
    plt.xlabel("t")
    plt.ylabel("y1")
    plt.title("y1(t), eta = 0.1")
    plt.grid()

    # y2(t)
    plt.figure(figsize=(8, 5))
    plt.plot(t, y[:, 1])
    plt.xlabel("t")
    plt.ylabel("y2")
    plt.title("y2(t), eta = 0.1")
    plt.grid()

    # Фазовая траектория
    plt.figure(figsize=(7, 6))
    plt.plot(y[:, 0], y[:, 1])
    plt.plot(2, 2, "o", label="Начальная точка")
    plt.plot(0, 0, "x", label="Стационарная точка")

    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("Фазовая траектория, eta = 0.1")
    plt.grid()
    plt.legend()

# 6. Исследование различных eta

def eta_research():
    eta_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    plt.figure(figsize=(9, 7))

    for eta in eta_values:
        # Для малых eta требуется больше времени
        T = max(300, 10 / eta)

        t, y = solve_vdp(eta, T=T, h=0.02)

        # Отбрасываем переходный процесс
        start = len(t) // 2

        plt.plot(
            y[start:, 0],
            y[start:, 1],
            label=f"eta={eta}"
        )

    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("Фазовые траектории при различных eta")
    plt.grid()
    plt.legend()

# Главная программа

def main():
    test_rk4()
    error_research()
    stationary_points()
    plot_solution()
    eta_research()

    plt.show()


if __name__ == "__main__":
    main()