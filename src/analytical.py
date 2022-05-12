import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

'''
    { dx/dt = a - (b + 1)x + yx^2
    { dy/dt = bx - yx^2
'''

if __name__ == "__main__":
    x, y, a, b = sp.symbols("x y a b")
    x_eq = sp.Eq(a - (b + 1)*x + y * x**2, 0)
    y_eq = sp.Eq(b*x - y * x**2, 0)
    
    # Fixed Points
    print(f"Fixed Points: {sp.solve([x_eq, y_eq], [x, y])}")

    # Stability
    # Jacobian :
    #
    # F1(x)   =   dx/dt   =   a - (b + 1)x + yx^2 
    # F2(x)   =   dy/dt   =   bx - yx^2
    #
    #     | 𝛅F1/𝛅x     𝛅F1/𝛅y  |
    # J = |                   |
    #     | 𝛅F2/𝛅x     𝛅F2/𝛅y  |
    #
    # 𝛅F1/𝛅x = (a - (b + 1)x + yx^2)'x = 0 - (b + 1) + 2xy  =     2xy - b - 1 
    # 𝛅F1/𝛅y = (a - (b + 1)x + yx^2)'y = 0 - 0 + x^2        =     x^2
    # 𝛅F2/𝛅x = (bx - yx^2)'x                                =     b - 2xy
    # 𝛅F2/𝛅y = (bx - yx^2)'y = 0 - x^2                      =     -x^2 
    #
    #     | 2xy - b - 1      x^2 |
    # J = |                      |
    #     | b - 2xy         -x^2 |
    #
    # At fixed point (a, b/a):
    #     | b - 1      a^2  |
    # J = |                 |
    #     |  -b        -a^2 |
    #
    # Determinant Det(J):
    #     | b - 1      a^2  |
    # J = |                 | = (b - 1)(-a^2) - (-b)(a^2) = -ba^2 + a^2 + ba^2 = a^2
    #     |  -b        -a^2 |
    # Trace Tr(J):
    #     | b - 1      a^2  |
    # J = |                 | = (b - 1) + (-a^2) = -a^2 + b - 1
    #     |  -b        -a^2 |

    # Eigenvalues at Fixed Point (a, b/a)
    #
    #               | b - 1 - ƛ        a^2   |
    # det(J - ƛI) = |                        | = (b - 1 - ƛ)(-a^2 - ƛ) - (a^2)(-b) = -ba^2 + a^2 + ƛa^2 - bƛ + ƛ + ƛ^2 + ba^2
    #               |   -b          -a^2 - ƛ |                                     = a^2 + ƛa^2 - bƛ + ƛ + ƛ^2 =
    #                                                                              = ƛ^2 + ƛa^2 - bƛ + ƛ + a^2 =
    #                                                                              = ƛ^2 + (a^2 - b + 1)ƛ + a^2 = 
    #                                                                              = ƛ^2 - (-a^2 + b - 1)ƛ + a^2 =
    #                                                                              = ƛ^2 - Tr(J)ƛ + Det(J) =

    # Nullclives
    print(f"dx/dt NC: {sp.solve(x_eq, [x, y, a, b], dict=True)}")
    print(f"dy/dt NC: {sp.solve(y_eq, [x, y, a, b], dict=True)}")


    x_plot = np.linspace(-10, 10, 1000)
    a_plot, b_plot = 0.1, 20
    arrow_delta = 0.3
    plt.figure()
    plt.plot(x_plot, (-a_plot + x_plot*(b_plot + 1))/x_plot**2, linewidth=1, c="red", label=r"$y =\frac{x\cdot(b+1)-a}{x^{2}}$")
    plt.plot(x_plot, b_plot/x_plot, linewidth=1, c="blue", label=r"$y = \frac{b}{x}$")
    plt.plot([0] * len(x_plot), x_plot, linewidth=1, c="green", label="x = 0")
    plt.plot(a_plot, b_plot/a_plot, ".", c="black", label="Fixed Point")
    arr_len = 0.07
    for xi, yi in [[-0.5,0.5], [-0.8, -0.8], [-1, -1.5], [0.02, 0.25], [1,1.5], [0.75, 0.8], [1, -1]]:
        direction_x = (a_plot - (b_plot + 1)*xi + yi * xi**2) > 0
        direction_y = (b_plot*xi - yi * xi**2) > 0
        plt.arrow(xi, yi, -arr_len + 2 * arr_len * direction_x, -arr_len + 2 * arr_len * direction_y, width=0.03)

    plt.xlim([-2, 2])
    plt.ylim(plt.xlim())
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Nullclives for a={a_plot} and b={b_plot}")
    plt.show()
