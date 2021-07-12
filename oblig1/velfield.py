# oppgave 4.b)

import numpy as np

def velfield(n):
    x = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
    [x, y] = np.meshgrid(x, x)
    vx = np.cos(x) * np.sin(y)
    vy = -np.sin(x) * np.cos(y)

    return x, y, vx, vy
