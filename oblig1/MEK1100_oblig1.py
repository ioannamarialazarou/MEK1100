# Oblig 1

import numpy as np
import matplotlib.pyplot as plt

# oppgave 1.c)

n = 0
N = 1000

t = np.linspace(0, 1, N); x = np.zeros(N); y = np.zeros(N)

v0 = 1.0
theta = [np.pi/6, np.pi/4, np.pi/3]

for n in range(3):
    for i in range(N):
        x[i] = t[i]
        y[i] = x[i]*np.tan(theta[n])*(1-x[i])

    plt.plot(x, y)

plt.xlabel('x'); plt.ylabel('y')
plt.legend(['pi/6', 'i/4', 'pi/3'])
# plt.savefig("1c.pdf")
plt.show()


# oppgave 2.b)

N = 1000

x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)

x, y = np.meshgrid(x, y)
z = np.log(np.abs(x)) - y

plt.contour(x, y, z)
# plt.savefig("2b.pdf")
plt.show()

# oppgave 3.b)

delta = 0.25
x, y = np.arange(-2, 2, delta), np.arange(-2, 2, delta)
vx, vy = np.meshgrid(np.cos(x) * np.sin(y), -np.sin(x) * np.cos(y))

plt.quiver(x, y, vx, vy)
plt.axis('equal')
# plt.savefig('3b.pdf')
plt.show()

# oppgave 4

#a)
from streamfun import streamfun

for n in(5, 30):
    (x, y, psi) = streamfun(n)
    plt.figure()
    plt.plot()
    plt.contour(x, y, psi)
    plt.title(f"n = {n}")
    plt.axis('equal')
#     plt.savefig(f"4a{n}.pdf")
    plt.show()


#b
def velfield(n):

    x = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
    [x, y] = np.meshgrid(x, x)
    vx = np.cos(x) * np.sin(y)
    vy = -np.sin(x) * np.cos(y)

    return x, y, vx, vy

n = 20

x, y, vx, vy = velfield(n)

plt.figure()
plt.quiver(x, y, vx, vy)
plt.axis('equal')
# plt.savefig('4b.pdf')
plt.show()

"""
(plots)
"""
