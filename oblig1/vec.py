# oppgave 4b) - call velfield

import matplotlib.pyplot as plt
from velfield import velfield

n = 20

x, y, vx, vy = velfield(n)

plt.figure()
plt.quiver(x, y, vx, vy)
plt.axis('equal')
# plt.savefig('4b.pdf')
plt.show()
