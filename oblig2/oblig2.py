import numpy as np
import matplotlib.pyplot as plt

### a)
import scipy.io as sio
data = sio.loadmat('data.mat')
x = data.get('x')
y = data.get('y')
u = data.get('u')
v = data.get('v')
xit = data.get('xit')
yit = data.get('yit')
print(f"{x}\n")
print(y)

dat = [x,y,u,v,xit,yit]

#Sjekker informasjon om matrisene og vektorene
for j in range(6):
    value = []
    indx = ['x','y','u','v','xit','yit']
    for i in dat:
        value += [np.shape(i)]
        punkt = [(x * y) for x, y in value]
    if j < 4:
        print(f"Matrisen {indx[j]} har (x,y) = {value[j]}")
    elif j >= 4:
        print(f"Vektoren {indx[j]} har (x,y) = {value[j]}")
print(f"Tilsammen [x,y,u,v,xit,yit] har {punkt} punkter i xy-planet")
print()

#test funksjoner
#Sjekk at griddet i xy-planet er regulært med intervall 0.5 mm i begge retninger
def test_griddet(x):
    tol = 1E-14
    for i in x:
        for j in range(194-1):
            t = i[j+1]-i[j]
            success = abs(t - 0.5) < tol
            assert success

# Sjekk at y-koordinatene spenner ut hele diameteren til røret
def test_y_kord(ykord):
    tol = 1E-14
    ykord = []
    for i in y:
        ykord += [i[1]]
    for j in range(194):
        t = ykord[j+1]-ykord[j]
        success = abs(t - 0.5) < tol
        assert success

#kaller test funksjoner
test_griddet(x)
test_y_kord(y)

### b)
H = np.sqrt(u**2 + v**2) #farten for hastighetskomponentene i xy-planet
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.title('Luft')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(xit, yit, "k*")
plt.contourf(x, y, H, np.linspace(0, 500, 10), cmap=plt.cm.BuPu_r)
plt.colorbar()

plt.subplot(2, 1, 2)
plt.title('Vann')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(xit, yit, "k*")
plt.contourf(x, y, H, np.linspace(1000, 5000, 10), cmap=plt.cm.BuPu_r)
plt.colorbar()

plt.tight_layout()
# plt.savefig("b.pdf")
plt.show()

### c)
n=10 # vi plotter en pil per n=10 punkter
plt.figure(figsize=(8, 6))
plt.quiver(x[::n, ::n], y[::n, ::n], u[::n, ::n], v[::n, ::n], color='purple')

def rektangler(xi,yi,xj,yj):
    x1 = x[yi][xi]; x2 = x[yj][xj]
    y1 = y[yi][xi]; y2 = y[yj][xj]

    plt.plot([x1,x2],[y1,y1], color='red')
    plt.plot([x2,x1],[y2,y2], color='blue')
    plt.plot([x1,x1],[y1,y2], color='black')
    plt.plot([x2,x2],[y2,y1], color='green')

rektangler(35,160,70,170)
rektangler(35,85,70,100)
rektangler(35,50,70,60)

plt.plot(xit, yit, "k*") # skilleflaten
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vektor pilplott med hastigheten i xy-planet $u\hat{i} + v\hat{j}$.')
# plt.savefig('c.pdf')
plt.show()

### d)
# divegrens
dudx = np.gradient(u, axis=0)
dvdy = np.gradient(v, axis=1)
div = dudx + dvdy

plt.figure(figsize=(8, 6))
divergence = plt.contourf(x, y, div, cmap=plt.cm.BuPu_r)
plt.colorbar(divergence)

plt.plot(xit, yit, "k*") # skilleflaten
rektangler(35,160,70,170)
rektangler(35,85,70,100)
rektangler(35,50,70,60)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Konturplott av divergensen $u\hat{i} + v\hat{j}$')
# plt.savefig('d.pdf')
plt.show()

### e)
# virvling
k = np.gradient(v,axis=1) - np.gradient(u,axis=0)

plt.figure(figsize=(8, 6))
plt.contourf(x, y, k, cmap=plt.cm.BuPu_r)

strm = plt.streamplot(x, y, u, v, color=u, linewidth=2, cmap='Purples') # strømlinjer
plt.colorbar(strm.lines)

plt.plot(xit, yit, "k*") # skilleflaten
rektangler(35,160,70,170)
rektangler(35,85,70,100)
rektangler(35,50,70,60)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Konturplott av $\mathbf{k}$ virvlingskomponenten')
# plt.savefig('e.pdf')
plt.show()

### f)
# sirkulasjon

# kurveintegral
def kurveintegral(x1,y1,x2,y2):
    dt = 0.5
    side1 = sum(u[y1,x1:x2+1]*dt)
    side2 = sum(v[y1:y2+1,x2]*dt)
    side3 = -sum(u[y2,x1:x2+1]*dt)
    side4 = -sum(v[y1:y2+1,x1]*dt)
    sirkulasjon = side1 + side2 + side3 + side4
    return side1, side2, side3, side4, sirkulasjon

# flateintegral
def flateintegral(x1,y1,x2,y2):
    virvl = np.gradient(v,0.5,axis=1) - np.gradient(u,0.5,axis=0)
    S = np.sum(virvl[y1:y2+1,x1:x2+1])*0.25
    return S

print ('Rektangel 1')
print ('--------------------------------------------')
a,b,c,d,s = kurveintegral(34,159,69,169)
print("side       kurveintegral")
print(f"1              {a:.3f}")
print(f"2              {b:.3f}")
print(f"3              {c:.3f}")
print(f"4              {d:.3f}")
print (f"kurveintegral: {s:.3f}")
S = flateintegral(34,159,69,169)
print(f"flateintegral: {S:.3f}")
print(f"forskjell (kurve - flateintegral): {np.abs(s - S):.3f}")
print()

print ('Rektangel 2')
print ('--------------------------------------------')
a,b,c,d,s = kurveintegral(34,85,69,99)
print("side       kurveintegral")
print(f"1              {a:.3f}")
print(f"2              {b:.3f}")
print(f"3              {c:.3f}")
print(f"4              {d:.3f}")
print (f"kurveintegral: {s:.3f}")
S = flateintegral(34,85,69,99)
print(f"flateintegral: {S:.3f}")
print(f"forskjell (kurve - flateintegral): {np.abs(s - S):.3f}")
print()

print ('Rektangel 3')
print ('--------------------------------------------')
a,b,c,d,s =  kurveintegral(34,49,69,59)
print("side       kurveintegral")
print(f"1              {a:.3f}")
print(f"2              {b:.3f}")
print(f"3              {c:.3f}")
print(f"4              {d:.3f}")
print (f"kurveintegral: {s:.3f}")
S = flateintegral(34,49,69,59)
print(f"flateintegral: {S:.3f}")
print(f"forskjell (kurve - flateintegral): {np.abs(s - S):.3f}")
print()

### g)
# integrert flux
def flux_int(u,v,x1,y1,x2,y2):
    dt = 0.5
    I1= -sum(v[y1-1,x1-1:x2])*dt
    I2= sum(u[y1-1:y2,x2-1])*dt
    I3= sum(v[y2-1,x1-1:x2])*dt
    I4= -sum(u[y1-1:y2,x1-1])*dt
    return I1, I2, I3, I4, I1+I2+I3+I4

# gauss sats
def gauss(div,x1,y1,x2,y2):
    return sum(sum(div[y1-1:y2,x1-1:x2]))*0.25

print ('Rektangel 1')
print ('-------------------------')
I1,I2,I3,I4,flux1 = flux_int(u,v,35,160,70,170)
print("side       kurveintegral")
print(f"1              {I1:.3f}")
print(f"2              {I2:.3f}")
print(f"3              {I3:.3f}")
print(f"4              {I4:.3f}")
print (f"flux: {flux1:3f}")
gauss1 = gauss(div,35,160,70,170)
print(f"gauss: {gauss1:3f}")
print()

print ('Rektangel 2')
print ('-------------------------')
I1,I2,I3,I4,flux2 = flux_int(u,v,35,85,70,100)
print("side       kurveintegral")
print(f"1              {I1:.3f}")
print(f"2              {I2:.3f}")
print(f"3              {I3:.3f}")
print(f"4              {I4:.3f}")
print (f"flux: {flux2:3f}")
gauss2 = gauss(div,35,85,70,100)
print(f"gauss: {gauss2:3f}")
print()

print ('Rektangel 3')
print ('-------------------------')
I1,I2,I3,I4,flux3 = flux_int(u,v,35,50,70,60)
print("side       kurveintegral")
print(f"1              {I1:.3f}")
print(f"2              {I2:.3f}")
print(f"3              {I3:.3f}")
print(f"4              {I4:.3f}")
print (f"flux: {flux3:3f}")
gauss3 = gauss(div,35,50,70,60)
print(f"gauss: {gauss1:3f}")


"""
(plots)

[[ 0.   0.5  1.  ... 95.5 96.  96.5]
 [ 0.   0.5  1.  ... 95.5 96.  96.5]
 [ 0.   0.5  1.  ... 95.5 96.  96.5]
 ...
 [ 0.   0.5  1.  ... 95.5 96.  96.5]
 [ 0.   0.5  1.  ... 95.5 96.  96.5]
 [ 0.   0.5  1.  ... 95.5 96.  96.5]]

[[-50.  -50.  -50.  ... -50.  -50.  -50. ]
 [-49.5 -49.5 -49.5 ... -49.5 -49.5 -49.5]
 [-49.  -49.  -49.  ... -49.  -49.  -49. ]
 ...
 [ 49.   49.   49.  ...  49.   49.   49. ]
 [ 49.5  49.5  49.5 ...  49.5  49.5  49.5]
 [ 50.   50.   50.  ...  50.   50.   50. ]]
Matrisen x har (x,y) = (201, 194)
Matrisen y har (x,y) = (201, 194)
Matrisen u har (x,y) = (201, 194)
Matrisen v har (x,y) = (201, 194)
Vektoren xit har (x,y) = (1, 194)
Vektoren yit har (x,y) = (1, 194)
Tilsammen [x,y,u,v,xit,yit] har [38994, 38994, 38994, 38994, 194, 194] punkter i xy-planet

Rektangel 1
--------------------------------------------
side       kurveintegral
1              70100.524
2              266.274
3              -68332.856
4              661.573
kurveintegral: 2695.514
flateintegral: 2621.559
forskjell (kurve - flateintegral): 73.955

Rektangel 2
--------------------------------------------
side       kurveintegral
1              652.329
2              118.499
3              -61243.465
4              -163.303
kurveintegral: -60635.940
flateintegral: -61095.332
forskjell (kurve - flateintegral): 459.392

Rektangel 3
--------------------------------------------
side       kurveintegral
1              5133.348
2              207.910
3              -5410.040
4              78.303
kurveintegral: 9.521
flateintegral: -12.214
forskjell (kurve - flateintegral): 21.735

Rektangel 1
-------------------------
side       kurveintegral
1              1556.868
2              21664.567
3              -2059.677
4              -21056.906
flux: 104.852605
gauss: -599.927284

Rektangel 2
-------------------------
side       kurveintegral
1              -5187.564
2              14782.533
3              -4074.052
4              -11997.856
flux: -6476.939182
gauss: 30792.991312

Rektangel 3
-------------------------
side       kurveintegral
1              -195.570
2              1536.822
3              284.944
4              -1750.764
flux: -124.568666
gauss: -599.927284
"""
