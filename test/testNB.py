import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit

system = mm.System()
system.addParticle(1.0)
system.addParticle(1.0)
system.addParticle(1.0)

cell = [[6., 0., 0.], [0., 5., 0.], [0., 0., 5.]]
q = np.array([1.0, -0.4, -0.6])
delta = 0.00001

system.setDefaultPeriodicBoxVectors(*cell)

nb = mm.CoulFluxForce()
print(nb)
nb.setNonbondedMethod(mm.CoulFluxForce.Ewald)
nb.setCutoffDistance(2.0 * unit.nanometer)
nb.setEwaldErrorTolerance(delta)
nb.addParticle(q[0])
nb.addParticle(q[1])
nb.addParticle(q[2])
system.addForce(nb)

integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))

pos = np.array([
    [3.0, 3.0, 3.0],
    [2.5, 3.1, 3.2],
    [2.7, 2.5, 3.0]
    ])
r = pos[0,:] - pos[1,:]

context.setPositions(pos * unit.nanometer)
state = context.getState(getEnergy=True)
print("CoulFluxForce : ", state.getPotentialEnergy())


dist = np.sqrt(np.power(pos[0,:] - pos[1,:], 2).sum())
ONE_4PI_EPS0 = 138.935456
alpha = np.sqrt(- np.log(2. * delta)) / 2.0
print("alpha: ", alpha)
Eself = - alpha / np.sqrt(np.pi) * (1.0 ** 2 + 0.4 ** 2 + 0.6 ** 2) * ONE_4PI_EPS0
print("Eself: ", Eself)

klist = []
for axis in range(3):
    for k in range(1, 100):
        error = k * np.sqrt(cell[axis][axis] * alpha) / 20. * np.exp(- np.power(np.pi * k / cell[axis][axis] / alpha, 2))
        if error < delta and k % 2 > 0:
            klist.append(k)
            break
        #print("kmax:", k)

dx = 2. * np.pi / cell[0][0]
dy = 2. * np.pi / cell[1][1]
dz = 2. * np.pi / cell[2][2]
dlist = [dx, dy, dz]

lowry = 0
lowrz = 1

Erec = 0.0
V = np.zeros((2, 3))
const = 1. / cell[0][0] / cell[1][1] / cell[2][2] * 4. * np.pi * ONE_4PI_EPS0
for rx in range(klist[0]):
    kx = rx * dx
    kx2 = kx * kx
    for ry in range(lowry, klist[1]):
        ky = ry * dy
        ky2 = ky * ky
        for rz in range(lowrz, klist[2]):
            kz = rz * dz
            k2 = kx2 + ky2 + kz * kz
            eak = np.exp(- k2 / 4. / alpha / alpha) / k2
            cs, ss = 0.0, 0.0
            for n in range(3):
                gr = kx * pos[n,0] + ky * pos[n,1] + kz * pos[n,2]
                cs += q[n] * np.cos(gr)
                ss += q[n] * np.sin(gr)
            Erec += const * eak * (cs * cs + ss * ss)

            lowrz = 1 - klist[2]
        lowry = 1- klist[1]

print("Erec: ", Erec)

Ereal = 0.0
for i in range(3):
    for j in range(i+1,3):
        dist = np.sqrt(np.power(pos[i,:] - pos[j,:], 2).sum())
        Ereal += q[i] * q[j] * np.math.erfc(alpha * dist) / dist * ONE_4PI_EPS0
print("Ereal: ", Ereal)

print("Total: ", Eself + Erec + Ereal)
