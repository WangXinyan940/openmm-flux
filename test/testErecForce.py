import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit

cell = [[6., 0., 0.], [0., 5., 0.], [0., 0., 5.]]
q = np.array([1.0, -0.5, -0.5])
delta = 0.00001

def calcDQ(position):
    dq = np.array([0., 0., 0.])
    dr = np.sqrt(np.power(position[1,:] - position[2,:],2).sum())
    dq[1] = 0.5 * (dr - 0.6633249580710802)
    dq[2] = - 0.5 * (dr - 0.6633249580710802)
    return dq

pos = np.array([
    [3.0, 3.0, 3.0],
    [2.5, 3.1, 3.2],
    [2.7, 2.5, 3.0]
    ])
r = pos[0,:] - pos[1,:]

ONE_4PI_EPS0 = 138.935456
alpha = np.sqrt(- np.log(2. * delta)) / 2.0
print("alpha: ", alpha)
Eself = - alpha / np.sqrt(np.pi) * (1.0 ** 2 + 0.5 ** 2 + 0.5 ** 2) * ONE_4PI_EPS0
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



def calcErec(position, printd=False):
    lowry = 0
    lowrz = 1
    Erec = 0.0
    V = np.zeros((2, 3))
    dq = calcDQ(position)
    qcharge = q + dq
    
    dqdr = np.zeros((3, 3, 3)) # q, n, dim
    dist12 = np.sqrt(np.power(position[1,:] - position[2,:],2).sum())
    for n in range(3):
        dqdr[1,1,n] = 0.5 * (position[1,n] - position[2,n]) / dist12
        dqdr[1,2,n] = -0.5 * (position[1,n] - position[2,n]) / dist12
        dqdr[2,1,n] = 0.5 * (position[2,n] - position[1,n]) / dist12
        dqdr[2,2,n] = -0.5 * (position[2,n] - position[1,n]) / dist12
    if printd:
        print("DqDr", dqdr)

    grec = np.zeros(position.shape)
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

                ssd = np.zeros((3,3))
                csd = np.zeros((3,3))

                for n in range(3):
                    gr = kx * position[n,0] + ky * position[n,1] + kz * position[n,2]
                    cs += qcharge[n] * np.cos(gr)
                    ss += qcharge[n] * np.sin(gr)
                    for n2 in range(3):
                        ssd[n2,:] += np.sin(gr) * dqdr[n,n2,:]
                        csd[n2,:] += np.cos(gr) * dqdr[n,n2,:]
                for n in range(3):
                    gr = kx * position[n,0] + ky * position[n,1] + kz * position[n,2]
                    gradr = 2. * const * eak * (ss * qcharge[n] * np.cos(gr) - cs * np.sin(gr) * qcharge[n])
                    grec[n,0] += gradr * kx
                    grec[n,1] += gradr * ky
                    grec[n,2] += gradr * kz

                grec += const * eak * 2. * (cs * csd + ss * ssd)
                #for n in range(3):
                #    for dim in range(3):
                #        grec[n,dim] += const * eak * 2. * cs * csd[n,dim]
                #        grec[n,dim] += const * eak * 2. * ss * ssd[n,dim]
                    
                Erec += const * eak * (cs * cs + ss * ss)

                lowrz = 1 - klist[2]
            lowry = 1- klist[1]
    return Erec, grec

Erec, grec = calcErec(pos)
print("Erec: ", Erec)

Ereal = 0.0
for i in range(3):
    for j in range(i+1,3):
        dist = np.sqrt(np.power(pos[i,:] - pos[j,:], 2).sum())
        Ereal += q[i] * q[j] * np.math.erfc(alpha * dist) / dist * ONE_4PI_EPS0
print("Ereal: ", Ereal)

print("Total: ", Eself + Erec + Ereal)

print("Grad:")
print(grec)

newpos = np.zeros(pos.shape)
numgres = np.zeros(pos.shape)
for n in range(3):
    for dim in range(3):
        #pos
        newpos[:,:] = pos[:,:]
        newpos[n,dim] = newpos[n,dim] + 0.00001
        Epos, _ = calcErec(newpos)
        #neg
        newpos[:,:] = pos[:,:]
        newpos[n,dim] = newpos[n,dim] - 0.00001
        Eneg, _ = calcErec(newpos)
        numgres[n,dim] = (Epos - Eneg) / 2. / 0.00001
print(numgres)