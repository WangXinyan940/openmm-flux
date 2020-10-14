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
    [2.5, 3.0, 3.2],
    [2.7, 2.5, 3.0]
    ])
r = pos[0,:] - pos[1,:]
print(r)
ONE_4PI_EPS0 = 138.935456
alpha = np.sqrt(- np.log(2. * delta)) / 2.0
print("alpha: ", alpha)
def calcEself(position):
    gself = np.zeros(position.shape)
    dq = calcDQ(position)
    qcharge = q + dq

    dqdr = np.zeros((3, 3, 3)) # q, n, dim
    dist12 = np.sqrt(np.power(position[1,:] - position[2,:],2).sum())
    for n in range(3):
        dqdr[1,1,n] = 0.5 * (position[1,n] - position[2,n]) / dist12
        dqdr[1,2,n] = -0.5 * (position[1,n] - position[2,n]) / dist12
        dqdr[2,1,n] = 0.5 * (position[2,n] - position[1,n]) / dist12
        dqdr[2,2,n] = -0.5 * (position[2,n] - position[1,n]) / dist12
    Eself = 0.0
    for n in range(3):
        Eself += - alpha / np.sqrt(np.pi) * qcharge[n] ** 2 * ONE_4PI_EPS0
        for n2 in range(3):
            for dim in range(3):
                gself[n2,dim] += - alpha / np.sqrt(np.pi) * qcharge[n] * 2. * ONE_4PI_EPS0 * dqdr[n,n2,dim]
    return Eself, gself

Eself, gself = calcEself(pos)
print("Eself: ", Eself)

print("Grad:")
print(gself)

newpos = np.zeros(pos.shape)
numgres = np.zeros(pos.shape)
for n in range(3):
    for dim in range(3):
        #pos
        newpos[:,:] = pos[:,:]
        newpos[n,dim] = newpos[n,dim] + 0.0000001
        Epos, _ = calcEself(newpos)
        #neg
        newpos[:,:] = pos[:,:]
        newpos[n,dim] = newpos[n,dim] - 0.0000001
        Eneg, _ = calcEself(newpos)
        numgres[n,dim] = (Epos - Eneg) / 2. / 0.0000001
print(numgres)

newpos = np.zeros(pos.shape)
newpos[:,:] = pos[:,:]
newpos[2,2] = 3.2
E1, _ = calcEself(newpos)
newpos[2,2] = 3.0
E2, _ = calcEself(newpos)
print(E1, E2)