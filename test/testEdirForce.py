import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit

cell = [[6., 0., 0.], [0., 5., 0.], [0., 0., 5.]]
q = np.array([0.5, -1.0, +0.5])
delta = 0.00001

def calcDQ(position):
    dq = np.array([0., 0., 0.])
    dr = np.sqrt(np.power(position[1,:] - position[2,:],2).sum())
    dq[1] = 0.5 * (dr - 1.0392304845413265)
    dq[2] = - 0.5 * (dr - 1.0392304845413265)
    return dq

pos = np.array([
    [3.0, 3.0, 3.0],
    [2.5, 3.5, 3.2],
    [2.7, 2.5, 3.0]
    ])
r = pos[0,:] - pos[1,:]

ONE_4PI_EPS0 = 138.935456
alpha = np.sqrt(- np.log(2. * delta)) / 2.0
print("alpha: ", alpha)

def calcEreal(position, printd=False):
    Ereal = 0.0

    dq = calcDQ(position)
    qcharge = q + dq
    if printd:
        print(qcharge, dq)
    
    dqdx = np.zeros((3, 3, 3)) # q, n, dim
    dist12 = np.sqrt(np.power(position[1,:] - position[2,:],2).sum())
    for n in range(3):
        dqdx[1,1,n] = 0.5 * (position[1,n] - position[2,n]) / dist12
        dqdx[1,2,n] = -0.5 * (position[1,n] - position[2,n]) / dist12
        dqdx[2,1,n] = 0.5 * (position[2,n] - position[1,n]) / dist12
        dqdx[2,2,n] = -0.5 * (position[2,n] - position[1,n]) / dist12

    if printd:
        print("dqdx")
        print(dqdx)

    greal = np.zeros(position.shape)

    for i in range(3):
        for j in range(i+1,3):
            dist = np.sqrt(np.power(position[i,:] - position[j,:], 2).sum())
            distA = dist * alpha
            erfc = np.math.erfc(distA)
            Ereal += qcharge[i] * qcharge[j] * erfc / dist * ONE_4PI_EPS0
            for dim in range(3):
                greal[i,dim] += ONE_4PI_EPS0 * qcharge[i] * qcharge[j] * (- erfc / dist ** 2 - alpha * 2. * np.exp(- distA ** 2) / np.sqrt(np.pi) / dist) * (position[i,dim] - position[j,dim]) / dist
                greal[j,dim] += ONE_4PI_EPS0 * qcharge[i] * qcharge[j] * (- erfc / dist ** 2 - alpha * 2. * np.exp(- distA ** 2) / np.sqrt(np.pi) / dist) * (position[j,dim] - position[i,dim]) / dist
                for k in range(3):
                    greal[k,dim] += dqdx[i,k,dim] * qcharge[j] * erfc / dist * ONE_4PI_EPS0 
                    greal[k,dim] += qcharge[i] * dqdx[j,k,dim] * erfc / dist * ONE_4PI_EPS0
                    
    return Ereal, greal

Ereal, greal = calcEreal(pos, printd=True)
print("Ereal: ", Ereal)

print("Grad:")
print(greal)

newpos = np.zeros(pos.shape)
numgres = np.zeros(pos.shape)
for n in range(3):
    for dim in range(3):
        #pos
        newpos[:,:] = pos[:,:]
        newpos[n,dim] = newpos[n,dim] + 0.00000001
        Epos, _ = calcEreal(newpos)
        #neg
        newpos[:,:] = pos[:,:]
        newpos[n,dim] = newpos[n,dim] - 0.00000001
        Eneg, _ = calcEreal(newpos)
        numgres[n,dim] = (Epos - Eneg) / 2. / 0.00000001
print(numgres)