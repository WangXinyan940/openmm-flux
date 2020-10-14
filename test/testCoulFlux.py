import numpy as np 
import simtk.openmm as mm
import simtk.unit as unit 
from testdQdX import Topology

ONE_4PI_EPS0 = 138.935456
pos = np.array([
    [0.01, 0.01, 0.11],
    [0.1, 0.3, 0.0],
    [0.0, 0.4, 0.2]
])

cell = np.array([
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0]
])

charges = [-1.2, 0.8, 0.4]

bondprms = [
    [0, 1, 0.1, 0.2],
    [0, 2, 0.1, 0.2]
]

angleprms = [
    [1, 0, 2, 0.5 * np.pi, 0.1, 0.1]
]

topo = Topology()
for i in bondprms:
    topo.addBond(i[0], i[1], i[2], i[3])
for j in angleprms:
    topo.addAngle(j[0], j[1], j[2], j[3], j[4], j[5])

def calcE_noPBC(crd):
    E = 0.0
    dq, _ = topo.calcdQdx(crd)
    qfin = charges + dq
    for i in range(crd.shape[0]):
        for j in range(i+1, crd.shape[0]):
            if i == 0 and j == 1:
                continue
            dist = np.sqrt(np.power(crd[i,:] - crd[j,:], 2).sum())
            E += qfin[i] * qfin[j] / dist * ONE_4PI_EPS0
    return E


print("Test NoPBC")
system = mm.System()
system.addParticle(1.0)
system.addParticle(1.0)
system.addParticle(1.0)
cf = mm.CoulFluxForce()
cf.setNonbondedMethod(cf.NoCutoff)
for i in bondprms:
    cf.setCoulFluxBond(i[0], i[1], i[2], i[3])
for j in angleprms:
    cf.setCoulFluxAngle(j[0], j[1], j[2], j[3], j[4], j[5])
for c in charges:
    cf.addParticle(c)
cf.addException(0, 1, 0.0, 0.0, 0.0)
system.addForce(cf)

integ = mm.VerletIntegrator(1e-10 * unit.femtosecond)
context = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
context.setPositions(pos * unit.nanometer)
state = context.getState(getEnergy=True, getForces=True)

print("OpenMM result")
print(state.getPotentialEnergy())

print("Analytical result")
print(calcE_noPBC(pos))

print("OpenMM result")
print(state.getForces(asNumpy=True))

print("Analytical result")
delta = 0.0001
newpos = np.zeros(pos.shape)
anal_force = np.zeros(pos.shape)
for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
        newpos[:,:] = pos[:,:]
        newpos[i,j] += delta
        epos = calcE_noPBC(newpos)
        newpos[:,:] = pos[:,:]
        newpos[i,j] -= delta
        eneg = calcE_noPBC(newpos)
        anal_force[i,j] = -(epos - eneg) / 2. / delta
print(anal_force)
print("Openmm Numerical Result")
delta = 0.0001
newpos = np.zeros(pos.shape)
anal_force = np.zeros(pos.shape)
for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
        newpos[:,:] = pos[:,:]
        newpos[i,j] += delta
        context.setPositions(newpos * unit.nanometer)
        epos = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        newpos[:,:] = pos[:,:]
        newpos[i,j] -= delta
        context.setPositions(newpos * unit.nanometer)
        eneg = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        anal_force[i,j] = -(epos - eneg) / 2. / delta
print(anal_force)

print("Test PBC")
system = mm.System()
system.addParticle(1.0)
system.addParticle(1.0)
system.addParticle(1.0)
system.setDefaultPeriodicBoxVectors(cell[0,:], cell[1,:], cell[2,:])
cf = mm.CoulFluxForce()
cf.setNonbondedMethod(cf.Ewald)
for i in bondprms:
    cf.setCoulFluxBond(i[0], i[1], i[2], i[3])
for j in angleprms:
    cf.setCoulFluxAngle(j[0], j[1], j[2], j[3], j[4], j[5])
for c in charges:
    cf.addParticle(c)
cf.setEwaldErrorTolerance(0.00001)
cf.addException(0, 1, 0.0, 0.0, 0.0)
system.addForce(cf)

integ = mm.VerletIntegrator(1e-10 * unit.femtosecond)
context = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
context.setPositions(pos * unit.nanometer)
state = context.getState(getEnergy=True, getForces=True)

print("OpenMM result")
print(state.getPotentialEnergy())


def calcEself(position):
    delta = 0.00001
    alpha = np.sqrt(- np.log(2. * delta)) / 1.0
    dq, _ = topo.calcdQdx(position)
    qcharge = charges + dq
    dist12 = np.sqrt(np.power(position[1,:] - position[2,:],2).sum())
    Eself = 0.0
    for n in range(3):
        Eself += - alpha / np.sqrt(np.pi) * qcharge[n] ** 2 * ONE_4PI_EPS0
    return Eself

def calcEdir(position):
    delta = 0.00001
    alpha = np.sqrt(- np.log(2. * delta)) / 1.0
    Ereal = 0.0
    dq, _ = topo.calcdQdx(position)
    qcharge = charges + dq
    
    dist12 = np.sqrt(np.power(position[1,:] - position[2,:],2).sum())

    for i in range(3):
        for j in range(i+1,3):
            if i==0 and j == 1:
                continue
            dist = np.sqrt(np.power(position[i,:] - position[j,:], 2).sum())
            distA = dist * alpha
            erfc = np.math.erfc(distA)
            Ereal += qcharge[i] * qcharge[j] * erfc / dist * ONE_4PI_EPS0   
    return Ereal

def calcEexlu(position, i, j):
    delta = 0.00001
    alpha = np.sqrt(- np.log(2. * delta)) / 1.0
    Ereal = 0.0
    dq, _ = topo.calcdQdx(position)
    qcharge = charges + dq
    
    dist12 = np.sqrt(np.power(position[1,:] - position[2,:],2).sum())

    dist = np.sqrt(np.power(position[i,:] - position[j,:], 2).sum())
    distA = dist * alpha
    erf = np.math.erf(distA)
    Ereal -= qcharge[i] * qcharge[j] * erf / dist * ONE_4PI_EPS0   
    return Ereal

def calcErec(position):
    delta = 0.00001
    alpha = np.sqrt(- np.log(2. * delta)) / 1.0

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
    dq, _ = topo.calcdQdx(position)
    qcharge = charges + dq
    dist12 = np.sqrt(np.power(position[1,:] - position[2,:],2).sum())
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
                    gr = kx * position[n,0] + ky * position[n,1] + kz * position[n,2]
                    cs += qcharge[n] * np.cos(gr)
                    ss += qcharge[n] * np.sin(gr)
                Erec += const * eak * (cs * cs + ss * ss)

                lowrz = 1 - klist[2]
            lowry = 1- klist[1]
    return Erec

def calcEPBC(position):
    Eself = calcEself(position)
    Edir = calcEdir(position)
    Erec = calcErec(position)
    Eexlu = calcEexlu(position, 0, 1)
    print("Eself:", Eself)
    print("Erec:", Erec)
    print("Edir:", Edir)
    print("Eexlu:", Eexlu)
    return Eself + Edir + Erec + Eexlu

print("Analytical result")
print(calcEPBC(pos))

print("OpenMM result")
print(state.getForces(asNumpy=True))

print("Openmm Numerical Result")
delta = 0.0001
newpos = np.zeros(pos.shape)
anal_force = np.zeros(pos.shape)
for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
        newpos[:,:] = pos[:,:]
        newpos[i,j] += delta
        context.setPositions(newpos * unit.nanometer)
        epos = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        newpos[:,:] = pos[:,:]
        newpos[i,j] -= delta
        context.setPositions(newpos * unit.nanometer)
        eneg = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        anal_force[i,j] = -(epos - eneg) / 2. / delta
print(anal_force)

print("Analytical Result")
delta = 0.0001
newpos = np.zeros(pos.shape)
anal_force = np.zeros(pos.shape)
for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
        newpos[:,:] = pos[:,:]
        newpos[i,j] += delta
        epos = calcEself(newpos) + calcErec(newpos) + calcEdir(newpos) + calcEexlu(newpos, 0, 1)
        newpos[:,:] = pos[:,:]
        newpos[i,j] -= delta
        eneg = calcEself(newpos) + calcErec(newpos) + calcEdir(newpos) + calcEexlu(newpos, 0, 1)
        anal_force[i,j] = -(epos - eneg) / 2. / delta
print(anal_force)