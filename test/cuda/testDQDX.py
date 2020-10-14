import numpy as np 
import numba as nb 
from numba import cuda
import math
import os 

def readPosQ(filename):
    with open(filename, "r") as f:
        text = [[float(j) for j in i.strip().split()] for i in f]
    return np.array(text)

def readBondPrm(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            tmp = line.strip().split()
            if len(tmp) == 4:
                data.append([int(tmp[0]), int(tmp[1]), float(tmp[2]), float(tmp[3])])
    return data

def readAnglePrm(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            tmp = line.strip().split()
            if len(tmp) > 4:
                data.append([int(tmp[0]), int(tmp[1]), int(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5])])
    return data

def readCell(filename):
    with open(filename, "r") as f:
        text = [[float(j) for j in i.strip().split()] for i in f]
    return np.array(text)

ONE_4PI_EPS0 = 138.935456

posq = readPosQ("posq.txt")
position = posq[:,:3]
charges = posq[:,3]
NATOMS = position.shape[0]
bondp = readBondPrm("bondp.txt")
anglep = readAnglePrm("anglep.txt")
cell = readCell("cell.txt")

NUM_BOND = len(bondp)
NUM_ANGLE = len(anglep)
TOTAL_PRM = NUM_BOND + NUM_ANGLE

array1 = []
for i in range(NATOMS):
    array1.append([])
for i in bondp:
    if i[0] not in array1[i[0]]:
        array1[i[0]].append(i[0])
    if i[1] not in array1[i[1]]:
        array1[i[1]].append(i[1])
    if i[1] not in array1[i[0]]:
        array1[i[0]].append(i[1])
    if i[0] not in array1[i[1]]:
        array1[i[1]].append(i[0])
for i in anglep:
    if i[0] not in array1[i[0]]:
        array1[i[0]].append(i[0])
    if i[1] not in array1[i[1]]:
        array1[i[1]].append(i[1])
    if i[2] not in array1[i[2]]:
        array1[i[2]].append(i[2])
    if i[0] not in array1[i[1]]:
        array1[i[1]].append(i[0])
    if i[0] not in array1[i[2]]:
        array1[i[2]].append(i[0])
    if i[1] not in array1[i[0]]:
        array1[i[0]].append(i[1])
    if i[1] not in array1[i[2]]:
        array1[i[2]].append(i[1])
    if i[2] not in array1[i[0]]:
        array1[i[0]].append(i[2])
    if i[2] not in array1[i[1]]:
        array1[i[1]].append(i[2])
TOTAL_COUNT = sum([len(i) for i in array1])

class Topology:

    def __init__(self):
        self.bondlist = []
        self.anglelist = []
        self.bondprm = []
        self.angleprm = []

    def addBond(self, a, b, r0, j):
        self.bondlist.append((a, b))
        self.bondprm.append((r0,j))

    def addAngle(self, a, b, c, theta0, ja, jc):
        # angle a-b-c
        self.anglelist.append((a, b, c))
        self.angleprm.append((theta0, ja, jc))

    def calcdQdx(self, coord):
        dQ = np.zeros((coord.shape[0]))
        dQdx = {}

        for a, b in self.bondlist:
            for i in [a, b]:
                if i not in dQdx:
                    dQdx[i] = {}
                for j in [a, b]:
                    dQdx[i][j] = np.zeros((3,))
        
        for a, b, c in self.anglelist:
            for i in [a, b, c]:
                if i not in dQdx:
                    dQdx[i] = {}
                for j in [a, b, c]:
                    dQdx[i][j] = np.zeros((3,))

        for n, pair in enumerate(self.bondlist):
            rvec = coord[pair[0]] - coord[pair[1]]
            dq = self.bondprm[n][1] * (np.sqrt(np.power(rvec, 2).sum()) - self.bondprm[n][0])
            dQ[pair[0]] += dq
            dQ[pair[1]] -= dq
            dQdx[pair[0]][pair[0]] += self.bondprm[n][1] * rvec / np.sqrt(np.power(rvec, 2).sum())
            dQdx[pair[0]][pair[1]] += - self.bondprm[n][1] * rvec / np.sqrt(np.power(rvec, 2).sum())
            dQdx[pair[1]][pair[0]] += - self.bondprm[n][1] * rvec / np.sqrt(np.power(rvec, 2).sum())
            dQdx[pair[1]][pair[1]] += self.bondprm[n][1] * rvec / np.sqrt(np.power(rvec, 2).sum())

        for n, pair in enumerate(self.anglelist):
            rba = coord[pair[0]] - coord[pair[1]]
            rbc = coord[pair[2]] - coord[pair[1]]
            dba = np.sqrt(np.power(rba, 2).sum())
            dbc = np.sqrt(np.power(rbc, 2).sum())
            one_ab = 1. / dba / dbc
            one_a3b = one_ab / dba / dba
            one_ab3 = one_ab / dbc / dbc
            dot = rba[0] * rbc[0] + rba[1] * rbc[1] + rba[2] * rbc[2]
            theta = np.arccos(dot / dba / dbc)
            const = 1. / np.sqrt(1.0 - dot * dot * one_ab * one_ab)
            dqa = self.angleprm[n][1] * (theta - self.angleprm[n][0])
            dqc = self.angleprm[n][2] * (theta - self.angleprm[n][0])
            dQ[pair[0]] += dqa
            dQ[pair[2]] += dqc
            dQ[pair[1]] += - dqa - dqc
            
            one_da = - const * (rbc * one_ab - rba * dot * one_a3b)
            one_db = - const * (dot * one_ab3 * rbc + dot * one_a3b * rba - one_ab * (rba + rbc))
            one_dc = - const * (rba * one_ab - rbc * dot * one_ab3)

            dqa_da = self.angleprm[n][1] * one_da
            dqa_db = self.angleprm[n][1] * one_db
            dqa_dc = self.angleprm[n][1] * one_dc
            dqc_da = self.angleprm[n][2] * one_da
            dqc_db = self.angleprm[n][2] * one_db
            dqc_dc = self.angleprm[n][2] * one_dc
            dqb_da = - dqa_da - dqc_da
            dqb_db = - dqa_db - dqc_db
            dqb_dc = - dqa_dc - dqc_dc

            dQdx[pair[0]][pair[0]] += dqa_da
            dQdx[pair[0]][pair[1]] += dqa_db
            dQdx[pair[0]][pair[2]] += dqa_dc
            dQdx[pair[1]][pair[0]] += dqb_da
            dQdx[pair[1]][pair[1]] += dqb_db
            dQdx[pair[1]][pair[2]] += dqb_dc
            dQdx[pair[2]][pair[0]] += dqc_da
            dQdx[pair[2]][pair[1]] += dqc_db
            dQdx[pair[2]][pair[2]] += dqc_dc

        return dQ, dQdx

def calcBond(bondindex, bondparam, position, parray_1, parray_2, varray):
    for nb in range(len(bondindex)):
        i0, i1 = bondindex[nb]
        r0, j = bondparam[nb]
        dist = math.sqrt((position[i0][0] - position[i1][0]) ** 2 + (position[i0][1] - position[i1][1]) ** 2 + (position[i0][2] - position[i1][2]) ** 2)
        istart = parray_1[i0*2+0]
        iend = parray_1[i0*2+1]
        for i in range(istart, iend):
            p2 = parray_2[i]
            if p2 == i0:
                varray[i*3+0] += j * (position[i0][0] - position[i1][0]) / dist
                varray[i*3+1] += j * (position[i0][1] - position[i1][1]) / dist
                varray[i*3+2] += j * (position[i0][2] - position[i1][2]) / dist
            if p2 == i1:
                varray[i*3+0] += - j * (position[i0][0] - position[i1][0]) / dist
                varray[i*3+1] += - j * (position[i0][1] - position[i1][1]) / dist
                varray[i*3+2] += - j * (position[i0][2] - position[i1][2]) / dist
        istart = parray_1[i1*2+0]
        iend = parray_1[i1*2+1]
        for i in range(istart, iend):
            p2 = parray_2[i]
            if p2 == i0:
                varray[i*3+0] += - j * (position[i0][0] - position[i1][0]) / dist
                varray[i*3+1] += - j * (position[i0][1] - position[i1][1]) / dist
                varray[i*3+2] += - j * (position[i0][2] - position[i1][2]) / dist
            if p2 == i1:
                varray[i*3+0] += j * (position[i0][0] - position[i1][0]) / dist
                varray[i*3+1] += j * (position[i0][1] - position[i1][1]) / dist
                varray[i*3+2] += j * (position[i0][2] - position[i1][2]) / dist

def calcAngle(angleindex, angleparam, position, parray_1, parray_2, varray):
    for na in range(len(angleindex)):
        i0, i1, i2 = angleindex[na]
        theta0, ja, jb = angleparam[na]
        pos0, pos1, pos2 = position[i0], position[i1], position[i2]

        rba_0 = pos0[0] - pos1[0]
        rba_1 = pos0[1] - pos1[1]
        rba_2 = pos0[2] - pos1[2]
        rbc_0 = pos2[0] - pos1[0]
        rbc_1 = pos2[1] - pos1[1]
        rbc_2 = pos2[2] - pos1[2]
        dba = math.sqrt(rba_0 * rba_0 + rba_1 * rba_1 + rba_2 * rba_2)
        dbc = math.sqrt(rbc_0 * rbc_0 + rbc_1 * rbc_1 + rbc_2 * rbc_2)
        one_ab = 1.0 / dba / dbc
        one_a3b = one_ab / dba / dba
        one_ab3 = one_ab / dbc / dbc
        dot = rba_0 * rbc_0 + rba_1 * rbc_1 + rba_2 * rbc_2
        theta = math.acos(dot / dba / dbc)
        const = 1.0 / math.sqrt(1.0 - dot * dot * one_ab * one_ab)
        one_da_0 = - const * (rbc_0 * one_ab - rba_0 * dot * one_a3b)
        one_da_1 = - const * (rbc_1 * one_ab - rba_1 * dot * one_a3b)
        one_da_2 = - const * (rbc_2 * one_ab - rba_2 * dot * one_a3b)
        one_db_0 = - const * (dot * one_ab3 * rbc_0 + dot * one_a3b * rba_0 - one_ab * (rba_0 + rbc_0))
        one_db_1 = - const * (dot * one_ab3 * rbc_1 + dot * one_a3b * rba_1 - one_ab * (rba_1 + rbc_1))
        one_db_2 = - const * (dot * one_ab3 * rbc_2 + dot * one_a3b * rba_2 - one_ab * (rba_2 + rbc_2))
        one_dc_0 = - const * (rba_0 * one_ab - rbc_0 * dot * one_ab3)
        one_dc_1 = - const * (rba_1 * one_ab - rbc_1 * dot * one_ab3)
        one_dc_2 = - const * (rba_2 * one_ab - rbc_2 * dot * one_ab3)
        dqa_da_0 = ja * one_da_0
        dqa_da_1 = ja * one_da_1
        dqa_da_2 = ja * one_da_2
        dqa_db_0 = ja * one_db_0
        dqa_db_1 = ja * one_db_1
        dqa_db_2 = ja * one_db_2
        dqa_dc_0 = ja * one_dc_0
        dqa_dc_1 = ja * one_dc_1
        dqa_dc_2 = ja * one_dc_2
        dqc_da_0 = jb * one_da_0
        dqc_da_1 = jb * one_da_1
        dqc_da_2 = jb * one_da_2
        dqc_db_0 = jb * one_db_0
        dqc_db_1 = jb * one_db_1
        dqc_db_2 = jb * one_db_2
        dqc_dc_0 = jb * one_dc_0
        dqc_dc_1 = jb * one_dc_1
        dqc_dc_2 = jb * one_dc_2
        dqb_da_0 = - dqa_da_0 - dqc_da_0
        dqb_da_1 = - dqa_da_1 - dqc_da_1
        dqb_da_2 = - dqa_da_2 - dqc_da_2
        dqb_db_0 = - dqa_db_0 - dqc_db_0
        dqb_db_1 = - dqa_db_1 - dqc_db_1
        dqb_db_2 = - dqa_db_2 - dqc_db_2
        dqb_dc_0 = - dqa_dc_0 - dqc_dc_0
        dqb_dc_1 = - dqa_dc_1 - dqc_dc_1
        dqb_dc_2 = - dqa_dc_2 - dqc_dc_2

        istart, iend = parray_1[i0*2+0], parray_1[i0*2+1]
        for i in range(istart, iend):
            p2 = parray_2[i]
            if p2 == i0:
                varray[i*3+0] += dqa_da_0
                varray[i*3+1] += dqa_da_1
                varray[i*3+2] += dqa_da_2
            if p2 == i1:
                varray[i*3+0] += dqa_db_0
                varray[i*3+1] += dqa_db_1
                varray[i*3+2] += dqa_db_2
            if p2 == i2:
                varray[i*3+0] += dqa_dc_0
                varray[i*3+1] += dqa_dc_1
                varray[i*3+2] += dqa_dc_2

        istart, iend = parray_1[i1*2+0], parray_1[i1*2+1]
        for i in range(istart, iend):
            p2 = parray_2[i]
            if p2 == i0:
                varray[i*3+0] += dqb_da_0
                varray[i*3+1] += dqb_da_1
                varray[i*3+2] += dqb_da_2
            if p2 == i1:
                varray[i*3+0] += dqb_db_0
                varray[i*3+1] += dqb_db_1
                varray[i*3+2] += dqb_db_2
            if p2 == i2:
                varray[i*3+0] += dqb_dc_0
                varray[i*3+1] += dqb_dc_1
                varray[i*3+2] += dqb_dc_2

        istart, iend = parray_1[i2*2+0], parray_1[i2*2+1]
        for i in range(istart, iend):
            p2 = parray_2[i]
            if p2 == i0:
                varray[i*3+0] += dqc_da_0
                varray[i*3+1] += dqc_da_1
                varray[i*3+2] += dqc_da_2
            if p2 == i1:
                varray[i*3+0] += dqc_db_0
                varray[i*3+1] += dqc_db_1
                varray[i*3+2] += dqc_db_2
            if p2 == i2:
                varray[i*3+0] += dqc_dc_0
                varray[i*3+1] += dqc_dc_1
                varray[i*3+2] += dqc_dc_2

@cuda.jit
def calcDQ_gpu(bondindex, bondparam, angleindex, angleparam, position, parray_1, parray_2, varray):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    while idx < TOTAL_PRM:
        if idx < NUM_BOND:
            i0, i1 = bondindex[idx][0], bondindex[idx][1]
            r0, j = bondparam[idx][0], bondparam[idx][1]
            r_x = position[i0][0] - position[i1][0]
            r_y = position[i0][1] - position[i1][1]
            r_z = position[i0][2] - position[i1][2]
            dist = math.sqrt(r_x * r_x + r_y * r_y + r_z * r_z)
            one_dist = 1.0 / dist
            istart = parray_1[i0*2+0]
            iend = parray_1[i0*2+1]
            r_x = position[i0][0] - position[i1][0]
            r_y = position[i0][1] - position[i1][1]
            r_z = position[i0][2] - position[i1][2]
            for i in range(istart, iend):
                p2 = parray_2[i]
                if p2 == i0:
                    cuda.atomic.add(varray, i*3+0, j * r_x * one_dist)
                    cuda.atomic.add(varray, i*3+1, j * r_y * one_dist)
                    cuda.atomic.add(varray, i*3+2, j * r_z * one_dist)
                if p2 == i1:
                    cuda.atomic.add(varray, i*3+0, - j * r_x * one_dist)
                    cuda.atomic.add(varray, i*3+1, - j * r_y * one_dist)
                    cuda.atomic.add(varray, i*3+2, - j * r_z * one_dist)
            istart = parray_1[i1*2+0]
            iend = parray_1[i1*2+1]
            for i in range(istart, iend):
                p2 = parray_2[i]
                if p2 == i0:
                    cuda.atomic.add(varray, i*3+0, - j * r_x * one_dist)
                    cuda.atomic.add(varray, i*3+1, - j * r_y * one_dist)
                    cuda.atomic.add(varray, i*3+2, - j * r_z * one_dist)
                if p2 == i1:
                    cuda.atomic.add(varray, i*3+0, j * r_x * one_dist)
                    cuda.atomic.add(varray, i*3+1, j * r_y * one_dist)
                    cuda.atomic.add(varray, i*3+2, j * r_z * one_dist)
        
        else:
            i0, i1, i2 = angleindex[idx-NUM_BOND][0], angleindex[idx-NUM_BOND][1], angleindex[idx-NUM_BOND][2]
            theta0, ja, jb = angleparam[idx-NUM_BOND][0], angleparam[idx-NUM_BOND][1], angleparam[idx-NUM_BOND][2]

            rba_0 = position[i0][0] - position[i1][0]
            rba_1 = position[i0][1] - position[i1][1]
            rba_2 = position[i0][2] - position[i1][2]
            rbc_0 = position[i2][0] - position[i1][0]
            rbc_1 = position[i2][1] - position[i1][1]
            rbc_2 = position[i2][2] - position[i1][2]
            dba = math.sqrt(rba_0 * rba_0 + rba_1 * rba_1 + rba_2 * rba_2)
            dbc = math.sqrt(rbc_0 * rbc_0 + rbc_1 * rbc_1 + rbc_2 * rbc_2)
            one_ab = 1.0 / dba / dbc
            one_a3b = one_ab / dba / dba
            one_ab3 = one_ab / dbc / dbc
            dot = rba_0 * rbc_0 + rba_1 * rbc_1 + rba_2 * rbc_2
            theta = math.acos(dot / dba / dbc)
            const = 1.0 / math.sqrt(1.0 - dot * dot * one_ab * one_ab)
            one_da_0 = - const * (rbc_0 * one_ab - rba_0 * dot * one_a3b)
            one_da_1 = - const * (rbc_1 * one_ab - rba_1 * dot * one_a3b)
            one_da_2 = - const * (rbc_2 * one_ab - rba_2 * dot * one_a3b)
            one_db_0 = - const * (dot * one_ab3 * rbc_0 + dot * one_a3b * rba_0 - one_ab * (rba_0 + rbc_0))
            one_db_1 = - const * (dot * one_ab3 * rbc_1 + dot * one_a3b * rba_1 - one_ab * (rba_1 + rbc_1))
            one_db_2 = - const * (dot * one_ab3 * rbc_2 + dot * one_a3b * rba_2 - one_ab * (rba_2 + rbc_2))
            one_dc_0 = - const * (rba_0 * one_ab - rbc_0 * dot * one_ab3)
            one_dc_1 = - const * (rba_1 * one_ab - rbc_1 * dot * one_ab3)
            one_dc_2 = - const * (rba_2 * one_ab - rbc_2 * dot * one_ab3)
            dqa_da_0 = ja * one_da_0
            dqa_da_1 = ja * one_da_1
            dqa_da_2 = ja * one_da_2
            dqa_db_0 = ja * one_db_0
            dqa_db_1 = ja * one_db_1
            dqa_db_2 = ja * one_db_2
            dqa_dc_0 = ja * one_dc_0
            dqa_dc_1 = ja * one_dc_1
            dqa_dc_2 = ja * one_dc_2
            dqc_da_0 = jb * one_da_0
            dqc_da_1 = jb * one_da_1
            dqc_da_2 = jb * one_da_2
            dqc_db_0 = jb * one_db_0
            dqc_db_1 = jb * one_db_1
            dqc_db_2 = jb * one_db_2
            dqc_dc_0 = jb * one_dc_0
            dqc_dc_1 = jb * one_dc_1
            dqc_dc_2 = jb * one_dc_2
            dqb_da_0 = - dqa_da_0 - dqc_da_0
            dqb_da_1 = - dqa_da_1 - dqc_da_1
            dqb_da_2 = - dqa_da_2 - dqc_da_2
            dqb_db_0 = - dqa_db_0 - dqc_db_0
            dqb_db_1 = - dqa_db_1 - dqc_db_1
            dqb_db_2 = - dqa_db_2 - dqc_db_2
            dqb_dc_0 = - dqa_dc_0 - dqc_dc_0
            dqb_dc_1 = - dqa_dc_1 - dqc_dc_1
            dqb_dc_2 = - dqa_dc_2 - dqc_dc_2

            istart, iend = parray_1[i0*2+0], parray_1[i0*2+1]
            for i in range(istart, iend):
                p2 = parray_2[i]
                if p2 == i0:
                    cuda.atomic.add(varray, i*3+0, dqa_da_0)
                    cuda.atomic.add(varray, i*3+1, dqa_da_1)
                    cuda.atomic.add(varray, i*3+2, dqa_da_2)
                if p2 == i1:
                    cuda.atomic.add(varray, i*3+0, dqa_db_0)
                    cuda.atomic.add(varray, i*3+1, dqa_db_1)
                    cuda.atomic.add(varray, i*3+2, dqa_db_2)
                if p2 == i2:
                    cuda.atomic.add(varray, i*3+0, dqa_dc_0)
                    cuda.atomic.add(varray, i*3+1, dqa_dc_1)
                    cuda.atomic.add(varray, i*3+2, dqa_dc_2)

            istart, iend = parray_1[i1*2+0], parray_1[i1*2+1]
            for i in range(istart, iend):
                p2 = parray_2[i]
                if p2 == i0:
                    cuda.atomic.add(varray, i*3+0, dqb_da_0)
                    cuda.atomic.add(varray, i*3+1, dqb_da_1)
                    cuda.atomic.add(varray, i*3+2, dqb_da_2)
                if p2 == i1:
                    cuda.atomic.add(varray, i*3+0, dqb_db_0)
                    cuda.atomic.add(varray, i*3+1, dqb_db_1)
                    cuda.atomic.add(varray, i*3+2, dqb_db_2)
                if p2 == i2:
                    cuda.atomic.add(varray, i*3+0, dqb_dc_0)
                    cuda.atomic.add(varray, i*3+1, dqb_dc_1)
                    cuda.atomic.add(varray, i*3+2, dqb_dc_2)

            istart, iend = parray_1[i2*2+0], parray_1[i2*2+1]
            for i in range(istart, iend):
                p2 = parray_2[i]
                if p2 == i0:
                    cuda.atomic.add(varray, i*3+0, dqc_da_0)
                    cuda.atomic.add(varray, i*3+1, dqc_da_1)
                    cuda.atomic.add(varray, i*3+2, dqc_da_2)
                if p2 == i1:
                    cuda.atomic.add(varray, i*3+0, dqc_db_0)
                    cuda.atomic.add(varray, i*3+1, dqc_db_1)
                    cuda.atomic.add(varray, i*3+2, dqc_db_2)
                if p2 == i2:
                    cuda.atomic.add(varray, i*3+0, dqc_dc_0)
                    cuda.atomic.add(varray, i*3+1, dqc_dc_1)
                    cuda.atomic.add(varray, i*3+2, dqc_dc_2)
        idx += cuda.blockDim.x*cuda.gridDim.x


if __name__ == "__main__":
    topo = Topology()
    for i in bondp:
        topo.addBond(*i)
    for i in anglep:
        topo.addAngle(*i)
    dq, dqdx_tmp = topo.calcdQdx(position)

    for i1 in dqdx_tmp:
        for i2 in dqdx_tmp[i1]:
            print(i1, i2, dqdx_tmp[i1][i2])

    parray_1 = np.zeros((NATOMS*2), dtype=np.int8)
    parray_2 = np.zeros((TOTAL_COUNT), dtype=np.int8)
    varray = np.zeros((TOTAL_COUNT*3), dtype=np.float32)
    ncount = 0
    bondindex = np.array([[i[0], i[1]] for i in bondp], dtype=np.int8)
    bondparam = np.array([[i[2], i[3]] for i in bondp], dtype=np.float32)
    angleindex = np.array([[i[0], i[1], i[2]] for i in anglep], dtype=np.int8)
    angleparam = np.array([[i[3], i[4], i[5]] for i in anglep], dtype=np.float32)

    for n1,i in enumerate(array1):
        istart = n1 * 2 + 0
        iend = n1 * 2 + 1
        parray_1[istart] = ncount
        for n2,j in enumerate(i):
            parray_2[ncount] = j
            ncount += 1
        parray_1[iend] = ncount

    varray_topo = np.zeros((TOTAL_COUNT*3), dtype=np.float32)
    for i in range(NATOMS):
        istart, iend = parray_1[i*2], parray_1[i*2+1]
        for j in range(istart, iend):
            varray_topo[j*3:(j+1)*3] = dqdx_tmp[i][parray_2[j]]
    
    calcBond(bondindex, bondparam, position, parray_1, parray_2, varray)
    calcAngle(angleindex, angleparam, position, parray_1, parray_2, varray)
    print("====")
    for i in range(NATOMS):
        istart, iend = parray_1[i*2+0], parray_1[i*2+1]
        for j in range(istart, iend):
            print(i, parray_2[j], varray[3*j:3*j+3])
    
    bondindex_gpu = cuda.to_device(bondindex)
    bondparam_gpu = cuda.to_device(bondparam)
    angleindex_gpu = cuda.to_device(angleindex)
    angleparam_gpu = cuda.to_device(angleparam)
    position_tmp = np.zeros(position.shape, dtype=np.float32)
    position_tmp[:,:] = position[:,:]
    position_gpu = cuda.to_device(position_tmp)
    parray_1_gpu = cuda.to_device(parray_1)
    parray_2_gpu = cuda.to_device(parray_2)
    varray_gpu = cuda.to_device(np.zeros((TOTAL_COUNT*3), dtype=np.float32))
    calcDQ_gpu[1,512](bondindex_gpu, bondparam_gpu, angleindex_gpu, angleparam_gpu, position_gpu, parray_1_gpu, parray_2_gpu, varray_gpu)
    cuda.synchronize()
    varray_gpu_result = varray_gpu.copy_to_host()
    print("====")
    for i in range(NATOMS):
        istart, iend = parray_1[i*2+0], parray_1[i*2+1]
        for j in range(istart, iend):
            print(i, parray_2[j], varray_gpu_result[3*j:3*j+3])
    print()
    print("diff -- topo vs loopcpu:", np.abs(varray_topo - varray).max())
    print("diff -- topo vs loopgpu:", np.abs(varray_topo - varray_gpu_result).max())