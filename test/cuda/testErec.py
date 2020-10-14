import numpy as np 
import numba as nb 
from numba import cuda
import math
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

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
            if len(tmp) == 4:
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

delta = 0.0001
cutoff = 1.0
alpha = np.sqrt(- np.log(2. * delta)) / cutoff
klist = np.zeros((3,), dtype=np.int32)
for axis in range(3):
    for k in range(1, 100):
        error = k * np.sqrt(cell[axis][axis] * alpha) / 20. * np.exp(- np.power(np.pi * k / cell[axis][axis] / alpha, 2))
        if error < delta and k % 2 > 0:
            klist[axis] = k
            break
KMAX_X, KMAX_Y, KMAX_Z = klist
one_alpha2 = 1.0 / alpha / alpha

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

def calcErec(position, charges, dqdx, cell):
    lowry = 0
    lowrz = 1
    Erec = []
    dx = 2. * np.pi / cell[0][0]
    dy = 2. * np.pi / cell[1][1]
    dz = 2. * np.pi / cell[2][2]
    ksizex = 2 * KMAX_X - 1
    ksizey = 2 * KMAX_Y - 1
    ksizez = 2 * KMAX_Z - 1
    totalK = ksizex * ksizey * ksizez
    grec = np.zeros(position.shape)
    const = 1. / cell[0][0] / cell[1][1] / cell[2][2] * 4. * np.pi * ONE_4PI_EPS0

    index = 0
    energy = 0.0
    while index < (KMAX_X - 1) * ksizey * ksizez + (KMAX_Y - 1) * ksizez + KMAX_Z:
        index += 1
    while index < totalK:
        rx = index//(ksizey*ksizez)
        remainder = index - rx*ksizey*ksizez
        ry = remainder//ksizez
        rz = remainder - ry*ksizez - KMAX_Z + 1
        rx += -KMAX_X + 1
        ry += -KMAX_Y + 1
        kx = rx*dx
        ky = ry*dy
        kz = rz*dz
        kz = rz * dz
        k2 = kx * kx + ky * ky + kz * kz
        eak = math.exp(- k2 * 0.25 * one_alpha2) / k2
        cs, ss = 0.0, 0.0

        ssd = np.zeros((3,3))
        csd = np.zeros((3,3))

        for n in range(position.shape[0]):
            gr = kx * position[n,0] + ky * position[n,1] + kz * position[n,2]
            cs += charges[n] * math.cos(gr)
            ss += charges[n] * math.sin(gr)
            for n2 in range(position.shape[0]):
                for dim in range(3):
                    ssd[n2,dim] += math.sin(gr) * dqdx[n,n2,dim]
                    csd[n2,dim] += math.cos(gr) * dqdx[n,n2,dim]
        for n in range(position.shape[0]):
            gr = kx * position[n,0] + ky * position[n,1] + kz * position[n,2]
            gradr = 2. * const * eak * (ss * charges[n] * np.cos(gr) - cs * np.sin(gr) * charges[n])
            grec[n,0] += gradr * kx
            grec[n,1] += gradr * ky
            grec[n,2] += gradr * kz
        grec += const * eak * 2. * (cs * csd + ss * ssd)
        Erec.append(const * eak * (cs * cs + ss * ss))
        index += 1

    return Erec, grec

@nb.jit(nopython=True, parallel=True)
def iterErec_numba(position, charges, dqdx, cell, Erec, grec):
    lowry = 0
    lowrz = 1
    dx = 2. * np.pi / cell[0][0]
    dy = 2. * np.pi / cell[1][1]
    dz = 2. * np.pi / cell[2][2]
    const = 1. / cell[0][0] / cell[1][1] / cell[2][2] * 4. * np.pi * ONE_4PI_EPS0
    for rx in range(KMAX_X):
        kx = rx * dx
        kx2 = kx * kx
        for ry in range(lowry, KMAX_Y):
            ky = ry * dy
            ky2 = ky * ky
            for rz in range(lowrz, KMAX_Z):
                kz = rz * dz
                k2 = kx2 + ky2 + kz * kz
                eak = math.exp(- k2 * 0.25 * one_alpha2) / k2
                cs = 0.0
                ss = 0.0

                ssd = np.zeros((3,3))
                csd = np.zeros((3,3))

                for n in range(position.shape[0]):
                    gr = kx * position[n,0] + ky * position[n,1] + kz * position[n,2]
                    cs += charges[n] * math.cos(gr)
                    ss += charges[n] * math.sin(gr)
                    for n2 in range(position.shape[0]):
                        for dim in range(3):
                            ssd[n2,dim] += math.sin(gr) * dqdx[n,n2,dim]
                            csd[n2,dim] += math.cos(gr) * dqdx[n,n2,dim]
                for n in range(position.shape[0]):
                    gr = kx * position[n][0] + ky * position[n][1] + kz * position[n][2]
                    gradr = 2.0 * const * eak * (ss * charges[n] * math.cos(gr) - cs * math.sin(gr) * charges[n])
                    grec[n][0] += gradr * kx
                    grec[n][1] += gradr * ky
                    grec[n][2] += gradr * kz

                for n in range(position.shape[0]):
                    for dim in range(3):
                        grec[n][dim] += const * eak * 2. * cs * csd[n][dim]
                        grec[n][dim] += const * eak * 2. * ss * ssd[n][dim]
                Erec[0] += const * eak * (cs * cs + ss * ss)

                lowrz = 1 - KMAX_Z
            lowry = 1- KMAX_Y


def calcErec_numba(position, charges, dqdx, cell):
    lowry = 0
    lowrz = 1
    Erec = np.array([0.0])

    grec = np.zeros(position.shape)
    const = 1. / cell[0][0] / cell[1][1] / cell[2][2] * 4. * np.pi * ONE_4PI_EPS0

    iterErec_numba(position, charges, dqdx, cell, Erec, grec)
    return Erec, grec

@cuda.jit()
def calcErec_gpu2(posq_gpu, charges_gpu, cell_gpu, Erecbuffer_gpu, cossinSum_gpu, cossin_gpu):
    ksizex = 2 * KMAX_X - 1
    ksizey = 2 * KMAX_Y - 1
    ksizez = 2 * KMAX_Z - 1
    totalK = ksizex * ksizey * ksizez
    recBoxX = 2.0 * np.pi / cell_gpu[0]
    recBoxY = 2.0 * np.pi / cell_gpu[1]
    recBoxZ = 2.0 * np.pi / cell_gpu[2]
    const = 1. / (cell_gpu[0] * cell_gpu[1] * cell_gpu[2]) * 4. * np.pi * ONE_4PI_EPS0
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    energy = 0.0
    while index < (KMAX_X - 1) * ksizey * ksizez + (KMAX_Y - 1) * ksizez + KMAX_Z:
        index += cuda.blockDim.x * cuda.gridDim.x 
    while index < totalK:
        rx = index//(ksizey*ksizez)
        remainder = index - rx*ksizey*ksizez
        ry = remainder//ksizez
        rz = remainder - ry*ksizez - KMAX_Z + 1
        rx += -KMAX_X + 1
        ry += -KMAX_Y + 1
        kx = rx*recBoxX
        ky = ry*recBoxY
        kz = rz*recBoxZ
        cs, ss = 0.0, 0.0
        for atom in range(NATOMS):
            gr = kx * posq_gpu[atom*4+0] + ky * posq_gpu[atom*4+1] + kz * posq_gpu[atom*4+2]
            cs += charges_gpu[atom] * math.cos(gr)
            ss += charges_gpu[atom] * math.sin(gr)
            cossin_gpu[index*NATOMS*2+atom*2+0] = math.cos(gr)
            cossin_gpu[index*NATOMS*2+atom*2+1] = math.cos(gr)

        cossinSum_gpu[index*2+0] = cs
        cossinSum_gpu[index*2+1] = ss

        k2 = kx*kx + ky*ky + kz*kz
        eak = math.exp(- k2 * 0.25 * one_alpha2) / k2
        energy += const*eak*(cs * cs + ss * ss)
        index += cuda.blockDim.x*cuda.gridDim.x
    Erecbuffer_gpu[cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x] += energy


if __name__ == "__main__":
    topo = Topology()
    for i in bondp:
        topo.addBond(*i)
    for i in anglep:
        topo.addAngle(*i)
    dq, dqdx_tmp = topo.calcdQdx(position)
    dqdx = np.zeros((position.shape[0], position.shape[0], 3))
    for k in dqdx_tmp.keys():
        v = dqdx_tmp[k]
        for k2 in v.keys():
            dqdx[k, k2, :] += v[k2][:]

    import time
    
    e1, g1 = calcErec(position, charges+dq, dqdx, cell)
    print(sum(e1))
    
    e2, g2 = calcErec_numba(position, charges+dq, dqdx, cell)
    print(e2[0])

    posq_gpu = cuda.to_device(posq.ravel())
    charges_gpu = cuda.to_device(charges+dq)
    cell_gpu = cuda.to_device(np.array([cell[0,0], cell[1,1], cell[2,2]]))
    
    threads_per_block = 256
    NUM_KV = int((2*KMAX_X-1)*(2*KMAX_Y-1)*(2*KMAX_Z-1))
    blocks_per_grid = math.ceil(NUM_KV / threads_per_block)
    Erecbuffer_gpu = cuda.device_array(blocks_per_grid*threads_per_block, dtype=np.float32)
    cossinSum_gpu = cuda.device_array(NUM_KV*2)
    print("Block:", blocks_per_grid, "Thread:", threads_per_block)
    calcErec_gpu2[blocks_per_grid, threads_per_block](posq_gpu, charges_gpu, cell_gpu, Erecbuffer_gpu, cossinSum_gpu)
    cuda.synchronize()
    Erec_result = Erecbuffer_gpu.copy_to_host()
    print("Numba CUDA:", Erec_result.sum())
