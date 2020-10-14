import numpy as np 

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


if __name__ == "__main__":
    topo = Topology()
    #topo.addBond(0, 1, 0.1, 0.5)
    #topo.addBond(0, 2, 0.1, 0.6)
    topo.addAngle(1, 0, 2, 108.5 / 180. * np.pi, 0.3, 0.3)

    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.1]
    ])

    dQ, dQdx = topo.calcdQdx(pos)
    dqdx_out = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]))
    for i in range(dqdx_out.shape[0]):
        for j in range(dqdx_out.shape[1]):
            if i in dQdx and j in dQdx[i]:
                dqdx_out[i,j,:] = dQdx[i][j]
    print("Analytical:")
    print(dqdx_out[:,0,:])

    print("===========")
    print("Numerical:")
    dqdx_out2 = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]))
    newpos = np.zeros(pos.shape)
    for i in range(dqdx_out2.shape[1]):
        for j in range(dqdx_out2.shape[2]):
            newpos[:,:] = pos[:,:]
            newpos[i,j] += 0.00001
            dqpos, _ = topo.calcdQdx(newpos)
            newpos[:,:] = pos[:,:]
            newpos[i,j] -= 0.00001
            dqneg, _ = topo.calcdQdx(newpos)
            dqdx_out2[:,i,j] = (dqpos - dqneg) / 2. / 0.00001
    print(dqdx_out2[:,0,:])

    print("MSE:", np.power(dqdx_out2 - dqdx_out, 2).mean())