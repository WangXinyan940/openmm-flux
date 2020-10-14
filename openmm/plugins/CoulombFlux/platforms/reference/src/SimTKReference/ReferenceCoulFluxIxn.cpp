
/* Portions copyright (c) 2006-2018 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <string.h>
#include <sstream>
#include <complex>
#include <algorithm>
#include <iostream>
#include <map>

#include "SimTKOpenMMUtilities.h"
#include "ReferenceCoulFluxIxn.h"
#include "ReferenceForce.h"
#include "openmm/OpenMMException.h"

// In case we're using some primitive version of Visual Studio this will
// make sure that erf() and erfc() are defined.
#include "openmm/internal/MSVC_erfc.h"

using std::set;
using std::vector;
using std::map;
using namespace OpenMM;

/**---------------------------------------------------------------------------------------

   ReferenceCoulFluxIxn constructor

   --------------------------------------------------------------------------------------- */

ReferenceCoulFluxIxn::ReferenceCoulFluxIxn() : cutoff(false), periodic(false), ewald(false) {
}

/**---------------------------------------------------------------------------------------

   ReferenceCoulFluxIxn destructor

   --------------------------------------------------------------------------------------- */

ReferenceCoulFluxIxn::~ReferenceCoulFluxIxn() {
}

/**---------------------------------------------------------------------------------------

     Set the force to use a cutoff.

     @param distance            the cutoff distance
     @param neighbors           the neighbor list to use
     @param solventDielectric   the dielectric constant of the bulk solvent

     --------------------------------------------------------------------------------------- */

void ReferenceCoulFluxIxn::setUseCutoff(double distance, const OpenMM::NeighborList& neighbors) {

    cutoff = true;
    cutoffDistance = distance;
    neighborList = &neighbors;
}

/**---------------------------------------------------------------------------------------

     Set the force to use periodic boundary conditions.  This requires that a cutoff has
     also been set, and the smallest side of the periodic box is at least twice the cutoff
     distance.

     @param vectors    the vectors defining the periodic box

     --------------------------------------------------------------------------------------- */

void ReferenceCoulFluxIxn::setPeriodic(OpenMM::Vec3* vectors) {

    assert(cutoff);
    assert(vectors[0][0] >= 2.0*cutoffDistance);
    assert(vectors[1][1] >= 2.0*cutoffDistance);
    assert(vectors[2][2] >= 2.0*cutoffDistance);
    periodic = true;
    periodicBoxVectors[0] = vectors[0];
    periodicBoxVectors[1] = vectors[1];
    periodicBoxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

     Set the force to use Ewald summation.

     @param alpha  the Ewald separation parameter
     @param kmaxx  the largest wave vector in the x direction
     @param kmaxy  the largest wave vector in the y direction
     @param kmaxz  the largest wave vector in the z direction

     --------------------------------------------------------------------------------------- */

void ReferenceCoulFluxIxn::setUseEwald(double alpha, int kmaxx, int kmaxy, int kmaxz) {
    alphaEwald = alpha;
    numRx = kmaxx;
    numRy = kmaxy;
    numRz = kmaxz;
    ewald = true;
}

/**---------------------------------------------------------------------------------------

   Calculate Ewald ixn

   @param numberOfAtoms    number of atoms
   @param atomCoordinates  atom coordinates
   @param atomParameters   atom parameters                             atomParameters[atomIndex][paramterIndex]
   @param exclusions       atom exclusion indices
                           exclusions[atomIndex] contains the list of exclusions for that atom
   @param forces           force array (forces added)
   @param totalEnergy      total energy
   @param includeDirect      true if direct space interactions should be included
   @param includeReciprocal  true if reciprocal space interactions should be included

   --------------------------------------------------------------------------------------- */

void ReferenceCoulFluxIxn::calculateEwaldIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates,
                                              vector<double>& atomParameters, vector<map<int,Vec3>>& dqdx, vector<set<int> >& exclusions,
                                              vector<Vec3>& forces, double* totalEnergy, bool includeDirect, bool includeReciprocal) const {
    typedef std::complex<double> d_complex;

    static const double epsilon     =  1.0;

    int kmax                            = (ewald ? std::max(numRx, std::max(numRy,numRz)) : 0);
    double factorEwald              = -1 / (4*alphaEwald*alphaEwald);
    double SQRT_PI                  = sqrt(PI_M);
    double TWO_PI                   = 2.0 * PI_M;
    double recipCoeff               = ONE_4PI_EPS0*4*PI_M/(periodicBoxVectors[0][0] * periodicBoxVectors[1][1] * periodicBoxVectors[2][2]) /epsilon;

    double totalSelfEwaldEnergy     = 0.0;
    double realSpaceEwaldEnergy     = 0.0;
    double recipEnergy              = 0.0;
    double recipDispersionEnergy    = 0.0;
    double totalRecipEnergy         = 0.0;
    double vdwEnergy                = 0.0;

    // **************************************************************************************
    // SELF ENERGY
    // **************************************************************************************

    if (includeReciprocal) {
        for (int atomID = 0; atomID < numberOfAtoms; atomID++) {
            double selfEwaldEnergy       = ONE_4PI_EPS0*atomParameters[atomID]*atomParameters[atomID] * alphaEwald/SQRT_PI;
            totalSelfEwaldEnergy            -= selfEwaldEnergy;
            // self force
            for(map<int,Vec3>::iterator iter=dqdx[atomID].begin(); iter != dqdx[atomID].end(); iter++){
                int k = iter->first;
                Vec3 v = iter->second;
                forces[k][0] += 2.0 * ONE_4PI_EPS0*atomParameters[atomID] * alphaEwald/SQRT_PI * v[0];
                forces[k][1] += 2.0 * ONE_4PI_EPS0*atomParameters[atomID] * alphaEwald/SQRT_PI * v[1];
                forces[k][2] += 2.0 * ONE_4PI_EPS0*atomParameters[atomID] * alphaEwald/SQRT_PI * v[2];
            }
        }
    }
    if (totalEnergy) {
        *totalEnergy += totalSelfEwaldEnergy;
    }
    /*
    std::cout << "Print Fself:" << std::endl;
    for(int i = 0; i < forces.size(); i++){
        std::cout << forces[i][0] << " " << forces[i][1] << " " << forces[i][2] << std::endl;
    }*/
    // **************************************************************************************
    // RECIPROCAL SPACE EWALD ENERGY AND FORCES
    // **************************************************************************************
    // Ewald method
    if (ewald && includeReciprocal) {

        // setup reciprocal box

        double recipBoxSize[3] = { TWO_PI / periodicBoxVectors[0][0], TWO_PI / periodicBoxVectors[1][1], TWO_PI / periodicBoxVectors[2][2]};


        // setup K-vectors

#define EIR(x, y, z) eir[(x)*numberOfAtoms*3+(y)*3+z]
        vector<d_complex> eir(kmax*numberOfAtoms*3);
        vector<d_complex> tab_xy(numberOfAtoms);
        vector<d_complex> tab_xyz(numberOfAtoms);

        if (kmax < 1)
            throw OpenMMException("kmax for Ewald summation < 1");

        for (int i = 0; (i < numberOfAtoms); i++) {
            for (int m = 0; (m < 3); m++)
                EIR(0, i, m) = d_complex(1,0);

            for (int m=0; (m<3); m++)
                EIR(1, i, m) = d_complex(cos(atomCoordinates[i][m]*recipBoxSize[m]),
                                         sin(atomCoordinates[i][m]*recipBoxSize[m]));

            for (int j=2; (j<kmax); j++)
                for (int m=0; (m<3); m++)
                    EIR(j, i, m) = EIR(j-1, i, m) * EIR(1, i, m);
        }

        // calculate reciprocal space energy and forces
        //std::cout << "kmax:" << kmax << std::endl;
        int lowry = 0;
        int lowrz = 1;

        for (int rx = 0; rx < numRx; rx++) {

            double kx = rx * recipBoxSize[0];

            for (int ry = lowry; ry < numRy; ry++) {

                double ky = ry * recipBoxSize[1];

                if (ry >= 0) {
                    for (int n = 0; n < numberOfAtoms; n++)
                        tab_xy[n] = EIR(rx, n, 0) * EIR(ry, n, 1);
                }

                else {
                    for (int n = 0; n < numberOfAtoms; n++)
                        tab_xy[n]= EIR(rx, n, 0) * conj (EIR(-ry, n, 1));
                }

                for (int rz = lowrz; rz < numRz; rz++) {
                    //std::cout << rx << " " << ry << " "  << rz << std::endl;
                    if (rz >= 0) {
                        for (int n = 0; n < numberOfAtoms; n++)
                            tab_xyz[n] = tab_xy[n] * EIR(rz, n, 2);
                    }

                    else {
                        for (int n = 0; n < numberOfAtoms; n++)
                            tab_xyz[n] = tab_xy[n] * conj(EIR(-rz, n, 2));
                    }

                    double cs = 0.0f;
                    double ss = 0.0f;
                    vector<Vec3> csd, ssd;
                    for(int i = 0; i < numberOfAtoms; i++){
                        Vec3 tmp_v, tmp_v2;
                        csd.push_back(tmp_v);
                        ssd.push_back(tmp_v2);
                    }

                    for (int n = 0; n < numberOfAtoms; n++) {
                        for(map<int,Vec3>::iterator iter=dqdx[n].begin(); iter != dqdx[n].end(); iter++){
                            int k = iter->first;
                            Vec3 v = iter->second;
                            csd[k] += tab_xyz[n].real() * v;
                            ssd[k] += tab_xyz[n].imag() * v;
                        }
                        cs += atomParameters[n] * tab_xyz[n].real();
                        ss += atomParameters[n] * tab_xyz[n].imag();
                    }

                    double kz = rz * recipBoxSize[2];
                    double k2 = kx * kx + ky * ky + kz * kz;
                    double ak = exp(k2*factorEwald) / k2;

                    for (int n = 0; n < numberOfAtoms; n++) {
                        double force = ak * atomParameters[n] * (cs * tab_xyz[n].imag() - ss * tab_xyz[n].real());
                        forces[n][0] += 2 * recipCoeff * force * kx;
                        forces[n][1] += 2 * recipCoeff * force * ky;
                        forces[n][2] += 2 * recipCoeff * force * kz;
                        forces[n] -= 2.0 * recipCoeff * ak * (cs * csd[n] + ss * ssd[n]);
                    }

                    recipEnergy       = recipCoeff * ak * (cs * cs + ss * ss);
                    totalRecipEnergy += recipEnergy;

                    if (totalEnergy)
                        *totalEnergy += recipEnergy;

                    lowrz = 1 - numRz;
                }
                lowry = 1 - numRy;
            }
        }
    }
    /*
    std::cout << "Print Fself+Frec:" << std::endl;
    for(int i = 0; i < forces.size(); i++){
        std::cout << forces[i][0] << " " << forces[i][1] << " " << forces[i][2] << std::endl;
    }*/
    // **************************************************************************************
    // SHORT-RANGE ENERGY AND FORCES
    // **************************************************************************************

    if (!includeDirect)
        return;
    double totalRealSpaceEwaldEnergy = 0.0f;


    for (auto& pair : *neighborList) {
        int ii = pair.first;
        int jj = pair.second;

        double deltaR[2][ReferenceForce::LastDeltaRIndex];
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
        double r         = deltaR[0][ReferenceForce::RIndex];
        double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
        double alphaR = alphaEwald * r;


        double dEdR = ONE_4PI_EPS0 * atomParameters[ii] * atomParameters[jj] * inverseR * inverseR * inverseR;
        dEdR = dEdR * (erfc(alphaR) + 2 * alphaR * exp (- alphaR * alphaR) / SQRT_PI);

        // accumulate forces

        for (int kk = 0; kk < 3; kk++) {
            double force  = dEdR*deltaR[0][kk];
            forces[ii][kk]   += force;
            forces[jj][kk]   -= force;
        }
        for(map<int,Vec3>::iterator iter=dqdx[ii].begin(); iter != dqdx[ii].end(); iter++){
            int k = iter->first;
            Vec3 v = iter->second;
            forces[k] -= v * atomParameters[jj] * erfc(alphaR) * inverseR * ONE_4PI_EPS0;
        }
        for(map<int,Vec3>::iterator iter=dqdx[jj].begin(); iter != dqdx[jj].end(); iter++){
            int k = iter->first;
            Vec3 v = iter->second;
            forces[k] -= atomParameters[ii] * v * erfc(alphaR) * inverseR * ONE_4PI_EPS0;
        }

        // accumulate energies

        realSpaceEwaldEnergy        = ONE_4PI_EPS0*atomParameters[ii]*atomParameters[jj]*inverseR*erfc(alphaR);
        totalRealSpaceEwaldEnergy  += realSpaceEwaldEnergy;
    }
    if (totalEnergy)
        *totalEnergy += totalRealSpaceEwaldEnergy;
    // std::cout << "Edir:" << totalRealSpaceEwaldEnergy << std::endl;
    /*
    std::cout << "Print Fself+Frec+Fdir:" << std::endl;
    for(int i = 0; i < forces.size(); i++){
        std::cout << forces[i][0] << " " << forces[i][1] << " " << forces[i][2] << std::endl;
    }*/
    // Now subtract off the exclusions, since they were implicitly included in the reciprocal space sum.

    double totalExclusionEnergy = 0.0f;
    const double TWO_OVER_SQRT_PI = 2/sqrt(PI_M);
    for (int i = 0; i < numberOfAtoms; i++)
        for (int exclusion : exclusions[i]) {
            if (exclusion > i) {
                int ii = i;
                int jj = exclusion;
                // std::cout << "Exlu pair:" << ii << " " << jj << std::endl;
                double deltaR[2][ReferenceForce::LastDeltaRIndex];
                ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);
                double r         = deltaR[0][ReferenceForce::RIndex];
                double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
                double alphaR    = alphaEwald * r;
                if (erf(alphaR) > 1e-6) {
                    double dEdR = ONE_4PI_EPS0 * atomParameters[ii] * atomParameters[jj] * inverseR * inverseR * inverseR;
                    dEdR = dEdR * (erf(alphaR) - 2 * alphaR * exp (- alphaR * alphaR) / SQRT_PI);

                    // accumulate forces

                    for (int kk = 0; kk < 3; kk++) {
                        double force = dEdR*deltaR[0][kk];
                        forces[ii][kk] -= force;
                        forces[jj][kk] += force;
                    }
                    for(map<int,Vec3>::iterator iter=dqdx[ii].begin(); iter != dqdx[ii].end(); iter++){
                        int k = iter->first;
                        Vec3 v = iter->second;
                        forces[k] += ONE_4PI_EPS0*v*atomParameters[jj]*inverseR*erf(alphaR);
                    }
                    for(map<int,Vec3>::iterator iter=dqdx[jj].begin(); iter != dqdx[jj].end(); iter++){
                        int k = iter->first;
                        Vec3 v = iter->second;
                        forces[k] += ONE_4PI_EPS0*atomParameters[ii]*v*inverseR*erf(alphaR);
                    }

                    // accumulate energies
                    realSpaceEwaldEnergy = ONE_4PI_EPS0*atomParameters[ii]*atomParameters[jj]*inverseR*erf(alphaR);
                    // std::cout << "Real space E:" << realSpaceEwaldEnergy << std::endl;
                }
                else {
                    realSpaceEwaldEnergy = alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*atomParameters[ii]*atomParameters[jj];
                    for(map<int,Vec3>::iterator iter=dqdx[ii].begin(); iter != dqdx[ii].end(); iter++){
                        int k = iter->first;
                        Vec3 v = iter->second;
                        forces[k] += alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*v*atomParameters[jj];
                    }
                    for(map<int,Vec3>::iterator iter=dqdx[jj].begin(); iter != dqdx[jj].end(); iter++){
                        int k = iter->first;
                        Vec3 v = iter->second;
                        forces[k] += alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*atomParameters[ii]*v;
                    }
                }

                totalExclusionEnergy += realSpaceEwaldEnergy;
            }
        }
    if (totalEnergy)
        *totalEnergy -= totalExclusionEnergy;
    // std::cout << "Eexclu:" << - totalExclusionEnergy << std::endl;
    /*
    std::cout << "Print Fself+Frec+Fdir+Fexlu:" << std::endl;
    for(int i = 0; i < forces.size(); i++){
        std::cout << forces[i][0] << " " << forces[i][1] << " " << forces[i][2] << std::endl;
    }*/
}


/**---------------------------------------------------------------------------------------

   Calculate LJ Coulomb pair ixn

   @param numberOfAtoms    number of atoms
   @param atomCoordinates  atom coordinates
   @param atomParameters   atom parameters                             atomParameters[atomIndex][paramterIndex]
   @param exclusions       atom exclusion indices
                           exclusions[atomIndex] contains the list of exclusions for that atom
   @param forces           force array (forces added)
   @param totalEnergy      total energy
   @param includeDirect      true if direct space interactions should be included
   @param includeReciprocal  true if reciprocal space interactions should be included

   --------------------------------------------------------------------------------------- */

void ReferenceCoulFluxIxn::calculatePairIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates,
                                             vector<double>& atomParameters, vector<vector<int>>& bondlist, vector<vector<double>>& bondparam, 
                                             vector<vector<int>>& anglelist, vector<vector<double>>& angleparam, vector<set<int> >& exclusions,
                                             vector<Vec3>& forces, double* totalEnergy, bool includeDirect, bool includeReciprocal) const {

    int i, j, dim;
    vector<map<int,Vec3>> dqdx;
    // initialize dqdx
    for(i=0;i<numberOfAtoms;i++){
        map<int,Vec3> tmp;
        dqdx.push_back(tmp);
    }
    // initialize charges
    vector<double> charges;
    for(i=0;i<numberOfAtoms;i++){
        charges.push_back(atomParameters[i]);
    }
    // calc dqdx based on bond & angle list
    for(i=0;i<bondlist.size();i++) {
        Vec3 rba = atomCoordinates[bondlist[i][0]] - atomCoordinates[bondlist[i][1]];
        double dist = sqrt(rba[0] * rba[0] + rba[1] * rba[1] + rba[2] * rba[2]);
        // do something
        double dq = bondparam[i][1] * (dist - bondparam[i][0]);
        charges[bondlist[i][0]] += dq;
        charges[bondlist[i][1]] -= dq;
        Vec3 da = bondparam[i][1] * rba / dist;
        if (dqdx[bondlist[i][0]].find(bondlist[i][0]) != dqdx[bondlist[i][0]].end()){
            dqdx[bondlist[i][0]][bondlist[i][0]] += da;
        } else {
            dqdx[bondlist[i][0]][bondlist[i][0]] = da;
        }
        if (dqdx[bondlist[i][0]].find(bondlist[i][1]) != dqdx[bondlist[i][0]].end()){
            dqdx[bondlist[i][0]][bondlist[i][1]] -= da;
        } else {
            dqdx[bondlist[i][0]][bondlist[i][1]] = -da;
        }
        if (dqdx[bondlist[i][1]].find(bondlist[i][0]) != dqdx[bondlist[i][1]].end()){
            dqdx[bondlist[i][1]][bondlist[i][0]] -= da;
        } else {
            dqdx[bondlist[i][1]][bondlist[i][0]] = -da;
        }
        if (dqdx[bondlist[i][1]].find(bondlist[i][1]) != dqdx[bondlist[i][1]].end()){
            dqdx[bondlist[i][1]][bondlist[i][1]] += da;
        } else {
            dqdx[bondlist[i][1]][bondlist[i][1]] = da;
        }
    }
    for(i=0;i<anglelist.size();i++) {
        Vec3 rba = atomCoordinates[anglelist[i][0]] - atomCoordinates[anglelist[i][1]];
        Vec3 rbc = atomCoordinates[anglelist[i][2]] - atomCoordinates[anglelist[i][1]];
        double dista = sqrt(rba[0] * rba[0] + rba[1] * rba[1] + rba[2] * rba[2]);
        double distc = sqrt(rbc[0] * rbc[0] + rbc[1] * rbc[1] + rbc[2] * rbc[2]);
        double one_ac = 1.0 / dista / distc;
        double one_ac3 = one_ac / distc / distc;
        double one_a3c = one_ac / dista / dista;
        double dot = rba[0] * rbc[0] + rba[1] * rbc[1] + rba[2] * rbc[2];
        double theta = acos(dot / dista / distc);
        double cnst = 1.0 / sqrt(1.0 - dot * dot * one_ac * one_ac);
        //do something
        double dqa = angleparam[i][1] * (theta - angleparam[i][0]);
        double dqc = angleparam[i][2] * (theta - angleparam[i][0]);
        charges[anglelist[i][0]] += dqa;
        charges[anglelist[i][2]] += dqc;
        charges[anglelist[i][1]] -= dqa + dqc;

        // calc dqdx
        Vec3 one_da = - cnst * (rbc * one_ac - rba * dot * one_a3c);
        Vec3 one_db = - cnst * (dot * one_ac3 * rbc + dot * one_a3c * rba - one_ac * (rba + rbc));
        Vec3 one_dc = - cnst * (rba * one_ac - rbc * dot * one_ac3);

        Vec3 dqa_da = angleparam[i][1] * one_da;
        Vec3 dqa_db = angleparam[i][1] * one_db;
        Vec3 dqa_dc = angleparam[i][1] * one_dc;
        Vec3 dqc_da = angleparam[i][2] * one_da;
        Vec3 dqc_db = angleparam[i][2] * one_db;
        Vec3 dqc_dc = angleparam[i][2] * one_dc;
        Vec3 dqb_da = - dqa_da - dqc_da;
        Vec3 dqb_db = - dqa_db - dqc_db;
        Vec3 dqb_dc = - dqa_dc - dqc_dc;

        if (dqdx[anglelist[i][0]].find(anglelist[i][0]) != dqdx[anglelist[i][0]].end()){
            dqdx[anglelist[i][0]][anglelist[i][0]] += dqa_da;
        } else {
            dqdx[anglelist[i][0]][anglelist[i][0]] = dqa_da;
        }
        if (dqdx[anglelist[i][0]].find(anglelist[i][1]) != dqdx[anglelist[i][0]].end()){
            dqdx[anglelist[i][0]][anglelist[i][1]] += dqa_db;
        } else {
            dqdx[anglelist[i][0]][anglelist[i][1]] = dqa_db;
        }
        if (dqdx[anglelist[i][0]].find(anglelist[i][2]) != dqdx[anglelist[i][0]].end()){
            dqdx[anglelist[i][0]][anglelist[i][2]] += dqa_dc;
        } else {
            dqdx[anglelist[i][0]][anglelist[i][2]] = dqa_dc;
        }

        if (dqdx[anglelist[i][1]].find(anglelist[i][0]) != dqdx[anglelist[i][1]].end()){
            dqdx[anglelist[i][1]][anglelist[i][0]] += dqb_da;
        } else {
            dqdx[anglelist[i][1]][anglelist[i][0]] = dqb_da;
        }
        if (dqdx[anglelist[i][1]].find(anglelist[i][1]) != dqdx[anglelist[i][1]].end()){
            dqdx[anglelist[i][1]][anglelist[i][1]] += dqb_db;
        } else {
            dqdx[anglelist[i][1]][anglelist[i][1]] = dqb_db;
        }
        if (dqdx[anglelist[i][1]].find(anglelist[i][2]) != dqdx[anglelist[i][1]].end()){
            dqdx[anglelist[i][1]][anglelist[i][2]] += dqb_dc;
        } else {
            dqdx[anglelist[i][1]][anglelist[i][2]] = dqb_dc;
        }

        if (dqdx[anglelist[i][2]].find(anglelist[i][0]) != dqdx[anglelist[i][2]].end()){
            dqdx[anglelist[i][2]][anglelist[i][0]] += dqc_da;
        } else {
            dqdx[anglelist[i][2]][anglelist[i][0]] = dqc_da;
        }
        if (dqdx[anglelist[i][2]].find(anglelist[i][1]) != dqdx[anglelist[i][2]].end()){
            dqdx[anglelist[i][2]][anglelist[i][1]] += dqc_db;
        } else {
            dqdx[anglelist[i][2]][anglelist[i][1]] = dqc_db;
        }
        if (dqdx[anglelist[i][2]].find(anglelist[i][2]) != dqdx[anglelist[i][2]].end()){
            dqdx[anglelist[i][2]][anglelist[i][2]] += dqc_dc;
        } else {
            dqdx[anglelist[i][2]][anglelist[i][2]] = dqc_dc;
        }
    }

    if (ewald) {
        calculateEwaldIxn(numberOfAtoms, atomCoordinates, charges, dqdx, exclusions, forces,
                          totalEnergy, includeDirect, includeReciprocal);
        return;
    }
    if (!includeDirect)
        return;
    for (int ii = 0; ii < numberOfAtoms; ii++) {
        // loop over atom pairs
        for (int jj = ii+1; jj < numberOfAtoms; jj++)
            if (exclusions[jj].find(ii) == exclusions[jj].end()){
                calculateOneIxn(ii, jj, atomCoordinates, charges, dqdx, forces, totalEnergy);
            }
                
    }
}

/**---------------------------------------------------------------------------------------

     Calculate LJ Coulomb pair ixn between two atoms

     @param ii               the index of the first atom
     @param jj               the index of the second atom
     @param atomCoordinates  atom coordinates
     @param atomParameters   atom parameters (charges, c6, c12, ...)     atomParameters[atomIndex][paramterIndex]
     @param forces           force array (forces added)
     @param totalEnergy      total energy

     --------------------------------------------------------------------------------------- */

void ReferenceCoulFluxIxn::calculateOneIxn(int ii, int jj, vector<Vec3>& atomCoordinates,
                                            vector<double>& atomParameters, vector<map<int,Vec3>>& dqdx, vector<Vec3>& forces,
                                            double* totalEnergy) const {
    double deltaR[2][ReferenceForce::LastDeltaRIndex];

    double energy = 0.0;
    double dEdR = 0.0;

    // get deltaR, R2, and R between 2 atoms

    ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);
    double r2        = deltaR[0][ReferenceForce::R2Index];
    double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
    energy += ONE_4PI_EPS0*atomParameters[ii]*atomParameters[jj]*inverseR;
    dEdR += ONE_4PI_EPS0*atomParameters[ii]*atomParameters[jj]*inverseR;
    dEdR     *= inverseR*inverseR;
    // accumulate forces

    for (int kk = 0; kk < 3; kk++) {
        double force  = dEdR*deltaR[0][kk];
        forces[ii][kk]   += force;
        forces[jj][kk]   -= force;
    }
    for(map<int,Vec3>::iterator iter=dqdx[ii].begin(); iter != dqdx[ii].end(); iter++){
        int k = iter->first;
        Vec3 v = iter->second;
        // greal[k,dim] += dqdx[i,k,dim] * qcharge[j] * erfc / dist * ONE_4PI_EPS0
        forces[k] -= v * atomParameters[jj] * inverseR * ONE_4PI_EPS0;
    }
    for(map<int,Vec3>::iterator iter=dqdx[jj].begin(); iter != dqdx[jj].end(); iter++){
        int k = iter->first;
        Vec3 v = iter->second;
        // greal[k,dim] += qcharge[i] * dqdx[j,k,dim] * erfc / dist * ONE_4PI_EPS0
        forces[k] -= atomParameters[ii] * v * inverseR * ONE_4PI_EPS0;
    }
    // accumulate energies

    if (totalEnergy)
        *totalEnergy += energy;
}

