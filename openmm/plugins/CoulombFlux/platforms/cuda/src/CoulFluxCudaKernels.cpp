/* -------------------------------------------------------------------------- *
 *                               OpenMMAmoeba                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "CoulFluxCudaKernels.h"
#include "CudaCoulFluxKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "CudaBondedUtilities.h"
#include "CudaFFT3D.h"
#include "CudaForceInfo.h"
#include "CudaKernelSources.h"
#include "SimTKOpenMMRealType.h"
#include "jama_lu.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result, prefix) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

CudaCalcCoulFluxKernel::CudaCalcCoulFluxKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system) :
        CalcCoulFluxKernel(name, platform), cu(cu), system(system) {
}

CudaCalcCoulFluxKernel::~CudaCalcAmoebaMultipoleForceKernel() {
    cu.setAsCurrent();
}

void CudaCalcCoulFluxKernel::initialize(const System& system, const CoulFluxForce& force) {
    cu.setAsCurrent();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    useEwald = (force.NonbondedMethod == 1);
    numParticles = force.getNumParticles();

    // - Initialize parameters.
    //    - CudaArray dQdXidx;
    vector<set<int>> dQdXidxSet;
    for(int i = 0; i < numParticles; i++){
        set<int> tmp;
        dQdXidxSet.push_back(tmp);
    }
    for(int i = 0; i < force.getNumCoulFluxBond(); i++){
        int a, b; 
        double r0, j;
        force.getCoulFluxBond(i, a, b, r0, j);
        dQdXidxSet[a].insert(a);
        dQdXidxSet[a].insert(b);
        dQdXidxSet[b].insert(a);
        dQdXidxSet[b].insert(b);
    }

    for(int i = 0; i < force.getNumCoulFluxAngle(); i++){
        int a, b, c;
        double theta0, ja, jc;
        force.getCoulFluxAngle(i, a, b, c, theta0, ja, jc);
        dQdXidxSet[a].insert(a);
        dQdXidxSet[a].insert(b);
        dQdXidxSet[a].insert(c);
        dQdXidxSet[b].insert(a);
        dQdXidxSet[b].insert(b);
        dQdXidxSet[b].insert(c);
        dQdXidxSet[c].insert(a);
        dQdXidxSet[c].insert(b);
        dQdXidxSet[c].insert(c);
    }
    int maxnumdX = 0;
    for(int i=0; i < numParticles; i++){
        if (dQdXidxSet[i].size() > maxnumdX){
            maxnumdX = dQdXidxSet[i].size();
        }
    }

    vector<int> dQdXidxVec;
    for(int i = 0; i < numParticles*maxnumdX; i++){
        dQdXidxVec.push_back(0);
    }
    for(int i = 0; i < numParticles; i++){
        int numdX = 0;
        for(set<int>::iterator it=dQdXidxSet.begin() ;it!=dQdXidxSet.end();it++){
            dQdXidxVec[i * maxnumdX + numdX] = *it;
            numdX++;
        }
    }
    //    - CudaArray dQdXval;

    vector<float> dQdXvalVec;
    for(int i = 0; i < numParticles*maxnumdX; i++){
        dQdXvalVec.push_back(0.0f);
    }


    //    - CudaArray dEdQ;
    vector<float> dEdQVec;
    for(int i = 0; i < numParticles; i++){
        dEdQVec.push_back(0.0);
    }
    //    - CudaArray initCharge;
    if (cu.getUseDoublePrecision()){
        vector<double> initChargeVec;
        for(int i = 0; i < numParticles; i++){
            double chrg;
            force.getParticleParameters(i, chrg);
            initChargeVec.push_back(chrg); 
        }
    } else {
        vector<float> initChargeVec;
        for(int i = 0; i < numParticles; i++){
            double chrg;
            force.getParticleParameters(i, chrg);
            initChargeVec.push_back((float) chrg); 
        }
    }
    
    //    - CudaArray realCharge;
    if (cu.getUseDoublePrecision()){
        vector<double> realChargeVec;
    } else {
        vector<float> realChargeVec;
    }
    for(int i = 0; i < numParticles; i++){
        realChargeVec.push_back(0.0);
    }
    
    dQdXidx.initialize<int>(cu, numParticles*maxnumdX, "dQdXidx"); 
    dQdXidx.upload(dQdXidxVec);
    dQdXval.initialize(cu, numParticles*maxnumdX, elementSize, "dQdXval"); 
    dQdXval.upload(dQdXvalVec);
    dEdQ.initialize(cu, numParticles, elementSize, "dEdQ");
    dEdQ.upload(dEdQVec);
    initCharge.initialize(cu, numParticles, elementSize, "initCharge");
    initCharge.upload(initChargeVec);
    realCharge.initialize(cu, numParticles, elementSize, "realCharge");
    realCharge.upload(realChargeVec);

    // initialize bond parameters
    int numBond = force.getNumCoulFluxBond();
    bondidx.initialize<int2>(cu, numBond * 2, "bondidx");
    bondparam.initialize<float2>(cu, numBond * 2, "bondparam");
    int numAngle = force.getNumCoulFluxAngle();
    angleidx.initialize<int3>(cu, numAngle * 3, "angleidx");
    angleparam.initialize<float3>(cu, numAngle * 3, "angleparam");

    vector<int2> bondidxvec;
    vector<float2> bondparamvec;
    for(int i=0;i<numBond; i++){
        int pi, pj;
        double r0, j;
        force.getCoulFluxBond(i, pi, pj, r0, j);
        bondidxvec.push_back(make_int2(pi, pj));
        bondparamvec.push_back(make_float2((float) j, (float) r0));
    }
    vector<int3> angleidxvec;
    vector<float3> angleparamvec;
    for(int i=0;i<numAngle; i++){
        int pi, pj, pk;
        double theta0, ja, jc;
        force.getCoulFluxAngle(i, pi, pj, pk, theta0, ja, jc);
        angleidxvec.push_back(make_int3(pi, pj, pk));
        angleparamvec.push_back(make_float3((float) theta0, (float) ja, (float) jc));
    }
    bondidx.upload(bondidxvec);
    bondparam.upload(bondparamvec);
    angleidx.upload(angleidxvec);
    angleparam.upload(angleparamvec);


    // if use Ewald
    if (useEwald){
        // calc kmaxx, kmaxy, kmaxz
        // Work here
        CosSin.initialize(cu, 2*(2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1), elementSize, "CosSin");
        if (cu.getUseDoublePrecision()) {
            vector<double> CosSinVal;
            for (int i = 0; i < 2*(2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1); i++) {
                CosSinVal.push_back(0.0);
            }
            CosSin.upload(CosSinVal);
        } else {
            vector<float> CosSinVal;
            for (int i = 0; i < 2*(2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1); i++) {
                CosSinVal.push_back(0.0f);
            }
            CosSin.upload(CosSinVal);
        }
    }

    // -- Update exceptions
    numExceptions = force.getNumExceptions();
    vector<vector<int>> exclusions(numParticles);
    for (int i = 0; i < numExceptions; i++) {
        // (int index, int& particle1, int& particle2, double& charge1, double& charge2, double& scale)
        int p1, p2; 
        double c1,c2,scale;
        force.getExceptionParameters(i, p1, p2, c1, c2, scale);
        exclusions[p1].push_back(p2);
        exclusions[p2].push_back(p1);
    }
    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
        int x = atom1/CudaContext::TileSize;
        for (int atom2 : exclusions[atom1]) {
            int y = atom2/CudaContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }

    // -- initialize Neighborlist
    cu.getNonbondedUtilities().addInteraction(true, true, true, force.getCutoffDistance(), exclusions, "", force.getForceGroup());
    cu.getNonbondedUtilities().setUsePadding(false);

    // -- Build some kernels: calc constants
    map<string, string> defines;
    // Update num_atoms
    defines["NUM_ATOMS"] = cu.intToString(numParticles);
    defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
    defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
    int numExclusionTiles = tilesWithExclusions.size();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
    int numContexts = cu.getPlatformData().contexts.size();
    int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
    // Update kmaxx, kmaxy, kmaxz


    // -- Build some kernels: generate kernel files and load them

    cu.addForce(new ForceInfo(force));

}

double CudaCalcCoulFluxKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy){
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    // Run kernel for dQdX calculation

    // If NoCutoff
    if (not useEwald) {
        // Run NoCutoff version
    }
    // else
    else {
        // Run Ewald Sum version
    }
    return 0.0;
}

void CudaCalcCoulFluxKernel::copyParametersToContext(ContextImpl& context, const CoulFluxForce& force){
    cu.setAsCurrent();
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");
    cu.getPosq().download(cu.getPinnedBuffer());
    float4* posqf = (float4*) cu.getPinnedBuffer();
    double4* posqd = (double4*) cu.getPinnedBuffer();
    for(int i=0;i < force.getNumParticles(); i++){
        double charge;
        force.getParticleParameters(i, charge);
        if (cu.getUseDoublePrecision())
            posqd[i].w = charge;
        else
            posqf[i].w = (float) charge;
    }
    cu.getPosq().upload(cu.getPinnedBuffer());
    cu.invalidateMolecules();
}