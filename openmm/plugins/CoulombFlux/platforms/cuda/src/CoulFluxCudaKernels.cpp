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

    // - Initialize parameters.
    //    - CudaArray dQdXidx;
    //    - CudaArray dQdXval;
    //    - CudaArray dEdQ;
    //    - CudaArray CosSin; 
    //    - CudaArray initCharge;
    //    - CudaArray realCharge;
    numParticles = force.getNumParticles();
    dQdXidx.initialize()

    // -- Update initCharge

    // -- Update exceptions
    numExceptions = force.getNumExceptions();
    vector<vector<int>> exclusions(numParticles);
    for (int i = 0; i < numExceptions; i++){
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
    cu.getNonbondedUtilities().addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusions, "", force.getForceGroup());
    cu.getNonbondedUtilities().setUsePadding(false);

    // -- Build some kernels: calc constants

    // -- Build some kernels: generate kernel files and load them

    cu.addForce(new ForceInfo(force));

}

double CudaCalcCoulFluxKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy){
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    // Run kernel for dQdX calculation

    // If NoCutoff
    if (...) {
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