/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CoulFluxReferenceKernels.h"
#include "ReferenceCoulFluxIxn.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/CoulFluxForceImpl.h"
#include "openmm/OpenMMException.h"
#include "SimTKOpenMMUtilities.h"
#include <cmath>
#include <iostream>
#include <limits>

using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}

static vector<Vec3>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->velocities;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

static Vec3& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->periodicBoxSize;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return data->periodicBoxVectors;
}

ReferenceCalcCoulFluxKernel::~ReferenceCalcCoulFluxKernel() {
    if (neighborList != NULL)
        delete neighborList;
}

void ReferenceCalcCoulFluxKernel::initialize(const System& system, const CoulFluxForce& force) {

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    numParticles = force.getNumParticles();
    exclusions.resize(numParticles);
    vector<int> nb14s;
    map<int, int> nb14Index;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double charge1, charge2, scale;
        force.getExceptionParameters(i, particle1, particle2, charge1, charge2, scale);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
        if (scale != 0.0) {
            nb14Index[i] = nb14s.size();
            nb14s.push_back(i);
        }
    }

    // Build the arrays.

    num14 = nb14s.size();
    bonded14IndexArray.resize(num14, vector<int>(2));
    bonded14ParamArray.resize(num14, vector<double>(3));
    particleParamArray.resize(numParticles);
    baseParticleParams.resize(numParticles);
    baseExceptionParams.resize(num14);
    for (int i = 0; i < numParticles; ++i)
       force.getParticleParameters(i, baseParticleParams[i][0]);
    this->exclusions = exclusions;
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(nb14s[i], particle1, particle2, baseExceptionParams[i][0], baseExceptionParams[i][1], baseExceptionParams[i][2]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
    }
    nonbondedMethod = CalcCoulFluxKernel::NonbondedMethod(force.getNonbondedMethod());
    nonbondedCutoff = force.getCutoffDistance();
    if (nonbondedMethod == NoCutoff) {
        neighborList = NULL;
    }
    else {
        neighborList = new NeighborList();
    }
    if (nonbondedMethod == Ewald) {
        double alpha;
        CoulFluxForceImpl::calcEwaldParameters(system, force, alpha, kmax[0], kmax[1], kmax[2]);
        ewaldAlpha = alpha;
    }
    if (nonbondedMethod == NoCutoff)
        exceptionsArePeriodic = false;
    else
        exceptionsArePeriodic = force.getExceptionsUsePeriodicBoundaryConditions();
}

double ReferenceCalcCoulFluxKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    computeParameters(context);
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = 0;
    ReferenceCoulFluxIxn clj;
    bool ewald  = (nonbondedMethod == Ewald);
    if (nonbondedMethod != NoCutoff) {
        computeNeighborListVoxelHash(*neighborList, numParticles, posData, exclusions, extractBoxVectors(context), ewald, nonbondedCutoff, 0.0);
        clj.setUseCutoff(nonbondedCutoff, *neighborList);
    }
    if (ewald) {
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*nonbondedCutoff;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        clj.setPeriodic(boxVectors);
    }
    if (ewald)
        clj.setUseEwald(ewaldAlpha, kmax[0], kmax[1], kmax[2]);
    clj.calculatePairIxn(numParticles, posData, particleParamArray, exclusions, forceData, includeEnergy ? &energy : NULL, includeDirect, includeReciprocal);
    /*
    if (includeDirect) {
        ReferenceCoulFluxBondForce refBondForce;
        ReferenceCoulFlux14 nonbonded14;
        if (exceptionsArePeriodic) {
            Vec3* boxVectors = extractBoxVectors(context);
            nonbonded14.setPeriodic(boxVectors);
        }
        refBondForce.calculateForce(num14, bonded14IndexArray, posData, bonded14ParamArray, forceData, includeEnergy ? &energy : NULL, nonbonded14);
    }*/
    return energy;
}

void ReferenceCalcCoulFluxKernel::copyParametersToContext(ContextImpl& context, const CoulFluxForce& force) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    vector<int> nb14s;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double charge1, charge2, scale;
        force.getExceptionParameters(i, particle1, particle2, charge1, charge2, scale);
        if (scale != 0.0)
            nb14s.push_back(i);
    }
    if (nb14s.size() != num14)
        throw OpenMMException("updateParametersInContext: The number of non-excluded exceptions has changed");

    // Record the values.

    for (int i = 0; i < numParticles; ++i)
        force.getParticleParameters(i, baseParticleParams[i][0]);
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(nb14s[i], particle1, particle2, baseExceptionParams[i][0], baseExceptionParams[i][1], baseExceptionParams[i][2]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
    }
}

void ReferenceCalcCoulFluxKernel::computeParameters(ContextImpl& context) {
    // Compute particle parameters.

    vector<double> charges(numParticles);
    for (int i = 0; i < numParticles; i++) {
        charges[i] = baseParticleParams[i][0];
    }
    for (int i = 0; i < numParticles; i++) {
        particleParamArray[i] = charges[i];
    }

    // Compute exception parameters.

    charge1s.resize(num14);
    charge2s.resize(num14);
    scales.resize(num14);
    for (int i = 0; i < num14; i++) {
        charge1s[i] = baseExceptionParams[i][0];
        charge2s[i] = baseExceptionParams[i][1];
        scales[i] = baseExceptionParams[i][2];
    }
    for (int i = 0; i < num14; i++) {
        bonded14ParamArray[i][0] = charge1s[i];
        bonded14ParamArray[i][1] = charge2s[i];
        bonded14ParamArray[i][2] = scales[i];
    }
}

