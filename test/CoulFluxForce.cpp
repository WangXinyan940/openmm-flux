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

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/CoulFluxForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/CoulFluxForceImpl.h"
#include <cmath>
#include <map>
#include <sstream>
#include <utility>

using namespace OpenMM;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

CoulFluxForce::CoulFluxForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0),
        ewaldErrorTol(5e-4), alpha(0.0), exceptionsUsePeriodic(false), recipForceGroup(-1) {
}

CoulFluxForce::NonbondedMethod CoulFluxForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void CoulFluxForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < 0 || method > 5)
        throw OpenMMException("CoulFluxForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

double CoulFluxForce::getCutoffDistance() const {
    return cutoffDistance;
}

void CoulFluxForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

double CoulFluxForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void CoulFluxForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

int CoulFluxForce::addParticle(double charge) {
    particles.push_back(ParticleInfo(charge));
    return particles.size()-1;
}

void CoulFluxForce::getParticleParameters(int index, double& charge) const {
    ASSERT_VALID_INDEX(index, particles);
    charge = particles[index].charge;
}

void CoulFluxForce::setParticleParameters(int index, double charge) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index].charge = charge;
}

void CoulFluxForce::setCoulFluxBond(int a, int b, double r0, double j){
    //vector<int> bondpair;
    //vector<double> param;
    //bondpair.push_back(a);
    //bondpair.push_back(b);
    //param.push_back(r0);
    //param.push_back(j);
    //bondList.push_back(bondpair);
    //bondParameters.push_back(param);
}

void CoulFluxForce::setCoulFluxAngle(int a, int b, int c, double theta0, double ja, double jc){
    //vector<int> anglepair;
    //vector<double> param;
    //anglepair.push_back(a);
    //anglepair.push_back(b);
    //anglepair.push_back(c);
    //param.push_back(theta0);
    //param.push_back(ja);
    //param.push_back(jc);
    //angleList.push_back(anglepair);
    //angleParameters.push_back(param);
}

void CoulFluxForce::getCoulFluxBond(int index, int& a, int& b, double& r0, double& j){
    //a = bondList[index][0];
    //b = bondList[index][1];
    //r0 = bondParameters[index][0];
    //j = bondParameters[index][1];
}

void CoulFluxForce::getCoulFluxAngle(int index, int& a, int& b, int& c, double& theta0, double& ja, double& jc){
    //a = angleList[index][0];
    //b = angleList[index][1];
    //c = angleList[index][2];
    //theta0 = angleParameters[index][0];
    //ja = angleParameters[index][1];
    //jc = angleParameters[index][2];
}

int CoulFluxForce::addException(int particle1, int particle2, double charge1, double charge2, double scale, bool replace) {
    map<pair<int, int>, int>::iterator iter = exceptionMap.find(pair<int, int>(particle1, particle2));
    int newIndex;
    if (iter == exceptionMap.end())
        iter = exceptionMap.find(pair<int, int>(particle2, particle1));
    if (iter != exceptionMap.end()) {
        if (!replace) {
            stringstream msg;
            msg << "CoulFluxForce: There is already an exception for particles ";
            msg << particle1;
            msg << " and ";
            msg << particle2;
            throw OpenMMException(msg.str());
        }
        exceptions[iter->second] = ExceptionInfo(particle1, particle2, charge1, charge2, scale);
        newIndex = iter->second;
        exceptionMap.erase(iter->first);
    }
    else {
        exceptions.push_back(ExceptionInfo(particle1, particle2, charge1, charge2, scale));
        newIndex = exceptions.size()-1;
    }
    exceptionMap[pair<int, int>(particle1, particle2)] = newIndex;
    return newIndex;
}
void CoulFluxForce::getExceptionParameters(int index, int& particle1, int& particle2, double& charge1, double& charge2, double& scale) const {
    ASSERT_VALID_INDEX(index, exceptions);
    particle1 = exceptions[index].particle1;
    particle2 = exceptions[index].particle2;
    charge1 = exceptions[index].charge1;
    charge2 = exceptions[index].charge2;
    scale = exceptions[index].scale;
}

void CoulFluxForce::setExceptionParameters(int index, int particle1, int particle2, double charge1, double charge2, double scale) {
    ASSERT_VALID_INDEX(index, exceptions);
    exceptions[index].particle1 = particle1;
    exceptions[index].particle2 = particle2;
    exceptions[index].charge1 = charge1;
    exceptions[index].charge2 = charge2;
    exceptions[index].scale = scale;
}

ForceImpl* CoulFluxForce::createImpl() const {
    return new CoulFluxForceImpl(*this);
}

void CoulFluxForce::createExceptionsFromBonds(const vector<pair<int, int> >& bonds, double coulomb14Scale) {
    for (auto& bond : bonds)
        if (bond.first < 0 || bond.second < 0 || bond.first >= particles.size() || bond.second >= particles.size())
            throw OpenMMException("createExceptionsFromBonds: Illegal particle index in list of bonds");

    // Find particles separated by 1, 2, or 3 bonds.

    vector<set<int> > exclusions(particles.size());
    vector<set<int> > bonded12(exclusions.size());
    for (auto& bond : bonds) {
        bonded12[bond.first].insert(bond.second);
        bonded12[bond.second].insert(bond.first);
    }
    for (int i = 0; i < (int) exclusions.size(); ++i)
        addExclusionsToSet(bonded12, exclusions[i], i, i, 2);

    // Find particles separated by 1 or 2 bonds and create the exceptions.

    for (int i = 0; i < (int) exclusions.size(); ++i) {
        set<int> bonded13;
        addExclusionsToSet(bonded12, bonded13, i, i, 1);
        for (int j : exclusions[i]) {
            if (j < i) {
                if (bonded13.find(j) == bonded13.end()) {
                    // This is a 1-4 interaction.

                    const ParticleInfo& particle1 = particles[j];
                    const ParticleInfo& particle2 = particles[i];
                    const double charge1 = particle1.charge;
                    const double charge2 = particle2.charge;
                    const double scale = coulomb14Scale;
                    addException(j, i, charge1, charge2, scale);
                }
                else {
                    // This interaction should be completely excluded.

                    addException(j, i, 0.0, 1.0, 0.0);
                }
            }
        }
    }
}

void CoulFluxForce::addExclusionsToSet(const vector<set<int> >& bonded12, set<int>& exclusions, int baseParticle, int fromParticle, int currentLevel) const {
    for (int i : bonded12[fromParticle]) {
        if (i != baseParticle)
            exclusions.insert(i);
        if (currentLevel > 0)
            addExclusionsToSet(bonded12, exclusions, baseParticle, i, currentLevel-1);
    }
}

int CoulFluxForce::addGlobalParameter(const string& name, double defaultValue) {
    globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
    return globalParameters.size()-1;
}

const string& CoulFluxForce::getGlobalParameterName(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].name;
}

void CoulFluxForce::setGlobalParameterName(int index, const string& name) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].name = name;
}

double CoulFluxForce::getGlobalParameterDefaultValue(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].defaultValue;
}

void CoulFluxForce::setGlobalParameterDefaultValue(int index, double defaultValue) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].defaultValue = defaultValue;
}

int CoulFluxForce::getGlobalParameterIndex(const std::string& parameter) const {
    for (int i = 0; i < globalParameters.size(); i++)
        if (globalParameters[i].name == parameter)
            return i;
    throw OpenMMException("There is no global parameter called '"+parameter+"'");
}

int CoulFluxForce::getReciprocalSpaceForceGroup() const {
    return recipForceGroup;
}

void CoulFluxForce::setReciprocalSpaceForceGroup(int group) {
    if (group < -1 || group > 31)
        throw OpenMMException("Force group must be between -1 and 31");
    recipForceGroup = group;
}

void CoulFluxForce::updateParametersInContext(Context& context) {
    dynamic_cast<CoulFluxForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

bool CoulFluxForce::getExceptionsUsePeriodicBoundaryConditions() const {
    return exceptionsUsePeriodic;
}

void CoulFluxForce::setExceptionsUsePeriodicBoundaryConditions(bool periodic) {
    exceptionsUsePeriodic = periodic;
}
