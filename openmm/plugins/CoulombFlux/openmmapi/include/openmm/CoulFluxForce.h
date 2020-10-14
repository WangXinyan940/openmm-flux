#ifndef OPENMM_COULFLUXFORCE_H_
#define OPENMM_COULFLUXFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "internal/windowsExportCoulFlux.h"
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <iostream>

namespace OpenMM {

/**
 * This class implements nonbonded interactions between particles, including a Coulomb force to represent
 * electrostatics and a Lennard-Jones force to represent van der Waals interactions.  It optionally supports
 * periodic boundary conditions and cutoffs for long range interactions.  Lennard-Jones interactions are
 * calculated with the Lorentz-Berthelot combining rule: it uses the arithmetic mean of the sigmas and the
 * geometric mean of the epsilons for the two interacting particles.
 *
 * To use this class, create a CoulFlux object, then call addParticle() once for each particle in the
 * System to define its parameters.  The number of particles for which you define nonbonded parameters must
 * be exactly equal to the number of particles in the System, or else an exception will be thrown when you
 * try to create a Context.  After a particle has been added, you can modify its force field parameters
 * by calling setParticleParameters().  This will have no effect on Contexts that already exist unless you
 * call updateParametersInContext().
 *
 * CoulFlux also lets you specify "exceptions", particular pairs of particles whose interactions should be
 * computed based on different parameters than those defined for the individual particles.  This can be used to
 * completely exclude certain interactions from the force calculation, or to alter how they interact with each other.
 *
 * Many molecular force fields omit Coulomb and Lennard-Jones interactions between particles separated by one
 * or two bonds, while using modified parameters for those separated by three bonds (known as "1-4 interactions").
 * This class provides a convenience method for this case called createExceptionsFromBonds().  You pass to it
 * a list of bonds and the scale factors to use for 1-4 interactions.  It identifies all pairs of particles which
 * are separated by 1, 2, or 3 bonds, then automatically creates exceptions for them.
 *
 * When using a cutoff, by default Lennard-Jones interactions are sharply truncated at the cutoff distance.
 * Optionally you can instead use a switching function to make the interaction smoothly go to zero over a finite
 * distance range.  To enable this, call setUseSwitchingFunction().  You must also call setSwitchingDistance()
 * to specify the distance at which the interaction should begin to decrease.  The switching distance must be
 * less than the cutoff distance.
 *
 * Another optional feature of this class (enabled by default) is to add a contribution to the energy which approximates
 * the effect of all Lennard-Jones interactions beyond the cutoff in a periodic system.  When running a simulation
 * at constant pressure, this can improve the quality of the result.  Call setUseDispersionCorrection() to set whether
 * this should be used.
 * 
 * In some applications, it is useful to be able to inexpensively change the parameters of small groups of particles.
 * Usually this is done to interpolate between two sets of parameters.  For example, a titratable group might have
 * two states it can exist in, each described by a different set of parameters for the atoms that make up the
 * group.  You might then want to smoothly interpolate between the two states.  This is done by first calling
 * addGlobalParameter() to define a Context parameter, then addParticleParameterOffset() to create a "parameter offset"
 * that depends on the Context parameter.  Each offset defines the following:
 * 
 * <ul>
 * <li>A Context parameter used to interpolate between the states.</li>
 * <li>A single particle whose parameters are influenced by the Context parameter.</li>
 * <li>Three scale factors (chargeScale, sigmaScale, and epsilonScale) that specify how the Context parameter
 * affects the particle.</li>
 * </ul>
 * 
 * The "effective" parameters for a particle (those used to compute forces) are given by
 * 
 * <tt><pre>
 * charge = baseCharge + param*chargeScale
 * sigma = baseSigma + param*sigmaScale
 * epsilon = baseEpsilon + param*epsilonScale
 * </pre></tt>
 * 
 * where the "base" values are the ones specified by addParticle() and "oaram" is the current value
 * of the Context parameter.  A single Context parameter can apply offsets to multiple particles,
 * and multiple parameters can be used to apply offsets to the same particle.  Parameters can also be used
 * to modify exceptions in exactly the same way by calling addExceptionParameterOffset().
 */

class OPENMM_EXPORT_COULFLUX CoulFluxForce : public Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Periodic boundary conditions are used, and Ewald summation is used to compute the Coulomb interaction of each particle
         * with all periodic copies of every other particle.
         */
        Ewald = 1,
    };
    /**
     * Create a CoulFluxForce.
     */
    CoulFluxForce();
    /**
     * Get the number of particles for which force field parameters have been defined.
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Get the number of special interactions that should be calculated differently from other interactions.
     */
    int getNumExceptions() const {
        return exceptions.size();
    }
    /**
     * Get the number of global parameters that have been added.
     */
    int getNumGlobalParameters() const {
        return globalParameters.size();
    }
    /** 
     * Get the number of bond pair.
     */
    int getNumCoulFluxBond() const {
        return bondParameters.size();
    }
    /**
     * Get the number of angle pair.
     */
    int getNumCoulFluxAngle() const {
        return angleParameters.size();
    }
    /**
     * Set the bond pair of electron fluxing. The function is dq_a = j (r_ab - r0); dq_b = - dq_a
     * @param a        atom index a
     * @param b        atom index b
     * @param r0       eq bond length r0
     * @param j        pre factor j
     */
    void setCoulFluxBond(int a, int b, double r0, double j);
    /**
     * Set the angle pair of electron fluxing. The angle is angle a-b-c, and function is:
     * dq_a = j_a * (theta_abc - theta0); dq_c = j_c * (theta_abc - theta0); dq_b = - dq_a - dq_c
     * @param a        atom index a
     * @param b        atom index b
     * @param c        atom index c
     * @param theta0   bond angle eq
     * @param ja       pre factor for atom a
     * @param jc       pre factor for atom c
     */
    void setCoulFluxAngle(int a, int b, int c, double theta0, double ja, double jc);
    /**
     * Get the bond pair of electron fluxing. The function is dq_a = j (r_ab - r0); dq_b = - dq_a
     * @param index    index of bond param
     * @param[out] a        atom index a
     * @param[out] b        atom index b
     * @param[out] r0       eq bond length r0
     * @param[out] j        pre factor j
     */
    void getCoulFluxBond(int index, int& a, int& b, double& r0, double& j) const;
    /**
     * Get the angle pair of electron fluxing. The angle is angle a-b-c, and function is:
     * dq_a = j_a * (theta_abc - theta0); dq_c = j_c * (theta_abc - theta0); dq_b = - dq_a - dq_c
     * @param index    index of angle param
     * @param[out] a        atom index a
     * @param[out] b        atom index b
     * @param[out] c        atom index c
     * @param[out] theta0   bond angle eq
     * @param[out] ja       pre factor for atom a
     * @param[out] jc       pre factor for atom c
     */
    void getCoulFluxAngle(int index, int& a, int& b, int& c, double& theta0, double& ja, double& jc) const;
    /**
     * Get the method used for handling long range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);
    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the reciprocal space cutoff and separation
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * For PME calculations, if setPMEParameters() is used to set alpha to something other than 0,
     * this value is ignored.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Set the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the reciprocal space cutoff and separation
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * For PME calculations, if setPMEParameters() is used to set alpha to something other than 0,
     * this value is ignored.
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Add the nonbonded force parameters for a particle.  This should be called once for each particle
     * in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.
     * For calculating the Lennard-Jones interaction between two particles, the arithmetic mean of the sigmas
     * and the geometric mean of the epsilons for the two interacting particles is used (the Lorentz-Berthelot
     * combining rule).
     *
     * @param charge    the charge of the particle, measured in units of the proton charge
     * @return the index of the particle that was added
     */
    int addParticle(double charge);
    /**
     * Get the nonbonded force parameters for a particle.
     *
     * @param index          the index of the particle for which to get parameters
     * @param[out] charge    the charge of the particle, measured in units of the proton charge
     */
    void getParticleParameters(int index, double& charge) const;
    /**
     * Set the nonbonded force parameters for a particle.  When calculating the Lennard-Jones interaction between two particles,
     * it uses the arithmetic mean of the sigmas and the geometric mean of the epsilons for the two interacting particles
     * (the Lorentz-Berthelot combining rule).
     *
     * @param index     the index of the particle for which to set parameters
     * @param charge    the charge of the particle, measured in units of the proton charge
     */
    void setParticleParameters(int index, double charge);
    /**
     * Add an interaction to the list of exceptions that should be calculated differently from other interactions.
     * If chargeProd and epsilon are both equal to 0, this will cause the interaction to be completely omitted from
     * force and energy calculations.
     *
     * In many cases, you can use createExceptionsFromBonds() rather than adding each exception explicitly.
     *
     * @param particle1  the index of the first particle involved in the interaction
     * @param particle2  the index of the second particle involved in the interaction
     * @param charge1    the scaled product of the atomic charges (i.e. the strength of the Coulomb interaction), measured in units of the proton charge squared
     * @param charge2    the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
     * @param scale      the scale
     * @param replace    determines the behavior if there is already an exception for the same two particles.  If true, the existing one is replaced.  If false,
     *                   an exception is thrown.
     * @return           the index of the exception that was added
     */
    int addException(int particle1, int particle2, double charge1, double charge2, double scale, bool replace = false);
    /**
     * Get the force field parameters for an interaction that should be calculated differently from others.
     *
     * @param index           the index of the interaction for which to get parameters
     * @param[out] particle1  the index of the first particle involved in the interaction
     * @param[out] particle2  the index of the second particle involved in the interaction
     * @param[out] charge1    the scaled product of the atomic charges (i.e. the strength of the Coulomb interaction), measured in units of the proton charge squared
     * @param[out] charge2    the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
     * @param[out] scale      the scale
     */
    void getExceptionParameters(int index, int& particle1, int& particle2, double& charge1, double& charge2, double& scale) const;
    /**
     * Set the force field parameters for an interaction that should be calculated differently from others.
     * If chargeProd and epsilon are both equal to 0, this will cause the interaction to be completely omitted from
     * force and energy calculations.
     *
     * @param index      the index of the interaction for which to get parameters
     * @param particle1  the index of the first particle involved in the interaction
     * @param particle2  the index of the second particle involved in the interaction
     * @param charge1    the scaled product of the atomic charges (i.e. the strength of the Coulomb interaction), measured in units of the proton charge squared
     * @param charge2    the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
     * @param scale      the scale
     */
    void setExceptionParameters(int index, int particle1, int particle2, double charge1, double charge2, double scale);
    /**
     * Get the force group that reciprocal space interactions for Ewald or PME are included in.  This allows multiple
     * time step integrators to evaluate direct and reciprocal space interactions at different intervals: getForceGroup()
     * specifies the group for direct space, and getReciprocalSpaceForceGroup() specifies the group for reciprocal space.
     * If this is -1 (the default value), the same force group is used for reciprocal space as for direct space.
     */
    int getReciprocalSpaceForceGroup() const;
    /**
     * Set the force group that reciprocal space interactions for Ewald or PME are included in.  This allows multiple
     * time step integrators to evaluate direct and reciprocal space interactions at different intervals: setForceGroup()
     * specifies the group for direct space, and setReciprocalSpaceForceGroup() specifies the group for reciprocal space.
     * If this is -1 (the default value), the same force group is used for reciprocal space as for direct space.
     *
     * @param group    the group index.  Legal values are between 0 and 31 (inclusive), or -1 to use the same force group
     *                 that is specified for direct space.
     */
    void setReciprocalSpaceForceGroup(int group);
    /**
     * Identify exceptions based on the molecular topology.  Particles which are separated by one or two bonds are set
     * to not interact at all, while pairs of particles separated by three bonds (known as "1-4 interactions") have
     * their Coulomb and Lennard-Jones interactions reduced by a fixed factor.
     *
     * @param bonds           the set of bonds based on which to construct exceptions.  Each element specifies the indices of
     *                        two particles that are bonded to each other.
     * @param coulomb14Scale  pairs of particles separated by three bonds will have the strength of their Coulomb interaction
     *                        multiplied by this factor
     */
    void createExceptionsFromBonds(const std::vector<std::pair<int, int> >& bonds, double coulomb14Scale);
    /**
     * Add a new global parameter that parameter offsets may depend on.  The default value provided to
     * this method is the initial value of the parameter in newly created Contexts.  You can change
     * the value at any time by calling setParameter() on the Context.
     * 
     * @param name             the name of the parameter
     * @param defaultValue     the default value of the parameter
     * @return the index of the parameter that was added
     */
    int addGlobalParameter(const std::string& name, double defaultValue);
    /**
     * Get the name of a global parameter.
     *
     * @param index     the index of the parameter for which to get the name
     * @return the parameter name
     */
    const std::string& getGlobalParameterName(int index) const;
    /**
     * Set the name of a global parameter.
     *
     * @param index          the index of the parameter for which to set the name
     * @param name           the name of the parameter
     */
    void setGlobalParameterName(int index, const std::string& name);
    /**
     * Get the default value of a global parameter.
     *
     * @param index     the index of the parameter for which to get the default value
     * @return the parameter default value
     */
    double getGlobalParameterDefaultValue(int index) const;
    /**
     * Set the default value of a global parameter.
     *
     * @param index          the index of the parameter for which to set the default value
     * @param defaultValue   the default value of the parameter
     */
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    /**
     * Update the particle and exception parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setParticleParameters() and setExceptionParameters() to modify this object's parameters, then call
     * updateParametersInContext() to copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the parameters of particles and exceptions.
     * All other aspects of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be
     * changed by reinitializing the Context.  Furthermore, only the chargeProd, sigma, and epsilon values of an exception
     * can be changed; the pair of particles involved in the exception cannot change.  Finally, this method cannot be used
     * to add new particles or exceptions, only to change the parameters of existing ones.
     */
    void updateParametersInContext(Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == CoulFluxForce::Ewald;
    }
    /**
     * Get whether periodic boundary conditions should be applied to exceptions.  Usually this is not
     * appropriate, because exceptions are normally used to represent bonded interactions (1-2, 1-3, and
     * 1-4 pairs), but there are situations when it does make sense.  For example, you may want to simulate
     * an infinite chain where one end of a molecule is bonded to the opposite end of the next periodic
     * copy.
     * 
     * Regardless of this value, periodic boundary conditions are only applied to exceptions if they also
     * are applied to other interactions.  If the nonbonded method is NoCutoff or CutoffNonPeriodic, this
     * value is ignored.  Also note that cutoffs are never applied to exceptions, again because they are
     * normally used to represent bonded interactions.
     */
    bool getExceptionsUsePeriodicBoundaryConditions() const;
    /**
     * Set whether periodic boundary conditions should be applied to exceptions.  Usually this is not
     * appropriate, because exceptions are normally used to represent bonded interactions (1-2, 1-3, and
     * 1-4 pairs), but there are situations when it does make sense.  For example, you may want to simulate
     * an infinite chain where one end of a molecule is bonded to the opposite end of the next periodic
     * copy.
     * 
     * Regardless of this value, periodic boundary conditions are only applied to exceptions if they also
     * get applied to other interactions.  If the nonbonded method is NoCutoff or CutoffNonPeriodic, this
     * value is ignored.  Also note that cutoffs are never applied to exceptions, again because they are
     * normally used to represent bonded interactions.
     */
    void setExceptionsUsePeriodicBoundaryConditions(bool periodic);
protected:
    ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    class ExceptionInfo;
    class GlobalParameterInfo;
    class ParticleOffsetInfo;
    class ExceptionOffsetInfo;
    NonbondedMethod nonbondedMethod;
    double cutoffDistance, ewaldErrorTol, alpha;
    bool exceptionsUsePeriodic;
    int recipForceGroup, nx, ny, nz;
    void addExclusionsToSet(const std::vector<std::set<int> >& bonded12, std::set<int>& exclusions, int baseParticle, int fromParticle, int currentLevel) const;
    int getGlobalParameterIndex(const std::string& parameter) const;
    std::vector<ParticleInfo> particles;
    std::vector<ExceptionInfo> exceptions;
    std::vector<GlobalParameterInfo> globalParameters;
    std::vector<std::vector<int>> bondList, angleList;
    std::vector<std::vector<double>> bondParameters, angleParameters;
    std::map<std::pair<int, int>, int> exceptionMap;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class CoulFluxForce::ParticleInfo {
public:
    double charge;
    ParticleInfo() {
        charge = 0.0;
    }
    ParticleInfo(double charge) :
        charge(charge) {
    }
};

/**
 * This is an internal class used to record information about an exception.
 * @private
 */
class CoulFluxForce::ExceptionInfo {
public:
    int particle1, particle2;
    double charge1, charge2, scale;
    ExceptionInfo() {
        particle1 = particle2 = -1;
        charge1 = charge2 = scale = 0.0;
    }
    ExceptionInfo(int particle1, int particle2, double charge1, double charge2, double scale) :
        particle1(particle1), particle2(particle2), charge1(charge1), charge2(charge2), scale(scale) {
    }
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class CoulFluxForce::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};

} // namespace OpenMM

#endif /*OPENMM_COULFLUXFORCE_H_*/
