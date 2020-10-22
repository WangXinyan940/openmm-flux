#ifndef COULFLUX_OPENMM_CUDAKERNELS_H_
#define COULFLUX_OPENMM_CUDAKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMAmoeba                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2020 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs, Peter Eastman                                    *
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

#include "openmm/CoulFluxKernels.h"
#include "openmm/kernels.h"
#include "openmm/System.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "CudaNonbondedUtilities.h"
#include "CudaSort.h"
#include <cufft.h>

namespace OpenMM {

class CudaCalcCoulFluxKernel : public CalcCoulFluxKernel {
public:
    CudaCalcCoulFluxKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcCoulFluxKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the CoulFluxForce this kernel will be used for
     */
    void initialize(const System& system, const CoulFluxForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);

    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CoulFluxForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CoulFluxForce& force);
private:
    void computeParameters(ContextImpl& context);

    int numParticles, numExceptions;
    class ForceInfo;
    CudaArray dQdXidx;
    CudaArray dQdXval;
    CudaArray dEdQ;
    CudaArray CosSin; 
    CudaArray initCharge;
    CudaArray realCharge;
    CUfunction calcdQdXKernel;
    CUfunction calcEnergyCosSinKernel;
    CUfunction calcdEdQRecKernel;
    CUfunction calcdEdQRealKernel;
    CUfunction calcdPForceKernel;
    CUfunction calcdQForceKernel;
    CUfunction calcNoCutoffdPForceKernel;
    CUfunction calcNoCutoffdQForceKernel;
    CudaContext& cu;
    const System& system;

};

} // namespace OpenMM

#endif /*COULFLUX_OPENMM_CUDAKERNELS_H*/
