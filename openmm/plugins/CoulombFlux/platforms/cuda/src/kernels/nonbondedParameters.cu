/**
 * Compute the nonbonded parameters for particles and exceptions.
 */
extern "C" __global__ void computeParameters(mixed* __restrict__ energyBuffer, bool includeSelfEnergy, real* __restrict__ globalParams,
        int numAtoms, const float4* __restrict__ baseParticleParams, real4* __restrict__ posq, real* __restrict__ charge
#ifdef HAS_EXCEPTIONS
        , int numExceptions, const float4* __restrict__ baseExceptionParams, float4* __restrict__ exceptionParams
#endif
        ) {
    mixed energy = 0;

    // Compute particle parameters.
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numAtoms; i += blockDim.x*gridDim.x) {
        float4 params = baseParticleParams[i];
#ifdef USE_POSQ_CHARGES
        posq[i].w = params.x;
#else
        charge[i] = params.x;
#endif
    }

    // Compute exception parameters.
    
#ifdef HAS_EXCEPTIONS
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numExceptions; i += blockDim.x*gridDim.x) {
        float4 params = baseExceptionParams[i];
        exceptionParams[i] = make_float4((float) params.x, (float) params.y, (float) params.z, 0);
    }
#endif
    if (includeSelfEnergy)
        energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
}

/**
 * Compute parameters for subtracting the reciprocal part of excluded interactions.
 */
extern "C" __global__ void computeExclusionParameters(real4* __restrict__ posq, real* __restrict__ charge, float2* __restrict__ sigmaEpsilon,
        int numExclusions, const int2* __restrict__ exclusionAtoms, float4* __restrict__ exclusionParams) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numExclusions; i += blockDim.x*gridDim.x) {
        int2 atoms = exclusionAtoms[i];
#ifdef USE_POSQ_CHARGES
        real charge1 = posq[atoms.x].w;
        real charge2 = posq[atoms.y].w;
#else
        real charge1 = charge[atoms.x];
        real charge2 = charge[atoms.y];
#endif
        float epsilon = 0;
        exclusionParams[i] = make_float4((float) charge1, charge2, epsilon, 0);
    }
}