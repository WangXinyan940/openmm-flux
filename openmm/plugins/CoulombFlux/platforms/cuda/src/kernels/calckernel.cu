

extern "C" __global__ void calcdQdX(real* __restrict__ dQdXval, real* __restrict__ realCharge, const real4* __restrict__ posq, const int* __restrict__ dQdXidx, const real* __restrict__ initCharge, const int2* __restrict__ bondidx, const float2* __restrict__ bondparam, const int3* __restrict__ angleidx, const float3* __restrict__ angleparam){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < TOTAL_PARAM){
        if (idx < NUM_BOND){
            int i0, i1;
            i0 = bondidx[idx].x;
            i1 = bondidx[idx].y;
            real4 p0, p1;
            p0 = posq[i0];
            p1 = posq[i1];
            real r_x = p0.x - p1.x;
            real r_y = p0.y - p1.y;
            real r_z = p0.z - p1.z;
            real onedist = RSQRT(r_x * r_x + r_y * r_y + r_z * r_z);
            
        }
    }
}