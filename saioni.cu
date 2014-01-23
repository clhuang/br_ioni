#include "renderer.cuh"

#define MP 1.67e-27
#define KB 1.38e-23
#define CC 3.00e8
#define PLANCK 6.63e-34
#define PI 3.1415965
#define SQRTPI 1.772454
#define GRPH 2.3804910e-24
#define DCF 1e7 //density conversion factor

texture<float, 3, cudaReadModeElementType> dtex; // 3D texture
texture<float, 3, cudaReadModeElementType> eetex; // 3D texture
texture<float, 2, cudaReadModeElementType> tgtex; // 2D texture
texture<float, 3, cudaReadModeElementType> uatex; // velocity along axis of integration
texture<float, 2, cudaReadModeElementType> atex; // 2D texture
texture<float, 2, cudaReadModeElementType> entex; // 2D texture
texture<float, 2, cudaReadModeElementType> katex; // 2D texture
texture<float, 1, cudaReadModeElementType> xtex; // derivative of integration-axis
texture<float, 1, cudaReadModeElementType> aptex; // derivative of integration-axis

__constant__ float dmin;
__constant__ float drange;

__constant__ float emin;
__constant__ float erange;

__constant__ float nsteps;
__constant__ char axis;
__constant__ bool reverse; //go along axis in reverse direction

__constant__ int projectionXsize;
__constant__ int projectionYsize;

#define X_AXIS 0
#define Y_AXIS 1
#define Z_AXIS 2

__device__ float3 pointSpecificStuff(float x, float y, float z, bool iRenderOnly) {
    float d1 = __logf(tex3D(dtex, x, y, z)) + __logf(1.e-7);
    float e1 = __logf(tex3D(eetex, x, y, z)) - d1 + __logf(1.e5);
    float dd = (d1 - dmin) / drange; //density, energy lookup values
    float ee = (e1 - emin) / erange;

    float tt = tex2D(tgtex, ee, dd);
    float en = __expf(tex2D(entex, ee, dd));
    float edi = (en * tt - enmin) / enrange; //temperature, edensity lookup values
    float tti = (__log10f(tt) - tgmin) / tgrange;

    float g = tex2D(atex, edi, tti); //lookup g
    float ds = tex1D(aptex);

    if (iRenderOnly) return make_float3(en * en * g * ds, 0, 0);

    float uu = 1e4 * (tex3D(uatex, x, y, z) * (reverse ? -1 : 1));

    return make_float3(en * en * g * ds, uu, sqrtf(tt));
}

__device__ float pointSpecificTau(float x, float y, float z) {
    float d2 = tex3D(dtex, x, y, z);
    float d1 = __logf(d2) + __logf(1.e-7);
    float e1 = __logf(tex3D(eetex, x, y, z)) - d1 + __logf(1.e5);
    float dd = (d1 - dmin) / drange; //density, energy lookup values
    float ee = (e1 - emin) / erange;
    float kk = tex2D(katex, ee, dd);
    float ds = tex1D(aptex);

    return (kk * d2 * ds) / GRPH;
}

extern "C" {
    __global__ void iRender(float *out, float *tau, bool opacity) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > projectionXsize * projectionYsize) return;
        int ypixel = idx / projectionXsize;
        int xpixel = idx % projectionXsize;

        float3 cp;
        float* ati; //axis to increment
        float da = (reverse ? -1.0 : 1.0) / nsteps;
        float target = reverse ? 0 : 1;

        switch(axis) {
            case X_AXIS: cp.y = xpixel / (float) projectionXsize;
                         cp.z = ypixel / (float) projectionYsize;
                         ati = &cp.x;
                         break;
            case Y_AXIS: cp.x = xpixel / (float) projectionXsize;
                         cp.z = ypixel / (float) projectionYsize;
                         ati = &cp.y;
                         break;
            case Z_AXIS: cp.x = xpixel / (float) projectionXsize;
                         cp.y = ypixel / (float) projectionYsize;
                         ati = &cp.z;
                         break;
            default: return;
        }

        *ati = 1 - target; //start at either 0 or 1

        float emiss = 0;
        float tausum = opacity ? tau[idx] : 0;

        do {
            if (tausum <= 1e2) {
                emiss += pointSpecificStuff(cp.x, cp.y, cp.z, true).x *
                    expf(-tausum);
            }

            if (opacity) {
                tausum += pointSpecificTau(cp.x, cp.y, cp.z);
            }

            *ati += da;
        } while (reverse ? (*ati > target) : (*ati < target));

        if (opacity) tau[idx] = tausum;
        out[idx] = emiss;
    }

    __global__ void ilRender(float *out, float *dnus, float *tau,
            float nu0, float dopp_width0, int nlamb, bool opacity) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > projectionXsize * projectionYsize) return;
        int ypixel = idx / projectionXsize;
        int xpixel = idx % projectionXsize;

        float3 cp;
        float* ati; //axis to increment
        float da = (reverse ? -1.0 : 1.0) / nsteps;
        float target = reverse ? 0 : 1;

        switch(axis) {
            case X_AXIS: cp.x = 1 - target;
                         cp.y = xpixel / (float) projectionXsize;
                         cp.z = ypixel / (float) projectionYsize;
                         ati = &cp.x;
                         break;
            case Y_AXIS: cp.y = 1 - target;
                         cp.x = xpixel / (float) projectionXsize;
                         cp.z = ypixel / (float) projectionYsize;
                         ati = &cp.y;
                         break;
            case Z_AXIS: cp.z = 1 - target;
                         cp.x = xpixel / (float) projectionXsize;
                         cp.y = ypixel / (float) projectionYsize;
                         ati = &cp.z;
                         break;
            default: return;
        }

        *ati = 1 - target;

        float3 pointSpecificData;

        float dnu;
        float nu;

        float tausum = opacity ? tau[idx] : 0;

        int nfreq;
        float dopp_width, shift, phi;

        for (nfreq = 0; nfreq < nlamb; nfreq++) {
            out[idx * nlamb + nfreq] = 0;
        }

        do {
            if (tausum <= 1e2) {
                pointSpecificData = pointSpecificStuff(
                        cp.x, cp.y, cp.z, false);

                dopp_width = pointSpecificData.z * dopp_width0;

                pointSpecificData.x *= expf(-tausum);

                for (nfreq = 0; nfreq < nlamb; nfreq++) {
                    dnu = dnus[nfreq];
                    nu = dnu + nu0;
                    shift = (dnu - nu * pointSpecificData.y / CC) / dopp_width;
                    phi = __expf(-shift * shift) / (SQRTPI * dopp_width);

                    out[idx * nlamb + nfreq] += phi * pointSpecificData.x;
                }
            }

            if (opacity) {
                tausum += pointSpecificTau(cp.x, cp.y, cp.z);
            }

            *ati += da;
        } while (reverse ? (*ati > target) : (*ati < target));

        for (nfreq = 0; nfreq < nlamb; nfreq++) {
            out[idx * nlamb + nfreq] *= DCF * DCF * PLANCK * (dnus[nfreq] + nu0) / (4 * PI);
        }

        if (opacity) tau[idx] = tausum;
    }
}
