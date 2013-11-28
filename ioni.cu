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
texture<float, 3, cudaReadModeElementType> uxtex;
texture<float, 3, cudaReadModeElementType> uytex;
texture<float, 3, cudaReadModeElementType> uztex; // 3D texture
texture<float, 3, cudaReadModeElementType> eetex; // 3D texture
texture<float, 2, cudaReadModeElementType> atex; // 2D texture
texture<float, 2, cudaReadModeElementType> entex; // 2D texture
texture<float, 2, cudaReadModeElementType> tgtex; // 2D texture
texture<float, 2, cudaReadModeElementType> katex; // 2D texture

__constant__ float dmin;
__constant__ float drange;

__constant__ float emin;
__constant__ float erange;

__constant__ float enmin;
__constant__ float enrange;

__constant__ float tgmin;
__constant__ float tgrange;

/**
  (en * en * g, uu, sqrt(tt))
  x, y, z is a slice-normalized, scaled vector (from realToNormalized())
 **/
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

    if (iRenderOnly) return make_float3(en * en * g * ds, 0, 0);

    float uu = 1e4 * (
            tex3D(uxtex, x, y, z) * viewVector.x +
            tex3D(uytex, x, y, z) * viewVector.y +
            tex3D(uztex, x, y, z) * viewVector.z);

    return make_float3(en * en * g * ds, uu, sqrtf(tt));
}

__device__ float pointSpecificTau(float x, float y, float z) {
    float d2 = tex3D(dtex, x, y, z);
    float d1 = __logf(d2) + __logf(1.e-7);
    float e1 = __logf(tex3D(eetex, x, y, z)) - d1 + __logf(1.e5);
    float dd = (d1 - dmin) / drange; //density, energy lookup values
    float ee = (e1 - emin) / erange;
    float kk = tex2D(katex, ee, dd);

    return (kk * d2 * ds) / GRPH;
}

extern "C" {
    __global__ void iRender(float *out, float *tau, bool opacity) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > projectionXsize * projectionYsize) return;
        int ypixel = idx / projectionXsize;
        int xpixel = idx % projectionXsize;

        float3 positionIncrement = viewVector * ds;
        float3 currentPosition = initialCP(xpixel, ypixel);
        float3 np; //slice-normalized, scaled position

        float emiss = 0;
        float tausum = opacity ? tau[idx] : 0;

        if (currentPosition.x == INFINITY) {
            out[idx] = 0;
            return;
        }

        do {
            np = realToNormalized(currentPosition);

            if (tausum <= 1e2) {
                emiss += pointSpecificStuff(np.x, np.y, np.z, true).x *
                    expf(-tausum);
            }

            if (opacity) {
                tausum += pointSpecificTau(np.x, np.y, np.z);
            }

            currentPosition += positionIncrement;
        } while (isInSlice(currentPosition));

        if (opacity) tau[idx] = tausum;
        out[idx] = emiss;
    }

    __global__ void ilRender(float *out, float *dnus, float *tau,
            float nu0, float dopp_width0, int nlamb, bool opacity) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > projectionXsize * projectionYsize) return;
        int ypixel = idx / projectionXsize;
        int xpixel = idx % projectionXsize;
        float3 positionIncrement = viewVector * ds;

        float3 currentPosition = initialCP(xpixel, ypixel);
        float3 np;

        float3 pointSpecificData;

        float dnu;
        float nu;

        float tausum = opacity ? tau[idx] : 0;

        int nfreq;
        float dopp_width, shift, phi;

        for (nfreq = 0; nfreq < nlamb; nfreq++) {
            out[idx * nlamb + nfreq] = 0;
        }

        if (currentPosition.x == INFINITY) return;

        do {
            np = realToNormalized(currentPosition);

            if (tausum <= 1e2) {
                pointSpecificData = pointSpecificStuff(
                        np.x, np.y, np.z, false);

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
                tausum += pointSpecificTau(np.x, np.y, np.z);
            }

            currentPosition += positionIncrement;
        } while (isInSlice(currentPosition));

        for (nfreq = 0; nfreq < nlamb; nfreq++) {
            out[idx * nlamb + nfreq] *= DCF * DCF * PLANCK * (dnus[nfreq] + nu0) / (4 * PI);
        }

        if (opacity) tau[idx] = tausum;
    }
}
