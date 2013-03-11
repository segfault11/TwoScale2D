//-----------------------------------------------------------------------------
//  WCSPHSolver.cu
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 13.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#include <thrust\sort.h>
#include <thrust\device_ptr.h>
#include <thrust\for_each.h>
#include <thrust\iterator\zip_iterator.h>
#include "WCSPHSolver.h"
#include <iostream>

//-----------------------------------------------------------------------------
//  Macros
//-----------------------------------------------------------------------------
#define EMPTY_CELL 0xFFFFFFFF
#define PI 3.14159265358979323846
//-----------------------------------------------------------------------------
//  Constants on device
//-----------------------------------------------------------------------------
__constant__ float gdDomainOrigin[2];
__constant__ int   gdDomainDimensions[2];
__constant__ int   gdDomainDimensionsHigh[2];
__constant__ float gdEffectiveRadius;
__constant__ float gdEffectiveRadiusHigh;
__constant__ float gdRestDensity;
__constant__ float gdTaitCoefficient;
__constant__ float gdSpeedSound;
__constant__ float gdAlpha;
__constant__ float gdM4KernelCoeff; 
__constant__ float gdM4KernelGradCoeff; 
__constant__ float gdM4KernelCoeffHigh; 
__constant__ float gdM4KernelGradCoeffHigh; 
__constant__ float gdFluidParticleMass;
__constant__ float gdFluidParticleMassHigh;
__constant__ float gdBoundaryParticleMass;
__constant__ float gdTensionCoefficient;

texture<float, cudaTextureType1D, cudaReadModeElementType> 
    gdBoundaryPositionsTex;
texture<int, cudaTextureType1D, cudaReadModeElementType> 
    gdBoundaryParticleIDsTex;
texture<int, cudaTextureType1D, cudaReadModeElementType> 
    gdBoundaryCellStartIndicesTex;
texture<int, cudaTextureType1D, cudaReadModeElementType> 
    gdBoundaryCellEndIndicesTex;
//-----------------------------------------------------------------------------
//  Defintions of device kernels
//-----------------------------------------------------------------------------
__device__ inline float evaluateBoundaryForceWeight (float xNorm)  
{
    float q = xNorm*2.0f/gdEffectiveRadius;
   
    float c = 0.02f*gdSpeedSound*gdSpeedSound/(xNorm*xNorm);

    if (q < 2.0f/3.0f)
    {
        return c*2.0f/3.0f;
    }
    else if (q < 1.0f)
    {
        return c*(2.0f*q -3.0f/2.0f*q*q);
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;
        return c*0.5f*a*a;
    }
    else
    {
        return 0.0f;
    }
}
//-----------------------------------------------------------------------------
__device__ inline float evaluateBoundaryForceWeightHigh (float xNorm)  
{
    float q = xNorm*2.0f/gdEffectiveRadiusHigh;
   
    float c = 0.02f*gdSpeedSound*gdSpeedSound/(xNorm*xNorm);

    if (q < 2.0f/3.0f)
    {
        return c*2.0f/3.0f;
    }
    else if (q < 1.0f)
    {
        return c*(2.0f*q -3.0f/2.0f*q*q);
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;
        return c*0.5f*a*a;
    }
    else
    {
        return 0.0f;
    }
}
//-----------------------------------------------------------------------------
__device__ inline float evaluateM4Kernel (float xNorm)  
{
    float q = xNorm*2.0f/gdEffectiveRadius;
    
    if (q < 1.0f)
    {
        float a = 2.0f - q;
        float b = 1.0f - q;

        return gdM4KernelCoeff*(a*a*a - 4.0f*b*b*b);
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;

        return gdM4KernelCoeff*a*a*a;
    }
    else
    {
        return 0.0f;
    }
}
//-----------------------------------------------------------------------------
__device__ inline float evaluateM4KernelHigh (float xNorm)  
{
    float q = xNorm*2.0f/gdEffectiveRadiusHigh;
    
    if (q < 1.0f)
    {
        float a = 2.0f - q;
        float b = 1.0f - q;

        return gdM4KernelCoeffHigh*(a*a*a - 4.0f*b*b*b);
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;

        return gdM4KernelCoeffHigh*a*a*a;
    }
    else
    {
        return 0.0f;
    }
}
//-----------------------------------------------------------------------------
__device__ inline void evaluateGradientM4Kernel (float& gradX, float& gradY,
    const float2& x, float xNorm)  
{
    // NOTE xNorm == 0 lead to devision by zero

    float q = xNorm*2.0f/gdEffectiveRadius;

    if (q < 1.0f)
    {
        float a = 2.0f - q;
        float b = 1.0f - q;
        float c = gdM4KernelGradCoeff*(a*a - 4.0f*b*b)/xNorm;

        gradX = c*x.x;
        gradY = c*x.y;

        return;
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;
        float c = gdM4KernelGradCoeff*a*a/xNorm;

        gradX = c*x.x;
        gradY = c*x.y;

        return;
    }
    else
    {
        gradX = 0.0f;
        gradY = 0.0f;

        return;
    }
}
//-----------------------------------------------------------------------------
__device__ inline void evaluateGradientM4KernelHigh 
(
    float& gradX, 
    float& gradY,
    const float2& x, 
    float xNorm
)  
{
    // NOTE xNorm == 0 lead to devision by zero

    float q = xNorm*2.0f/gdEffectiveRadiusHigh;

    if (q < 1.0f)
    {
        float a = 2.0f - q;
        float b = 1.0f - q;
        float c = gdM4KernelGradCoeffHigh*(a*a - 4.0f*b*b)/xNorm;

        gradX = c*x.x;
        gradY = c*x.y;

        return;
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;
        float c = gdM4KernelGradCoeffHigh*a*a/xNorm;

        gradX = c*x.x;
        gradY = c*x.y;

        return;
    }
    else
    {
        gradX = 0.0f;
        gradY = 0.0f;

        return;
    }
}
//-----------------------------------------------------------------------------
__device__ inline int2 computeGridCoordinate (const float2& pos)
{
    int2 coord;
    coord.x = (pos.x - gdDomainOrigin[0])/gdEffectiveRadius;
    coord.y = (pos.y - gdDomainOrigin[1])/gdEffectiveRadius;
    
    coord.x = min(max(0, coord.x), gdDomainDimensions[0] - 1);
    coord.y = min(max(0, coord.y), gdDomainDimensions[1] - 1);

    return coord;
}
//-----------------------------------------------------------------------------
__device__ inline int2 computeGridCoordinateHigh (const float2& pos)
{
    int2 coord;
    coord.x = (pos.x - gdDomainOrigin[0])/gdEffectiveRadiusHigh;
    coord.y = (pos.y - gdDomainOrigin[1])/gdEffectiveRadiusHigh;
    
    coord.x = min(max(0, coord.x), gdDomainDimensionsHigh[0] - 1);
    coord.y = min(max(0, coord.y), gdDomainDimensionsHigh[1] - 1);

    return coord;
}
//-----------------------------------------------------------------------------
__device__ inline int2 computeGridCoordinate (const float2& pos, 
    float offset)
{
    int2 coord;
    coord.x = (pos.x + offset - gdDomainOrigin[0])/gdEffectiveRadius;
    coord.y = (pos.y + offset - gdDomainOrigin[1])/gdEffectiveRadius;
    
    coord.x = min(max(0, coord.x), gdDomainDimensions[0] - 1);
    coord.y = min(max(0, coord.y), gdDomainDimensions[1] - 1);
    
    return coord;
}
//-----------------------------------------------------------------------------
__device__ inline int2 computeGridCoordinateHigh (const float2& pos, 
    float offset)
{
    int2 coord;
    coord.x = (pos.x + offset - gdDomainOrigin[0])/gdEffectiveRadiusHigh;
    coord.y = (pos.y + offset - gdDomainOrigin[1])/gdEffectiveRadiusHigh;
    
    coord.x = min(max(0, coord.x), gdDomainDimensionsHigh[0] - 1);
    coord.y = min(max(0, coord.y), gdDomainDimensionsHigh[1] - 1);
    
    return coord;
}
//-----------------------------------------------------------------------------
__device__ inline int computeHash (const float2& pos)
{
    int2 coord = computeGridCoordinate(pos);
   
    return coord.y*gdDomainDimensions[0] + coord.x; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHashHigh (const float2& pos)
{
    int2 coord = computeGridCoordinateHigh(pos);
   
    return coord.y*gdDomainDimensionsHigh[0] + coord.x; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHash (const int2& coord)
{ 
    return coord.y*gdDomainDimensions[0] + coord.x; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHashHigh (const int2& coord)
{ 
    return coord.y*gdDomainDimensionsHigh[0] + coord.x; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHash (unsigned int i, unsigned int j)
{ 
    return j*gdDomainDimensions[0] + i; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHashHigh (unsigned int i, unsigned int j)
{ 
    return j*gdDomainDimensionsHigh[0] + i; 
}
//-----------------------------------------------------------------------------
__device__ inline float norm (const float2& v)
{
    return sqrt(v.x*v.x + v.y*v.y);
}
//-----------------------------------------------------------------------------
__device__ inline float dot (const float2& a, const float2& b)
{
    return a.x*b.x + a.y*b.y;
}
//-----------------------------------------------------------------------------
__device__ inline void updateDensityCell
(
    float& density, 
    const float2& pos,
    const float* const dParticlePositons, 
    const int* const dParticleIDs,
    int start, 
    int end
)
{
    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        float2 posj;
        posj.x = dParticlePositons[2*j + 0];
        posj.y = dParticlePositons[2*j + 1];

        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        
        float dist = norm(posij);
        
        if (dist <= gdEffectiveRadius)
        {
            density += evaluateM4Kernel(dist);
        }
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateDensityCellHigh
(
    float& density, 
    const float2& pos,
    const float* const dParticlePositons, 
    const int* const dParticleIDs,
    int start, 
    int end
)
{
    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        float2 posj;
        posj.x = dParticlePositons[2*j + 0];
        posj.y = dParticlePositons[2*j + 1];

        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        
        float dist = norm(posij);
        
        if (dist <= gdEffectiveRadiusHigh)
        {
            density += evaluateM4KernelHigh(dist);
        }
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateAccCell
(   
    float2& acc, 
    float2& accT, 
    float2& accB, 
    const float2& pos, 
    const float2& vel, 
    float dens, 
    float pre, 
    const float* const dParticlePositions, 
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const float* const dParticleVelocities, 
    const int* const dParticleIDs, 
    int start, 
    int end, 
    int startB, 
    int endB
)
{
    float dens2 = dens*dens;

    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];
        float2 velj;
        velj.x = dParticleVelocities[2*j + 0];
        velj.y = dParticleVelocities[2*j + 1];
        float densj = dParticleDensities[j];
        float prej = dParticlePressures[j];
        float densj2 = densj*densj;
        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        float dist = norm(posij);

        if (dist != 0.0f && dist < gdEffectiveRadius)
        {
            // compute pressure contribution
            float coeff = pre/dens2 + prej/densj2;

            // compute artifcial velocity
            float2 velij;
            velij.x = vel.x - velj.x;
            velij.y = vel.y - velj.y;
            float dvp = dot(velij, posij);

            if (dvp < 0.0f)
            {
                coeff -= dvp/(dist*dist + 0.01f*gdEffectiveRadius*
                    gdEffectiveRadius)*2.0f*gdEffectiveRadius*gdAlpha*
                    gdSpeedSound/(dens + densj);
            }
            
            float2 grad;
            evaluateGradientM4Kernel(grad.x, grad.y, posij, dist);

            acc.x += coeff*grad.x;
            acc.y += coeff*grad.y;


            float w = evaluateM4Kernel(dist);
            accT.x += w*posij.x;
            accT.y += w*posij.y; 
        }
    }

    float c = gdBoundaryParticleMass/(gdFluidParticleMass + 
        gdBoundaryParticleMass);

    for (int i = startB; i < endB; i++)
    {
        int k = tex1Dfetch(gdBoundaryParticleIDsTex, i);
        float2 posk;
        posk.x = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 0);
        posk.y = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 1);
        float2 posik;
        posik.x = pos.x - posk.x;
        posik.y = pos.y - posk.y;
        float dist = norm(posik);
        float gamma = evaluateBoundaryForceWeight(dist);

        accB.x += c*gamma*posik.x;
        accB.y += c*gamma*posik.y; 
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateAccCellHigh
(   
    float2& acc, 
    float2& accT, 
    const float2& pos, 
    const float2& vel, 
    float dens, 
    float pre, 
    const float* const dParticlePositions, 
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const float* const dParticleVelocities, 
    const int* const dParticleIDs, 
    int start, 
    int end
)
{
    float dens2 = dens*dens;

    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];
        float2 velj;
        velj.x = dParticleVelocities[2*j + 0];
        velj.y = dParticleVelocities[2*j + 1];
        float densj = dParticleDensities[j];
        float prej = dParticlePressures[j];
        float densj2 = densj*densj;
        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        float dist = norm(posij);

        if (dist != 0.0f && dist < gdEffectiveRadiusHigh)
        {
            // compute pressure contribution
            float coeff = pre/dens2 + prej/densj2;

            // compute artifcial velocity
            float2 velij;
            velij.x = vel.x - velj.x;
            velij.y = vel.y - velj.y;
            float dvp = dot(velij, posij);

            if (dvp < 0.0f)
            {
                coeff -= dvp/(dist*dist + 0.01f*gdEffectiveRadiusHigh*
                    gdEffectiveRadiusHigh)*2.0f*gdEffectiveRadiusHigh*gdAlpha*
                    gdSpeedSound/(dens + densj);
            }
            
            float2 grad;
            evaluateGradientM4KernelHigh(grad.x, grad.y, posij, dist);

            acc.x += coeff*grad.x;
            acc.y += coeff*grad.y;


            float w = evaluateM4Kernel(dist);
            accT.x += w*posij.x;
            accT.y += w*posij.y; 
        }
    }
}
//-----------------------------------------------------------------------------
__device__ void updateAccBCellHigh
(
    float2& accB, 
    float2 pos, 
    int start, 
    int end
)
{
    float c = gdBoundaryParticleMass/(gdFluidParticleMassHigh + 
        gdBoundaryParticleMass);

    for (int i = start; i < end; i++)
    {
        int k = tex1Dfetch(gdBoundaryParticleIDsTex, i);
        float2 posk;
        posk.x = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 0);
        posk.y = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 1);
        float2 posik;
        posik.x = pos.x - posk.x;
        posik.y = pos.y - posk.y;
        float dist = norm(posik);
        float gamma = evaluateBoundaryForceWeightHigh(dist);

        accB.x += c*gamma*posik.x;
        accB.y += c*gamma*posik.y; 
    }
}
//-----------------------------------------------------------------------------
__device__ void initSubParticlesAndAddToList
(
    float* const dParticlePositionsHigh,
    int* const dParticleIDsHigh,
    int numParticlesHigh,                   // current # of high particles             
    int id,                                 // id of the low particle
    const float2& pos
)
{
    // add high res particles to the list
    dParticleIDsHigh[numParticlesHigh + 0] = 4*id + 0;
    dParticleIDsHigh[numParticlesHigh + 1] = 4*id + 1;
    dParticleIDsHigh[numParticlesHigh + 2] = 4*id + 2;
    dParticleIDsHigh[numParticlesHigh + 3] = 4*id + 3;

    #define DIR_LEN 0.35355339f
    float h = gdEffectiveRadius;

    dParticlePositionsHigh[8*id + 0] = pos.x + DIR_LEN*h;
    dParticlePositionsHigh[8*id + 1] = pos.y + DIR_LEN*h;
    dParticlePositionsHigh[8*id + 2] = pos.x - DIR_LEN*h;
    dParticlePositionsHigh[8*id + 3] = pos.y + DIR_LEN*h;
    dParticlePositionsHigh[8*id + 4] = pos.x + DIR_LEN*h;
    dParticlePositionsHigh[8*id + 5] = pos.y - DIR_LEN*h;
    dParticlePositionsHigh[8*id + 6] = pos.x - DIR_LEN*h;
    dParticlePositionsHigh[8*id + 7] = pos.y - DIR_LEN*h;

}
//-----------------------------------------------------------------------------
//  Defintions of global kernels
//-----------------------------------------------------------------------------
__global__ void initParticleIDs 
(   
    int* const dParticleIDs, 
    unsigned int maxParticles
)
{
    // initially the particle id list stores particle ids in ascending order

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= maxParticles)
    {
        return;
    }

    dParticleIDs[idx] = idx;
}
//-----------------------------------------------------------------------------
__global__ void computeParticleHash 
(   
    int* const dParticleHashs, 
    int* const dParticleIDs, 
    const float* const dParticlePositions, 
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    // get particle id
    int id = dParticleIDs[idx];

    // get particle position
    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];
    
    // set particle hash
    dParticleHashs[idx] = computeHash(pos);
}
//-----------------------------------------------------------------------------
__global__ void computeParticleHashHigh
(   
    int* const dParticleHashs, 
    int* const dParticleIDs, 
    const float* const dParticlePositions, 
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    // get particle id
    int id = dParticleIDs[idx];

    // get particle position
    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];
    
    // set particle hash
    dParticleHashs[idx] = computeHashHigh(pos);
}
//-----------------------------------------------------------------------------
__global__ void computeCellStartEndIndices 
(
    int* const dCellStartIndices,
    int* const dCellEndIndices, 
    const int* const dParticleHashs, 
    unsigned int numParticles
)
{
    extern __shared__ int sharedHash[];
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles) 
    {
        return;
    }

    int hash = dParticleHashs[idx];
    sharedHash[threadIdx.x + 1] = hash;
        
    if (idx > 0 && threadIdx.x == 0) 
    {
        sharedHash[0] = dParticleHashs[idx - 1];
    }

    __syncthreads();

    if (idx == 0 || hash != sharedHash[threadIdx.x])
    {
        dCellStartIndices[hash] = idx;
        
        if (idx > 0) 
        {
            dCellEndIndices[sharedHash[threadIdx.x]] = idx;
        }
    }

    if (idx == numParticles - 1)
    {
        dCellEndIndices[hash] = idx + 1;
    }
}
//-----------------------------------------------------------------------------
__global__ void computeParticleDensityPressure 
(
    float* const dParticleDensities, 
    float* const dParticlePressures,
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    const float* const dParticlePositions,
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dParticleIDs[idx];    
    
    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];

    int2 cs = computeGridCoordinate(pos, -gdEffectiveRadius);
    int2 ce = computeGridCoordinate(pos, gdEffectiveRadius);
    float density = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHash(i, j);
            int start = dCellStartIndices[hash];
            int end = dCellEndIndices[hash];
            updateDensityCell(density, pos, dParticlePositions, 
                 dParticleIDs, start, end);
        }
    }

    density *= gdFluidParticleMass;

    dParticleDensities[id] = density;
    float a = density/gdRestDensity;
    float a3 = a*a*a;
    dParticlePressures[id] = gdTaitCoefficient*(a3*a3*a - 1.0f);
}
//-----------------------------------------------------------------------------
__global__ void computeParticleDensityPressureHigh
(
    float* const dParticleDensities, 
    float* const dParticlePressures,
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    const float* const dParticlePositions,
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dParticleIDs[idx];    
    
    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];

    int2 cs = computeGridCoordinateHigh(pos, -gdEffectiveRadiusHigh);
    int2 ce = computeGridCoordinateHigh(pos, gdEffectiveRadiusHigh);
    float density = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHashHigh(i, j);
            int start = dCellStartIndices[hash];
            int end = dCellEndIndices[hash];
            updateDensityCellHigh
            (
                density, 
                pos, 
                dParticlePositions, 
                dParticleIDs, 
                start, 
                end
            );
        }
    }

    density *= gdFluidParticleMassHigh;

    dParticleDensities[id] = density;
    float a = density/gdRestDensity;
    float a3 = a*a*a;
    dParticlePressures[id] = gdTaitCoefficient*(a3*a3*a - 1.0f);
}
//-----------------------------------------------------------------------------
__global__ void computeParticleAccelerationAndAdvance 
(
    float* const dParticlePositions, 
    float* const dParticleVelocities,
    float* const dParticlePositionsHigh,
    int* const dParticleIDsNew,
    int* const dParticleIDsHighNew,
    int* const dParticleCount,
    int* const dParticleCountHigh,
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    float dt, 
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dParticleIDs[idx];    

    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];
    float2 vel;
    vel.x = dParticleVelocities[2*id + 0]; 
    vel.y = dParticleVelocities[2*id + 1]; 
    float density = dParticleDensities[id];
    float pressure = dParticlePressures[id];

    int2 cs = computeGridCoordinate(pos, -gdEffectiveRadius);
    int2 ce = computeGridCoordinate(pos, gdEffectiveRadius);
    
    float2 acc;
    acc.x = 0.0f;
    acc.y = 0.0f;

    float2 accT;
    accT.x = 0.0f;
    accT.y = 0.0f;

    float2 accB;
    accB.x = 0.0f;
    accB.y = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHash(i, j);
            int start = dCellStartIndices[hash];
            int end = dCellEndIndices[hash];
            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
            updateAccCell(acc, accT, accB, pos, vel, density, pressure, 
                dParticlePositions, dParticleDensities, dParticlePressures, 
                dParticleVelocities, dParticleIDs, start, end, startB, endB);
        }
    }

    acc.x *= -gdFluidParticleMass;
    acc.y *= -gdFluidParticleMass;

    acc.x -= gdTensionCoefficient*accT.x;
    acc.y -= gdTensionCoefficient*accT.y;

    acc.x += accB.x;
    acc.y += accB.y;
    
    acc.y -= 9.81f;

    vel.x += dt*acc.x; 
    vel.y += dt*acc.y;

    pos.x += dt*vel.x;
    pos.y += dt*vel.y;

    dParticleVelocities[2*id + 0] = vel.x;
    dParticleVelocities[2*id + 1] = vel.y;
    
    dParticlePositions[2*id + 0] = pos.x;
    dParticlePositions[2*id + 1] = pos.y;

    // add particle to id list, get index for particle id and store
    // id to new particle id list
    if (pos.x < 0.7f)
    {
        int index = atomicAdd(dParticleCount, 1);
        dParticleIDsNew[index] = id;
    }
    else
    {
        int numParticlesHigh = atomicAdd(dParticleCountHigh, 4);
        initSubParticlesAndAddToList
        (
            dParticlePositionsHigh, 
            dParticleIDsHighNew, 
            numParticlesHigh, 
            id, 
            pos
        );
    }
}
//-----------------------------------------------------------------------------
__global__ void computeParticleAccelerationAndAdvanceHigh
(
    float* const dParticlePositions, 
    float* const dParticleVelocities,
    int* const dParticleIDsNew,
    int* const dParticleCount,
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    float dt, 
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dParticleIDs[idx];    

    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];
    float2 vel;
    vel.x = dParticleVelocities[2*id + 0]; 
    vel.y = dParticleVelocities[2*id + 1]; 
    float density = dParticleDensities[id];
    float pressure = dParticlePressures[id];

    int2 cs = computeGridCoordinateHigh(pos, -gdEffectiveRadiusHigh);
    int2 ce = computeGridCoordinateHigh(pos, gdEffectiveRadiusHigh);
    
    float2 acc;
    acc.x = 0.0f;
    acc.y = 0.0f;

    float2 accT;
    accT.x = 0.0f;
    accT.y = 0.0f;

    float2 accB;
    accB.x = 0.0f;
    accB.y = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHashHigh(i, j);
            int start = dCellStartIndices[hash];
            int end = dCellEndIndices[hash];
            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
            updateAccCellHigh(acc, accT, pos, vel, density, pressure, 
                dParticlePositions, dParticleDensities, dParticlePressures, 
                dParticleVelocities, dParticleIDs, start, end);
        }
    }

    cs = computeGridCoordinate(pos, -gdEffectiveRadiusHigh);
    ce = computeGridCoordinate(pos, gdEffectiveRadiusHigh);

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHash(i, j);
            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
            updateAccBCellHigh(accB, pos, startB, endB);
        }
    }

    acc.x *= -gdFluidParticleMassHigh;
    acc.y *= -gdFluidParticleMassHigh;

    acc.x -= gdTensionCoefficient*accT.x;
    acc.y -= gdTensionCoefficient*accT.y;

    acc.x += accB.x;
    acc.y += accB.y;
    
    acc.y -= 9.81f;

    vel.x += dt*acc.x; 
    vel.y += dt*acc.y;

    pos.x += dt*vel.x;
    pos.y += dt*vel.y;

    dParticleVelocities[2*id + 0] = vel.x;
    dParticleVelocities[2*id + 1] = vel.y;
    
    dParticlePositions[2*id + 0] = pos.x;
    dParticlePositions[2*id + 1] = pos.y;


    int index = atomicAdd(dParticleCount, 1);
    dParticleIDsNew[index] = id;
}
//-----------------------------------------------------------------------------
//  Definiton of WCSPHConfig
//-----------------------------------------------------------------------------
WCSPHConfig::WCSPHConfig 
(
    float xs, 
    float ys, 
    float xe,
    float ye, 
    float effectiveRadius, 
    float effectiveRadiusHigh,
    float restDensity, 
    float taitCoeff, 
    float speedSound, 
    float alpha, 
    float tensionCoefficient,
    float timeStep
)
: 
    EffectiveRadius(effectiveRadius), 
    EffectiveRadiusHigh(EffectiveRadiusHigh),
    RestDensity(restDensity), 
    TaitCoeffitient(taitCoeff), 
    SpeedSound(speedSound),
    Alpha(alpha), 
    TensionCoefficient(tensionCoefficient),
    TimeStep(timeStep)
{
    if (xs >= xe || ys >= ye)
    {
        UTIL::ThrowException("Invalid configuration parameters", __FILE__, 
            __LINE__);
    }

    DomainOrigin[0] = xs;
    DomainOrigin[1] = ys;
    DomainEnd[0] = xe;
    DomainEnd[1] = ye;

    DomainDimensions[0] = static_cast<int>(std::ceil((DomainEnd[0] - 
        DomainOrigin[0])/EffectiveRadius));
    DomainDimensions[1] = static_cast<int>(std::ceil((DomainEnd[1] - 
        DomainOrigin[1])/EffectiveRadius));

    DomainDimensionsHigh[0] = static_cast<int>(std::ceil((DomainEnd[0] - 
        DomainOrigin[0])/EffectiveRadiusHigh));
    DomainDimensionsHigh[1] = static_cast<int>(std::ceil((DomainEnd[1] - 
        DomainOrigin[1])/EffectiveRadiusHigh));
}
//-----------------------------------------------------------------------------
WCSPHConfig::~WCSPHConfig ()
{

}
//-----------------------------------------------------------------------------
// DEFINITION: WCSPHSolver
//-----------------------------------------------------------------------------
//  - Nested Class : Neighbour Grid 
//-----------------------------------------------------------------------------
WCSPHSolver::NeighborGrid::NeighborGrid (const int gridDimensions[2], 
    int maxParticles)
{
    // malloc device mem
    int sizeIds = maxParticles*sizeof(int);
    CUDA_SAFE_CALL( cudaMalloc(&dParticleHashs,  sizeIds) );

    int sizeCellLists = gridDimensions[0]*gridDimensions[1]*sizeof(int);
    CUDA_SAFE_CALL( cudaMalloc(&dCellStart, sizeCellLists) );
    CUDA_SAFE_CALL( cudaMalloc(&dCellEnd,   sizeCellLists) );
}
//-----------------------------------------------------------------------------
WCSPHSolver::NeighborGrid::~NeighborGrid ()
{
    CUDA::SafeFree<int>(&dParticleHashs);
    CUDA::SafeFree<int>(&dCellStart);
    CUDA::SafeFree<int>(&dCellEnd);
}
//-----------------------------------------------------------------------------
//  - Public definitions
//-----------------------------------------------------------------------------
WCSPHSolver::WCSPHSolver 
(
    const WCSPHConfig& config, 
    ParticleSystem& fluidParticles, 
    ParticleSystem& fluidParticlesHigh, 
    ParticleSystem& boundaryParticles
)
: 
    mEffectiveRadius(config.EffectiveRadius), 
    mEffectiveRadiusHigh(config.EffectiveRadius),
    mRestDensity(config.RestDensity),
    mTaitCoeffitient(config.TaitCoeffitient),
    mSpeedSound(config.SpeedSound),
    mAlpha(config.Alpha), 
    mTensionCoeffient(config.TensionCoefficient), 
    mTimeStep(config.TimeStep),
    mFluidParticles(&fluidParticles),
    mFluidParticlesHigh(&fluidParticlesHigh),
    mBoundaryParticles(&boundaryParticles), 
    mIsBoundaryInit(false),
    mFluidParticleGrid
    (
        config.DomainDimensions, 
        fluidParticles.GetMaxParticles()
    ),
    mFluidParticleGridHigh
    (
        config.DomainDimensionsHigh, 
        fluidParticlesHigh.GetMaxParticles()
    ),
    mBlockDim(256, 1 , 1),
    mGridDim(1, 1, 1)
{
    mDomainOrigin[0] = config.DomainOrigin[0];
    mDomainOrigin[1] = config.DomainOrigin[1];
    mDomainEnd[0] = config.DomainEnd[0];
    mDomainEnd[1] = config.DomainEnd[1];

    mDomainDimensions[0] = config.DomainDimensions[0];
    mDomainDimensions[1] = config.DomainDimensions[1];
    mDomainDimensionsHigh[0] = config.DomainDimensionsHigh[0];
    mDomainDimensionsHigh[1] = config.DomainDimensionsHigh[1];

    // set cuda block and grid dimensions
    mBlockDim.x = 256; 
    mBlockDim.y = 1; 
    mBlockDim.z = 1; 
    unsigned int numParticles = mFluidParticles->GetNumParticles();
    unsigned int numBlocks = numParticles/mBlockDim.x;
    mGridDim.x = numParticles % mBlockDim.x == 0 ? numBlocks : numBlocks + 1;
    mGridDim.y = 1;
    mGridDim.z = 1;
    CUDA_SAFE_CALL( cudaMalloc(&mdParticleCount, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemset(mdParticleCount, 0, sizeof(int)) );

    // set cuda block and grid dimensions (high)
    unsigned int numParticlesHigh = mFluidParticlesHigh->GetNumParticles();
    unsigned int numBlocksHigh = numParticlesHigh/mBlockDim.x;
    mGridDimHigh.x = numParticlesHigh % mBlockDim.x == 0 ? numBlocksHigh : 
        numBlocksHigh + 1;
    mGridDimHigh.y = 1;
    mGridDimHigh.z = 1;
    CUDA_SAFE_CALL( cudaMalloc(&mdParticleCountHigh, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemset(mdParticleCountHigh, 0, sizeof(int)) );

    // allocate extra device memory for neighbor search (boundaries)
    unsigned int size = sizeof(float)*mBoundaryParticles->GetNumParticles();
    CUDA_SAFE_CALL( cudaMalloc(&mdBoundaryParticleHashs, size) );
    CUDA_SAFE_CALL( cudaMalloc(&mdBoundaryParticleIDs, size) );
    unsigned int domainSize = mDomainDimensions[0]*mDomainDimensions[1]*
        sizeof(int);
    CUDA_SAFE_CALL( cudaMalloc(&mdBoundaryCellStartIndices, domainSize) );
    CUDA_SAFE_CALL( cudaMalloc(&mdBoundaryCellEndIndices, domainSize) );
}
//-----------------------------------------------------------------------------
WCSPHSolver::~WCSPHSolver ()
{
    CUDA::SafeFree<int>(&mdBoundaryParticleHashs);
    CUDA::SafeFree<int>(&mdBoundaryParticleIDs);
    CUDA::SafeFree<int>(&mdBoundaryCellStartIndices);
    CUDA::SafeFree<int>(&mdBoundaryCellEndIndices);
}
//-----------------------------------------------------------------------------
void WCSPHSolver::Bind () const
{
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdDomainOrigin, mDomainOrigin, 
        2*sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdDomainDimensions, mDomainDimensions, 
        2*sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdDomainDimensionsHigh, mDomainDimensionsHigh, 
        2*sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdEffectiveRadius, &mEffectiveRadius,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdEffectiveRadiusHigh, &mEffectiveRadiusHigh,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdRestDensity, &mRestDensity,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdTaitCoefficient, &mTaitCoeffitient,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdSpeedSound, &mSpeedSound,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdAlpha, &mAlpha, sizeof(float)) );    
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdTensionCoefficient, &mTensionCoeffient,
        sizeof(float)) );

    float m4KernelCoeff = 20.0f/(14.0f*PI*mEffectiveRadius*mEffectiveRadius);
    float m4GradKernelCoeff = -120.0f/(14.0f*PI*mEffectiveRadius*mEffectiveRadius*
        mEffectiveRadius);
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdM4KernelCoeff, 
        &m4KernelCoeff,
        sizeof(float)) 
    );
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (   
        gdM4KernelGradCoeff, 
        &m4GradKernelCoeff,
        sizeof(float)
    ) );
    float m4KernelCoeffHigh = 
        20.0f/(14.0f*PI*mEffectiveRadiusHigh*mEffectiveRadiusHigh);
    float m4GradKernelCoeffHigh = 
        -120.0f/(14.0f*PI*mEffectiveRadiusHigh*mEffectiveRadiusHigh*
        mEffectiveRadiusHigh);
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdM4KernelCoeffHigh, 
        &m4KernelCoeffHigh,
        sizeof(float)
    ) );
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdM4KernelGradCoeffHigh, 
        &m4GradKernelCoeffHigh,
        sizeof(float)
    ) );
    float mass = mFluidParticles->GetMass();
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdFluidParticleMass, 
        &mass, 
        sizeof(float)
    ) );
    mass = mFluidParticlesHigh->GetMass();
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdFluidParticleMassHigh, 
        &mass, 
        sizeof(float)
    ) );
    mass = mBoundaryParticles->GetMass();
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdBoundaryParticleMass, 
        &mass, 
        sizeof(float)
    ) );

    // init boundary handling
    if (!mIsBoundaryInit)
    {
        initBoundaries();
        mIsBoundaryInit = true;
    }

    // bind boundary handling information to textures
    mBoundaryParticles->Map();
    cudaChannelFormatDesc descf = cudaCreateChannelDesc(32, 0, 0, 0,
		cudaChannelFormatKindFloat);
    cudaChannelFormatDesc desci = cudaCreateChannelDesc(32, 0, 0, 0,
		cudaChannelFormatKindSigned);
    CUDA_SAFE_CALL( cudaBindTexture(0, gdBoundaryPositionsTex, 
        mBoundaryParticles->Positions(), descf, 
        2*mBoundaryParticles->GetNumParticles()*sizeof(float)) );
    CUDA_SAFE_CALL( cudaBindTexture(0, gdBoundaryParticleIDsTex, 
        mdBoundaryParticleIDs, desci, 
        mBoundaryParticles->GetNumParticles()*sizeof(int)) );
    unsigned int domainSize = mDomainDimensions[0]*mDomainDimensions[1]*
        sizeof(int);
    CUDA_SAFE_CALL( cudaBindTexture(0, gdBoundaryCellStartIndicesTex, 
        mdBoundaryCellStartIndices, desci, domainSize) );
    CUDA_SAFE_CALL( cudaBindTexture(0, gdBoundaryCellEndIndicesTex, 
        mdBoundaryCellEndIndices, desci, domainSize) );
    mBoundaryParticles->Unmap();
}
//-----------------------------------------------------------------------------
void WCSPHSolver::Unbind () const
{
    CUDA_SAFE_CALL( cudaUnbindTexture(gdBoundaryPositionsTex) );
    CUDA_SAFE_CALL( cudaUnbindTexture(gdBoundaryParticleIDsTex) );
    CUDA_SAFE_CALL( cudaUnbindTexture(gdBoundaryCellStartIndicesTex) );
    CUDA_SAFE_CALL( cudaUnbindTexture(gdBoundaryCellEndIndicesTex) );
}
//-----------------------------------------------------------------------------
void WCSPHSolver::Advance ()
{
    static unsigned char activeID = 0;

    CUDA::Timer timer;

    mFluidParticles->Map();
    mFluidParticlesHigh->Map();
    mBoundaryParticles->Map();
    
    this->updateNeighborGrid(activeID);
    
    computeParticleDensityPressure<<<mGridDim, mBlockDim>>>
    (
        mFluidParticles->Densities(), mFluidParticles->Pressures(), 
        mFluidParticles->mdParticleIDs[activeID], 
        mFluidParticleGrid.dCellStart, mFluidParticleGrid.dCellEnd, 
        mFluidParticles->Positions(), mFluidParticles->mNumParticles
    );    

    this->updatePositions(activeID);

    mBoundaryParticles->Unmap();
    mFluidParticlesHigh->Unmap();
    mFluidParticles->Unmap();


    activeID = (activeID + 1) % 2; 
}
//-----------------------------------------------------------------------------
void WCSPHSolver::AdvanceHigh ()
{
    static unsigned char activeID = 0;

    CUDA::Timer timer;

    mFluidParticles->Map();
    mFluidParticlesHigh->Map();
    mBoundaryParticles->Map();
    
    this->updateNeighborGridHigh(activeID);
    
    computeParticleDensityPressureHigh<<<mGridDimHigh, mBlockDim>>>
    (
        mFluidParticlesHigh->Densities(), 
        mFluidParticlesHigh->Pressures(), 
        mFluidParticlesHigh->mdParticleIDs[activeID], 
        mFluidParticleGridHigh.dCellStart, 
        mFluidParticleGridHigh.dCellEnd, 
        mFluidParticlesHigh->Positions(), 
        mFluidParticlesHigh->mNumParticles
    );    

    this->updatePositionsHigh(activeID);

    mBoundaryParticles->Unmap();
    mFluidParticlesHigh->Unmap();
    mFluidParticles->Unmap();


    activeID = (activeID + 1) % 2; 
}
//-----------------------------------------------------------------------------
//  - private methods
//-----------------------------------------------------------------------------
void WCSPHSolver::updateNeighborGrid (unsigned char activeID)
{
    CUDA::Timer timer;
    
    // compute hash of active particles
    computeParticleHash<<<mGridDim, mBlockDim>>>
    (
        mFluidParticleGrid.dParticleHashs, 
        mFluidParticles->mdParticleIDs[activeID], 
        mFluidParticles->Positions(),
        mFluidParticles->mNumParticles
    );
    
    // sort active ids by hash
    thrust::sort_by_key
    (
        thrust::device_ptr<int>(mFluidParticleGrid.dParticleHashs),
        thrust::device_ptr<int>(mFluidParticleGrid.dParticleHashs +
            mFluidParticles->mNumParticles), 
        thrust::device_ptr<int>(mFluidParticles->mdParticleIDs[activeID])
    );

    // set all grid cells to be empty
    unsigned int size = mDomainDimensions[0]*mDomainDimensions[1]*
        sizeof(unsigned int);

    CUDA_SAFE_CALL ( cudaMemset(mFluidParticleGrid.dCellStart, 
        EMPTY_CELL, size) ); 
    CUDA_SAFE_CALL ( cudaMemset(mFluidParticleGrid.dCellEnd, 
        EMPTY_CELL, size) ); 

    // fill grid cells according to current particles
    int sharedMemSize = sizeof(int)*(mBlockDim.x + 1);
    computeCellStartEndIndices<<<mGridDim, mBlockDim, sharedMemSize>>>
    (
        mFluidParticleGrid.dCellStart, 
        mFluidParticleGrid.dCellEnd, 
        mFluidParticleGrid.dParticleHashs, 
        mFluidParticles->mNumParticles
    );
}
//-----------------------------------------------------------------------------
void WCSPHSolver::updateNeighborGridHigh (unsigned char activeID)
{
    CUDA::Timer timer;
    
    // compute hash of active particles
    computeParticleHashHigh<<<mGridDimHigh, mBlockDim>>>
    (
        mFluidParticleGridHigh.dParticleHashs, 
        mFluidParticlesHigh->mdParticleIDs[activeID], 
        mFluidParticlesHigh->Positions(),
        mFluidParticlesHigh->mNumParticles
    );
    
    // sort active ids by hash
    thrust::sort_by_key
    (
        thrust::device_ptr<int>(mFluidParticleGridHigh.dParticleHashs),
        thrust::device_ptr<int>(mFluidParticleGridHigh.dParticleHashs +
            mFluidParticlesHigh->mNumParticles), 
        thrust::device_ptr<int>(mFluidParticlesHigh->mdParticleIDs[activeID])
    );

    // set all grid cells to be empty
    unsigned int size = mDomainDimensionsHigh[0]*mDomainDimensionsHigh[1]*
        sizeof(unsigned int);

    CUDA_SAFE_CALL ( cudaMemset(mFluidParticleGridHigh.dCellStart, 
        EMPTY_CELL, size) ); 
    CUDA_SAFE_CALL ( cudaMemset(mFluidParticleGridHigh.dCellEnd, 
        EMPTY_CELL, size) ); 

    // fill grid cells according to current particles
    int sharedMemSize = sizeof(int)*(mBlockDim.x + 1);
    computeCellStartEndIndices<<<mGridDimHigh, mBlockDim, sharedMemSize>>>
    (
        mFluidParticleGridHigh.dCellStart, 
        mFluidParticleGridHigh.dCellEnd, 
        mFluidParticleGridHigh.dParticleHashs, 
        mFluidParticlesHigh->mNumParticles
    );
}
//-----------------------------------------------------------------------------
void WCSPHSolver::updatePositions (unsigned char activeID)
{
    CUDA::Timer timer;

    // reset particle count to zero
    CUDA_SAFE_CALL( cudaMemset(mdParticleCount, 0, sizeof(int)) );
    
    // reset particle count to zero
    timer.Start();
    computeParticleAccelerationAndAdvance<<<mGridDim, mBlockDim>>>
    (
        mFluidParticles->Positions(), 
        mFluidParticles->Velocities(), 
        mFluidParticlesHigh->mdPositions,
        mFluidParticles->mdParticleIDs[(activeID + 1) % 2], 
        mFluidParticlesHigh->mdParticleIDs[0], 
        mdParticleCount,
        mdParticleCountHigh, 
        mFluidParticles->Densities(),
        mFluidParticles->Pressures(),
        mFluidParticles->mdParticleIDs[activeID], 
        mFluidParticleGrid.dCellStart, 
        mFluidParticleGrid.dCellEnd, 
        mTimeStep, 
        mFluidParticles->mNumParticles
    );
    timer.Stop();
    timer.DumpElapsed();
    
    // copy back the # of current particles in the list
    // update particle system information
    CUDA_SAFE_CALL( cudaMemcpy
    (
        &mFluidParticles->mNumParticles, 
        mdParticleCount, 
        sizeof(int), 
        cudaMemcpyDeviceToHost
    ) );
    mFluidParticles->mActive = (activeID + 1) % 2;
    mGridDim.x = std::ceil
    (
        static_cast<float>(mFluidParticles->mNumParticles)/mBlockDim.x
    );

    // copy back the # of current particles (high) in the list
    // update particle system information
    CUDA_SAFE_CALL( cudaMemcpy
    (
        &mFluidParticlesHigh->mNumParticles, 
        mdParticleCountHigh, 
        sizeof(int), 
        cudaMemcpyDeviceToHost
     ) )
     
    //std::cout << mFluidParticles->mNumParticles << "/" << mFluidParticlesHigh->mNumParticles <<  "/" << (mFluidParticles->mMaxParticles - mFluidParticles->mNumParticles)*4 << std::endl; 
}
//-----------------------------------------------------------------------------
void WCSPHSolver::updatePositionsHigh (unsigned char activeID)
{
    CUDA::Timer timer;
 
    // reset particle count to zero
    CUDA_SAFE_CALL( cudaMemset(mdParticleCountHigh, 0, sizeof(int)) );
    
    // reset particle count to zero
    //timer.Start();
    computeParticleAccelerationAndAdvanceHigh<<<mGridDimHigh, mBlockDim>>>
    (
        mFluidParticlesHigh->Positions(), 
        mFluidParticlesHigh->Velocities(), 
        mFluidParticlesHigh->mdParticleIDs[(activeID + 1) % 2],
        mdParticleCountHigh,
        mFluidParticlesHigh->Densities(),
        mFluidParticlesHigh->Pressures(),
        mFluidParticlesHigh->mdParticleIDs[activeID], 
        mFluidParticleGridHigh.dCellStart, 
        mFluidParticleGridHigh.dCellEnd, 
        mTimeStep, 
        mFluidParticlesHigh->mNumParticles
    );
    //timer.Stop();
    //timer.DumpElapsed();

    // copy back the # of current particles in the list
    // update particle system information
    CUDA_SAFE_CALL( cudaMemcpy
    (
        &mFluidParticlesHigh->mNumParticles, 
        mdParticleCountHigh, 
        sizeof(int), 
        cudaMemcpyDeviceToHost
    ) );
    mFluidParticlesHigh->mActive = (activeID + 1) % 2;
    mGridDimHigh.x = std::ceil
    (
        static_cast<float>(mFluidParticlesHigh->mNumParticles)/mBlockDim.x
    );
}
//-----------------------------------------------------------------------------
void WCSPHSolver::initBoundaries () const
{
    dim3 gridDim;
    dim3 blockDim;      
    blockDim.x = 256; 
    blockDim.y = 1; 
    blockDim.z = 1; 
    unsigned int numBoundaryParticles = mBoundaryParticles->GetNumParticles();
    unsigned int numBlocks = numBoundaryParticles/blockDim.x;  
    gridDim.x = numBoundaryParticles % blockDim.x == 0 ? numBlocks : numBlocks + 1;
    gridDim.y = 1;
    gridDim.z = 1;
    mBoundaryParticles->Map();
    initParticleIDs<<<gridDim, blockDim>>>(mdBoundaryParticleIDs, 
        numBoundaryParticles);
    computeParticleHash<<<gridDim, blockDim>>>
        (mdBoundaryParticleHashs, mdBoundaryParticleIDs, 
        mBoundaryParticles->Positions(), numBoundaryParticles);
    thrust::sort_by_key(thrust::device_ptr<int>(mdBoundaryParticleHashs),
        thrust::device_ptr<int>(mdBoundaryParticleHashs + numBoundaryParticles), 
        thrust::device_ptr<int>(mdBoundaryParticleIDs));

    unsigned int domainSize = mDomainDimensions[0]*mDomainDimensions[1]*
        sizeof(int); 
    CUDA_SAFE_CALL ( cudaMemset(mdBoundaryCellStartIndices, EMPTY_CELL,
        domainSize) ); 
    CUDA_SAFE_CALL ( cudaMemset(mdBoundaryCellEndIndices, EMPTY_CELL, 
        domainSize) );
    int sharedMemSize = sizeof(int)*(blockDim.x + 1);
    computeCellStartEndIndices<<<gridDim, blockDim, sharedMemSize>>>
        (mdBoundaryCellStartIndices, mdBoundaryCellEndIndices, 
        mdBoundaryParticleHashs, numBoundaryParticles);
    mBoundaryParticles->Unmap();
}
//-----------------------------------------------------------------------------
