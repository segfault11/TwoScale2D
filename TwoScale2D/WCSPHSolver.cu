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

//---------------------------------------------------------------------------
//  Macros
//---------------------------------------------------------------------------
#define EMPTY_CELL 0xFFFFFFFF
#define PI 3.14159265358979323846
//---------------------------------------------------------------------------
//  Constants on device
//---------------------------------------------------------------------------
__constant__ float gdDomainOrigin[2];
__constant__ int   gdDomainDimensions[2];
__constant__ float gdEffectiveRadius;
__constant__ float gdRestDensity;
__constant__ float gdTaitCoefficient;
__constant__ float gdSpeedSound;
__constant__ float gdAlpha;
__constant__ float gdM4KernelCoeff; 
__constant__ float gdM4KernelGradCoeff; 
__constant__ float gdFluidParticleMass;
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
//---------------------------------------------------------------------------
//  Defintions of device kernels
//---------------------------------------------------------------------------
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
//---------------------------------------------------------------------------
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
//---------------------------------------------------------------------------
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
//---------------------------------------------------------------------------
__device__ inline int2 computeGridCoordinate (const float2& pos)
{
    int2 coord;
    coord.x = (pos.x - gdDomainOrigin[0])/gdEffectiveRadius;
    coord.y = (pos.y - gdDomainOrigin[1])/gdEffectiveRadius;
    
    coord.x = min(max(0, coord.x), gdDomainDimensions[0] - 1);
    coord.y = min(max(0, coord.y), gdDomainDimensions[1] - 1);

    return coord;
}
//---------------------------------------------------------------------------
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
//---------------------------------------------------------------------------
__device__ inline int computeHash (const float2& pos)
{
    int2 coord = computeGridCoordinate(pos);
   
    return coord.y*gdDomainDimensions[0] + coord.x; 
}
//---------------------------------------------------------------------------
__device__ inline int computeHash (const int2& coord)
{ 
    return coord.y*gdDomainDimensions[0] + coord.x; 
}
//---------------------------------------------------------------------------
__device__ inline int computeHash (unsigned int i, unsigned int j)
{ 
    return j*gdDomainDimensions[0] + i; 
}
//---------------------------------------------------------------------------
__device__ inline float norm (const float2& v)
{
    return sqrt(v.x*v.x + v.y*v.y);
}
//---------------------------------------------------------------------------
__device__ inline float dot (const float2& a, const float2& b)
{
    return a.x*b.x + a.y*b.y;
}
//---------------------------------------------------------------------------
__device__ inline void updateDensityCell (float& density, const float2& pos,
    const float* const dParticlePositons, const int* const dParticleIDs,
    int start, int end)
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
//---------------------------------------------------------------------------
__device__ inline void updateAccCell(float2& acc, float2& accT, float2& accB, 
    const float2& pos, const float2& vel, float dens, float pre, 
    const float* const dParticlePositions, 
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const float* const dParticleVelocities, 
    const int* const dParticleIDs, 
    int start, int end, int startB, int endB)
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

    float c = gdBoundaryParticleMass/(gdFluidParticleMass + gdBoundaryParticleMass);

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
//---------------------------------------------------------------------------
//  Defintions of global kernels
//---------------------------------------------------------------------------
__global__ void updateParticleType (int* const dParticleTypes, 
    int* const dParticleIDs, const float* const dParticlePositions, 
    unsigned int numParticles)
{

}
__global__ void computeParticleHashAndResetIndex (int* const dParticleHashs, 
    int* const dParticleIDs, const float* const dParticlePositions, 
    unsigned int numParticles)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    // reset indices
    dParticleIDs[idx] = idx;

    // set particle hash
    float2 pos;
    pos.x = dParticlePositions[2*idx + 0];
    pos.y = dParticlePositions[2*idx + 1];
    dParticleHashs[idx] = computeHash(pos);
}
//---------------------------------------------------------------------------
__global__ void computeCellStartEndIndices (int* const dCellStartIndices,
    int* const dCellEndIndices, const int* const dParticleHashs, 
    unsigned int numParticles)
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
//---------------------------------------------------------------------------
__global__ void computeParticleDensityPressure 
    (float* const dParticleDensities, float* const dParticlePressures,
    const int* const dParticleIDs, const int* const dCellStartIndices,
    const int* const dCellEndIndices, const float* const dParticlePositions,
    unsigned int numParticles)
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
//---------------------------------------------------------------------------
__global__ void computeParticleAccelerationAndAdvance 
    (float* const dParticlePositions, float* const dParticleVelocities,
    const float* const dParticleDensities, const float* const dParticlePressures, 
    const int* const dParticleIDs, const int* const dCellStartIndices,
    const int* const dCellEndIndices, float dt, unsigned int numParticles)
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
            updateAccCell(acc, accT, accB, pos, vel, density, pressure, dParticlePositions,
                dParticleDensities, dParticlePressures, dParticleVelocities,
                dParticleIDs, start, end, startB, endB);
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

}
//---------------------------------------------------------------------------
//  Definiton of WCSPHConfig
//---------------------------------------------------------------------------
WCSPHConfig::WCSPHConfig 
(
    float xs, 
    float ys, 
    float xe,
    float ye, 
    float effectiveRadius, 
    float restDensity, 
    float taitCoeff, 
    float speedSound, 
    float alpha, 
    float tensionCoefficient,
    float timeStep
)
: 
    EffectiveRadius(effectiveRadius), 
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
    CUDA_SAFE_CALL( cudaMalloc(&dNumParticles, sizeof(int)) );

    int sizeIds = maxParticles*sizeof(int);
    CUDA_SAFE_CALL( cudaMalloc(&dParticleHashs,  sizeIds) );
    CUDA_SAFE_CALL( cudaMalloc(&dParticleIDs[0], sizeIds) );
    CUDA_SAFE_CALL( cudaMalloc(&dParticleIDs[1], sizeIds) );

    int sizeCellLists = gridDimensions[0]*gridDimensions[1]*sizeof(int);
    CUDA_SAFE_CALL( cudaMalloc(&dCellStart, sizeCellLists) );
    CUDA_SAFE_CALL( cudaMalloc(&dCellEnd,   sizeCellLists) );
}
//-----------------------------------------------------------------------------
WCSPHSolver::NeighborGrid::~NeighborGrid ()
{
    CUDA::SafeFree<int>(&dNumParticles);
    CUDA::SafeFree<int>(&dParticleIDs[0]);
    CUDA::SafeFree<int>(&dParticleIDs[1]);
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
    ParticleSystem& boundaryParticles
)
: 
    mEffectiveRadius(config.EffectiveRadius), 
    mRestDensity(config.RestDensity),
    mTaitCoeffitient(config.TaitCoeffitient),
    mSpeedSound(config.SpeedSound),
    mAlpha(config.Alpha), 
    mTensionCoeffient(config.TensionCoefficient), 
    mTimeStep(config.TimeStep),
    mFluidParticles(&fluidParticles), 
    mBoundaryParticles(&boundaryParticles), 

    mIsBoundaryInit(false),
    mFluidParticleGrid(config.DomainDimensions, fluidParticles.GetMaxParticles())
{
    mDomainOrigin[0] = config.DomainOrigin[0];
    mDomainOrigin[1] = config.DomainOrigin[1];
    mDomainEnd[0] = config.DomainEnd[0];
    mDomainEnd[1] = config.DomainEnd[1];

    mDomainDimensions[0] = static_cast<int>(std::ceil((mDomainEnd[0] - 
        mDomainOrigin[0])/ mEffectiveRadius));
    mDomainDimensions[1] = static_cast<int>(std::ceil((mDomainEnd[1] - 
        mDomainOrigin[1])/mEffectiveRadius));


    // set cuda block and grid dimensions
    mBlockDim.x = 256; 
    mBlockDim.y = 1; 
    mBlockDim.z = 1; 
    unsigned int numParticles = mFluidParticles->GetNumParticles();
    unsigned int numBlocks = numParticles/mBlockDim.x;
    mGridDim.x = numParticles % mBlockDim.x == 0 ? numBlocks : numBlocks + 1;
    mGridDim.y = 1;
    mGridDim.z = 1;
    
    // allocate extra device memory for neighbor search
    unsigned int size = sizeof(float)*mBoundaryParticles->GetNumParticles();
    CUDA_SAFE_CALL ( cudaMalloc(&mdBoundaryParticleHashs, size) );
    CUDA_SAFE_CALL ( cudaMalloc(&mdBoundaryParticleIDs, size) );
    unsigned int domainSize = mDomainDimensions[0]*mDomainDimensions[1]*
        sizeof(int);
    CUDA_SAFE_CALL ( cudaMalloc(&mdBoundaryCellStartIndices, domainSize) );
    CUDA_SAFE_CALL ( cudaMalloc(&mdBoundaryCellEndIndices, domainSize) );
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
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdEffectiveRadius, &mEffectiveRadius,
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
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol(gdM4KernelCoeff, &m4KernelCoeff,
        sizeof(float)) );
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol(gdM4KernelGradCoeff, &m4GradKernelCoeff,
        sizeof(float)) );
    float mass = mFluidParticles->GetMass();
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol(gdFluidParticleMass, 
        &mass, sizeof(float)) );
    mass = mBoundaryParticles->GetMass();
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol(gdBoundaryParticleMass, 
        &mass, sizeof(float)) );


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
    CUDA::Timer timer;

    mFluidParticles->Map();
    mBoundaryParticles->Map();
    timer.Start();
    //setUpNeighborSearch();
    this->updateNeighborGrid();
    unsigned int numParticles = mFluidParticles->GetNumParticles();
    computeParticleDensityPressure<<<mGridDim, mBlockDim>>>
        (mFluidParticles->Densities(), mFluidParticles->Pressures(), 
        mFluidParticleGrid.dParticleIDs[0], mFluidParticleGrid.dCellStart, 
        mFluidParticleGrid.dCellEnd, mFluidParticles->Positions(), 
        numParticles);    
    computeParticleAccelerationAndAdvance<<<mGridDim, mBlockDim>>>
        (mFluidParticles->Positions(), mFluidParticles->Velocities(),
        mFluidParticles->Densities(), mFluidParticles->Pressures(), 
        mFluidParticleGrid.dParticleIDs[0], mFluidParticleGrid.dCellStart, 
        mFluidParticleGrid.dCellEnd, mTimeStep, numParticles);
    timer.Stop();
    timer.DumpElapsed();
    mBoundaryParticles->Unmap();
    mFluidParticles->Unmap();
}
//-----------------------------------------------------------------------------
//  Private member 
//-----------------------------------------------------------------------------
void WCSPHSolver::updateNeighborGrid ()
{
    unsigned int numParticles = mFluidParticles->GetNumParticles();
    
    // compute hash of active particles
    computeParticleHashAndResetIndex<<<mGridDim, mBlockDim>>>
        (mFluidParticleGrid.dParticleHashs, mFluidParticleGrid.dParticleIDs[0], 
         mFluidParticles->Positions(), numParticles);
    
    // sort active ids by hash
    thrust::sort_by_key
        (thrust::device_ptr<int>(mFluidParticleGrid.dParticleHashs),
        thrust::device_ptr<int>(mFluidParticleGrid.dParticleHashs + numParticles), 
        thrust::device_ptr<int>(mFluidParticleGrid.dParticleIDs[0]));

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
        (mFluidParticleGrid.dCellStart, mFluidParticleGrid.dCellEnd, 
        mFluidParticleGrid.dParticleHashs, numParticles);

}
//-----------------------------------------------------------------------------
//void WCSPHSolver::setUpNeighborSearch ()
//{
//    unsigned int numParticles = mFluidParticles->GetNumParticles();
//    computeParticleHashAndResetIndex<<<mGridDim, mBlockDim>>>(mdParticleHashs,
//        mdParticleIDs, mFluidParticles->Positions(), numParticles);
//    thrust::sort_by_key(thrust::device_ptr<int>(mdParticleHashs),
//        thrust::device_ptr<int>(mdParticleHashs + numParticles), 
//        thrust::device_ptr<int>(mdParticleIDs));
//    unsigned int size = mDomainDimensions[0]*mDomainDimensions[1]*sizeof(int);
//    CUDA_SAFE_CALL ( cudaMemset(mdCellStartIndices, EMPTY_CELL, size) ); 
//    CUDA_SAFE_CALL ( cudaMemset(mdCellEndIndices, EMPTY_CELL, size) ); 
//    int sharedMemSize = sizeof(int)*(mBlockDim.x + 1);
//    computeCellStartEndIndices<<<mGridDim, mBlockDim, sharedMemSize>>>
//        (mdCellStartIndices, mdCellEndIndices, mdParticleHashs, numParticles);
//
//}
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
    computeParticleHashAndResetIndex<<<gridDim, blockDim>>>
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
    //CUDA::DumpArray<int>(mdBoundaryCellEndIndices,mDomainDimensions[0]*mDomainDimensions[1], 0, 1, 200);
    mBoundaryParticles->Unmap();
}
//-----------------------------------------------------------------------------
