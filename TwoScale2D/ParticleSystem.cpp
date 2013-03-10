//-----------------------------------------------------------------------------
//  ParticleSystem.cpp
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 12.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#include "ParticleSystem.h"
#include <iostream>
#include <fstream>

//-----------------------------------------------------------------------------
ParticleSystem::ParticleSystem (unsigned int maxParticles, float mass)
: mMaxParticles(maxParticles), mMass(mass), mNumParticles(0), mIsMapped(false),
mdPositions(NULL), mdVelocities(NULL), mdDensities(NULL), mdPressures(NULL),
mActive(0)
{
    unsigned int sizef = sizeof(float)*mMaxParticles;
    unsigned int sizei = sizeof(int)*mMaxParticles;

    // init particle positions
    float* posData = new float[2*maxParticles];
    memset(posData, 0, 2*sizef);
    GL::CreateBufferObject(mPositionsVBO, GL_ARRAY_BUFFER, 2*sizef, posData, 
        GL_DYNAMIC_COPY);
    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&mGraphicsResources[0], 
        mPositionsVBO, cudaGraphicsMapFlagsNone) );
    delete[] posData;

    // init index buffer
    int* indexData = new int[maxParticles];
    for (int i = 0; i < maxParticles; i++) indexData[i] = i;
    GL::CreateBufferObject(mParticleIDsVBO[0], GL_ELEMENT_ARRAY_BUFFER, sizei, 
        indexData, GL_DYNAMIC_COPY);
    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&mGraphicsResources[1], 
        mParticleIDsVBO[0], cudaGraphicsMapFlagsNone) );
    GL::CreateBufferObject(mParticleIDsVBO[1], GL_ELEMENT_ARRAY_BUFFER, sizei, 
        indexData, GL_DYNAMIC_COPY);
    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&mGraphicsResources[2], 
        mParticleIDsVBO[1], cudaGraphicsMapFlagsNone) );
    delete[] indexData;


    // allocate device memory
    CUDA_SAFE_CALL (cudaMalloc(&mdVelocities, 2*sizef));
    CUDA_SAFE_CALL (cudaMalloc(&mdDensities, sizef));
    CUDA_SAFE_CALL (cudaMalloc(&mdPressures, sizef));

    // set allocated memory to zero
    CUDA_SAFE_CALL (cudaMemset(mdVelocities, 0, 2*sizef));
    CUDA_SAFE_CALL (cudaMemset(mdDensities, 0, sizef));
    CUDA_SAFE_CALL (cudaMemset(mdPressures, 0, sizef));
}
//-----------------------------------------------------------------------------
ParticleSystem::~ParticleSystem ()
{
    this->release();
}
//-----------------------------------------------------------------------------
void ParticleSystem::PushParticle (float position[2])
{
    if (!mIsMapped)
    {
        UTIL::ThrowException("Positions is not mapped", __FILE__, __LINE__);
    }

    CUDA_SAFE_CALL( cudaMemcpy(mdPositions + 2*mNumParticles, position, 
        2*sizeof(float), cudaMemcpyHostToDevice) );
    mNumParticles++;
}
//-----------------------------------------------------------------------------
void ParticleSystem::Map ()
{
    CUDA_SAFE_CALL (cudaGraphicsMapResources(3, mGraphicsResources) );
    size_t nBytes;
    CUDA_SAFE_CALL (cudaGraphicsResourceGetMappedPointer
        (reinterpret_cast<void**>(&mdPositions), &nBytes,
        mGraphicsResources[0]));
    CUDA_SAFE_CALL (cudaGraphicsResourceGetMappedPointer
        (reinterpret_cast<void**>(&mdParticleIDs[0]), &nBytes,
        mGraphicsResources[1]));
    CUDA_SAFE_CALL (cudaGraphicsResourceGetMappedPointer
        (reinterpret_cast<void**>(&mdParticleIDs[1]), &nBytes,
        mGraphicsResources[2]));
    mIsMapped = true;
}
//-----------------------------------------------------------------------------
void ParticleSystem::Unmap ()
{
    cudaGraphicsUnmapResources(3, mGraphicsResources);
    mIsMapped = false;
}
//-----------------------------------------------------------------------------
void ParticleSystem::Save (const std::string& filename) const
{
    std::fstream file(filename, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char*>(&mNumParticles), 
        sizeof(unsigned int));
    file.flush();
    file.close();
}
//-----------------------------------------------------------------------------
// Private member definition
//-----------------------------------------------------------------------------
void ParticleSystem::release ()
{
    glDeleteBuffers(1, &mPositionsVBO);

    CUDA::SafeFree<float>(&mdVelocities);
    CUDA::SafeFree<float>(&mdDensities);
    CUDA::SafeFree<float>(&mdPressures);
}
//-----------------------------------------------------------------------------
//  public function defintions
//-----------------------------------------------------------------------------
ParticleSystem* CreateParticleBox (float xs, float ys, float dx, 
    unsigned int dimX, unsigned int dimY, float particleMass)
{
    unsigned int numParticles = dimX*dimY;
    ParticleSystem* particleSystem = new ParticleSystem(numParticles, 
        particleMass);
    
    particleSystem->Map();
    
    for (unsigned int j = 0; j < dimY; j++)
    {
        for (unsigned int i = 0; i < dimX; i++)
        {
            float pos[2];
            pos[0] = xs + dx*i;
            pos[1] = ys + dx*j;

            particleSystem->PushParticle(pos);
        }
    }

    particleSystem->Unmap();


    return particleSystem;
}
//-----------------------------------------------------------------------------
ParticleSystem* CreateParticleBoxCanvas (float xs, float ys, float dx,
    int dimX, int dimY, int offX, int offY,
    float particleMass)
{
    unsigned int numParticles = (dimX + 2*offX)*(dimY + 2*offY) - dimX*dimY;
    ParticleSystem* particleSystem = new ParticleSystem(numParticles, 
        particleMass);
     
    particleSystem->Map();

    for (int j = -offY; j < dimY + offY; j++)
    {
        for (int i = -offX; i < dimX + offX; i++)
        {
            if (i < 0 || i >= dimX || j < 0 || j >= dimY)
            {
                float pos[2];
                pos[0] = xs + dx*i;
                pos[1] = ys + dx*j;
                particleSystem->PushParticle(pos);
            }
        }
    }


    particleSystem->Unmap();


    return particleSystem;
}
//-----------------------------------------------------------------------------
