//-----------------------------------------------------------------------------
//  ParticleSystem.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 12.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once

#include <string>
#include "stdutil.h"
#include "OpenGL\OpenGL.h"
#include "CUDA\cuda.h"

class WCSPHSolver;

class ParticleSystem
{
    DECL_DEFAULTS (ParticleSystem)

    friend WCSPHSolver;

public:
    ParticleSystem (unsigned int maxParticles, float mass);
    ~ParticleSystem ();

    // class access
    inline float* Positions ();
    inline float* Velocities ();
    inline float* Densities ();
    inline float* Pressures ();
    inline int* ParticleIDs ();

    inline float GetMass () const;
    inline unsigned int GetNumParticles () const;
    inline unsigned int GetMaxParticles () const;


    inline GLuint GetIndexVBO() const;
    inline GLuint GetPositionsVBO () const;
    
    void PushParticle (float position[2]);

    // map opengl memory to cuda memory (particle positions)
    void Map ();
    void Unmap ();

    void Save (const std::string& filename) const;

private:
    void release ();

    unsigned int mNumParticles;
    unsigned int mMaxParticles;
    
    // cuda/gl interop member
    GLuint mPositionsVBO;
    GLuint mParticleIDsVBO[2];
    cudaGraphicsResource_t mGraphicsResources[3];
    bool mIsMapped;
    
    // cuda device ptr
    float* mdPositions;
    float* mdVelocities;
    float* mdDensities;
    float* mdPressures;

    int mActive;
    int* mdParticleIDs[2];   //!< flip flop array for 
                             //!< storing active particleIDS


    float mMass;    
};

#include "ParticleSystem.inl"

ParticleSystem* CreateParticleBox (float xs, float ys, float dx,
    unsigned int dimX, unsigned int dimY, float particleMass);
ParticleSystem* CreateParticleBoxCanvas (float xs, float ys, float dx,
    int dimX, int dimY, int offX, int offY,
    float particleMass);