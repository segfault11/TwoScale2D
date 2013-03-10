//-----------------------------------------------------------------------------
//  WCSPHSolver.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 13.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once

#include "ParticleSystem.h"
#include "stdutil.h"


struct WCSPHConfig
{
    DECL_DEFAULTS (WCSPHConfig)


    WCSPHConfig (float xs, float ys, float xe, float ye, float effectiveRadius,
        float restDensity, float taitCoeff, float speedSound, float alpha, 
        float tensionCoefficient, float timeStep);
    ~WCSPHConfig ();

    float DomainOrigin[2];
    float DomainEnd[2];
    int DomainDimensions[2];
    float EffectiveRadius;
    float RestDensity;
    float TaitCoeffitient;
    float SpeedSound;
    float Alpha;
    float TensionCoefficient;
    float TimeStep;
};

class WCSPHSolver
{
    DECL_DEFAULTS (WCSPHSolver)
    
    struct NeighborGrid
    {
        DECL_DEFAULTS (NeighborGrid)

        NeighborGrid (const int gridDimensions[2], int maxParticles,  
            dim3 blockDim);
        ~NeighborGrid ();

        int particleCount;      //!< current # of particles in the grid (host)
        int* dParticleCount;    //!< current # of particles in the grid (dev)
        int* dParticleHashs;    
       
        int* dCellStart;
        int* dCellEnd;

        const dim3 blockDim;
        dim3 gridDim;
    };

public: 
    WCSPHSolver (const WCSPHConfig& config, ParticleSystem& fluidParticles, 
        ParticleSystem& boundaryParticles);
    ~WCSPHSolver ();
    
    void Bind () const;
    void Unbind () const;
    void Advance ();

private:
    inline void initBoundaries () const;

    inline void updateNeighborGrid (unsigned char activeID);
    inline void updatePositions (unsigned char activeID);

    float mDomainOrigin[2];
    float mDomainEnd[2];
    int mDomainDimensions[2];
    float mEffectiveRadius;
    float mRestDensity;
    float mTaitCoeffitient;
    float mSpeedSound;
    float mAlpha;
    float mTensionCoeffient;
    float mTimeStep;

    ParticleSystem* mFluidParticles;
    NeighborGrid mFluidParticleGrid;
    
    //ParticleSystem* mFluidParticlesHigh;
    //NeighborGrid mFluidParticleGridHigh;


    ParticleSystem* mBoundaryParticles;
    
    dim3 mGridDim;
    dim3 mBlockDim;

    mutable bool mIsBoundaryInit;
    int* mdBoundaryParticleHashs;
    int* mdBoundaryParticleIDs;
    int* mdBoundaryCellStartIndices;
    int* mdBoundaryCellEndIndices;
};