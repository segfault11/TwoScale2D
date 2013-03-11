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
        float effectiveRadiusHigh, float restDensity, float taitCoeff, 
        float speedSound, float alpha, float tensionCoefficient, float timeStep);
    ~WCSPHConfig ();

    float DomainOrigin[2];
    float DomainEnd[2];
    int DomainDimensions[2];
    int DomainDimensionsHigh[2];
    float EffectiveRadius;
    float EffectiveRadiusHigh;
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

        NeighborGrid (const int gridDimensions[2], int maxParticles);
        ~NeighborGrid ();

        int* dParticleHashs;    
        int* dCellStart;
        int* dCellEnd;
    };

public: 
    WCSPHSolver (const WCSPHConfig& config, ParticleSystem& fluidParticles, 
        ParticleSystem& fluidParticlesHigh,ParticleSystem& boundaryParticles);
    ~WCSPHSolver ();
    
    void Bind () const;
    void Unbind () const;
    void Advance ();
    void AdvanceHigh ();

private:
    inline void initBoundaries () const;

    inline void updateNeighborGrid (unsigned char activeID);
    inline void updatePositions (unsigned char activeID);
    inline void updateNeighborGridHigh (unsigned char activeID);
    inline void updatePositionsHigh (unsigned char activeID);


    float mDomainOrigin[2];
    float mDomainEnd[2];
    int mDomainDimensions[2];
    int mDomainDimensionsHigh[2];
    float mEffectiveRadius;    
    float mEffectiveRadiusHigh;
    float mRestDensity;
    float mTaitCoeffitient;
    float mSpeedSound;
    float mAlpha;
    float mTensionCoeffient;
    float mTimeStep;

    dim3 mBlockDim;

    dim3 mGridDim;
    int* mdParticleCount;
    ParticleSystem* mFluidParticles;
    NeighborGrid mFluidParticleGrid;
    
    dim3 mGridDimHigh;
    int* mdParticleCountHigh;
    ParticleSystem* mFluidParticlesHigh;
    NeighborGrid mFluidParticleGridHigh;

    ParticleSystem* mBoundaryParticles;
    
    mutable bool mIsBoundaryInit;
    int* mdBoundaryParticleHashs;
    int* mdBoundaryParticleIDs;
    int* mdBoundaryCellStartIndices;
    int* mdBoundaryCellEndIndices;
};