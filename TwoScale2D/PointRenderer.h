//-----------------------------------------------------------------------------
//  PointRenderer.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 12.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once

#include "stdutil.h"
#include "OpenGL\OpenGL.h"
#include "ParticleSystem.h"

class PointRenderer 
{
    DECL_DEFAULTS (PointRenderer);

public:
    PointRenderer (const ParticleSystem& particleSystem, float xs, float ys,
        float xe, float ye, float r, float g, float b, float a);
    ~PointRenderer ();

    void Render () const;

private:
    const ParticleSystem* mParticleSystem;
    
    GLuint mProgram;
    GLuint mVAO;
};