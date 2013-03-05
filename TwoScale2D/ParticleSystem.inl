//-----------------------------------------------------------------------------
//  ParticleSystem.inl
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 12.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Inline member definitions
//-----------------------------------------------------------------------------
float* ParticleSystem::Positions ()
{
    if (!mIsMapped)
    {
        UTIL::ThrowException("Positions are not mapped", __FILE__, __LINE__);
    }

    return mdPositions;
}
//-----------------------------------------------------------------------------
float* ParticleSystem::Velocities ()
{
    return mdVelocities;
}
//-----------------------------------------------------------------------------
float* ParticleSystem::Densities ()
{
    return mdDensities;
}
//-----------------------------------------------------------------------------
float* ParticleSystem::Pressures ()
{
    return mdPressures;
}
//-----------------------------------------------------------------------------
float ParticleSystem::GetMass () const
{
    return mMass;
}
//-----------------------------------------------------------------------------
unsigned int ParticleSystem::GetNumParticles () const
{
    return mNumParticles;
}
//-----------------------------------------------------------------------------
unsigned int ParticleSystem::GetMaxParticles () const
{
    return mMaxParticles;
}
//-----------------------------------------------------------------------------
GLuint ParticleSystem::GetPositionsVBO () const
{
    if (mIsMapped)
    {
        UTIL::ThrowException("Positions are currently mapped to CUDA memory",
            __FILE__, __LINE__);
    }


    return mPositionsVBO;
}
//-----------------------------------------------------------------------------
