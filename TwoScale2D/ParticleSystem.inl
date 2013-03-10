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
int* ParticleSystem::ParticleIDs ()
{
    return mdParticleIDs[mActive];
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
    return mPositionsVBO;
}
//-----------------------------------------------------------------------------
GLuint ParticleSystem::GetIndexVBO () const
{
    return mParticleIDsVBO[mActive];
}
//-----------------------------------------------------------------------------
