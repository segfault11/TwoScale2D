//-----------------------------------------------------------------------------
//  PointRenderer.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 12.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#include "PointRenderer.h"
#include <iostream>

//-----------------------------------------------------------------------------
// Public member definitions
//-----------------------------------------------------------------------------
PointRenderer::PointRenderer (const ParticleSystem& particleSystem, float xs, 
    float ys, float xe, float ye, float r, float g, float b, float a)
: mParticleSystem(&particleSystem)
{
    mProgram = glCreateProgram();
    GL::AttachShader(mProgram, "PointRendererVertexShader.glsl", 
        GL_VERTEX_SHADER);
    //GL::AttachShader(mProgram, "PointRendererGeometryShader.glsl", 
    //    GL_GEOMETRY_SHADER);
    GL::AttachShader(mProgram, "PointRendererFragmentShader.glsl", 
        GL_FRAGMENT_SHADER);
    GL::BindAttribLocation(mProgram, "position", 0);
    GL::BindFragDataLocation(mProgram, "fragOutput", 0);
    GL::LinkProgram(mProgram);
    GL::DumpLog(mProgram);

    // set uniforms
    float width = xe - xs;
    float height = ye - ys;

    if (width <= 0.0f || height <= 0.0f)
    {
        UTIL::ThrowException("Invalid input parameters", __FILE__, __LINE__);
    }

    glUseProgram(mProgram);
    GLint loc;
    loc = glGetUniformLocation(mProgram, "xs");
    glUniform1f(loc, xs);
    loc = glGetUniformLocation(mProgram, "ys");
    glUniform1f(loc, ys);
    loc = glGetUniformLocation(mProgram, "width");
    glUniform1f(loc, width);
    loc = glGetUniformLocation(mProgram, "height");
    glUniform1f(loc, height);
    loc = glGetUniformLocation(mProgram, "color");
    glUniform4f(loc, r, g, b, a);

    glGenVertexArrays(1, &mVAO);
    glBindVertexArray(mVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleSystem->GetPositionsVBO());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleSystem->GetPositionsVBO());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mParticleSystem->GetIndexVBO());
}
//-----------------------------------------------------------------------------
PointRenderer::~PointRenderer ()
{
    glDeleteVertexArrays(1, &mVAO);
    glDeleteProgram(mProgram);
}
//-----------------------------------------------------------------------------
void PointRenderer::Render () const
{    
    glUseProgram(mProgram);
    glBindVertexArray(mVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleSystem->GetPositionsVBO());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mParticleSystem->GetIndexVBO());
    //glDrawArrays(GL_POINTS, 0, mParticleSystem->GetNumParticles());
    glDrawElements(GL_POINTS, mParticleSystem->GetNumParticles(), 
        GL_UNSIGNED_INT, 0);
}
//-----------------------------------------------------------------------------