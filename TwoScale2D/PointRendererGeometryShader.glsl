//-----------------------------------------------------------------------------
//  PointRendererGeometryShader.glsl
//  SPHFLUIDS2D
//
//  Created by Arno in Wolde Lübke on 04.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#version 330 core

uniform float xs;
uniform float ys;
uniform float width;
uniform float height;

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

out GeometryData
{
    vec2 relCoord;
}
gGeometryData;

void main() 
{
    float dx = 0.002f;
    
    gl_Position = gl_in[0].gl_Position + vec4(dx, dx, 0.0f, 0.0f);
    gGeometryData.relCoord = vec2(1.0f, 1.0f);
    EmitVertex();
    
    gl_Position = gl_in[0].gl_Position + vec4(-dx, dx, 0.0f, 0.0f);
    gGeometryData.relCoord = vec2(-1.0f, 1.0f);
    EmitVertex();
    
    gl_Position = gl_in[0].gl_Position + vec4(dx, -dx, 0.0f, 0.0f);
    gGeometryData.relCoord = vec2(1.0f, -1.0f);
    EmitVertex();
    
    gl_Position = gl_in[0].gl_Position + vec4(-dx, -dx, 0.0f, 0.0f);
    gGeometryData.relCoord = vec2(-1.0f, -1.0f);
    EmitVertex();
    
    EndPrimitive();
}