//-----------------------------------------------------------------------------
//  PointRendererFragmentShader.glsl
//  SPHFLUIDS2D
//
//  Created by Arno in Wolde Lübke on 30.01.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#version 330 core

uniform float xs;
uniform float ys;
uniform float width;
uniform float height;
uniform vec4 color;

in GeometryData
{
    vec2 relCoord;
}
gGeometryData;

out vec4 fragOutput;

void main() 
{
    float x = gGeometryData.relCoord.x;
    float y = gGeometryData.relCoord.y;

    if (x*x + y*y > 1.0f)
    {
        discard;
    }

    fragOutput = color;
}