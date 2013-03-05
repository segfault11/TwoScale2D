//-----------------------------------------------------------------------------
//  PointRendererVertexShader.glsl
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

in vec2 position;

void main()
{
    float x = (position.x - xs)/width*2.0f - 1.0f;
    float y = (position.y - ys)/height*2.0f - 1.0f;
    gl_Position = vec4(x, y, 0.0f, 1.0f);
}
