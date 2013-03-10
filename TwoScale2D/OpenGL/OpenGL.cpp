//-----------------------------------------------------------------------------
//  OpenGL.cpp
//  SPHFLUIDS2D
//
//  Created by Arno in Wolde Lübke on 29.01.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------

#include "stdio.h"
#include "stdlib.h"
#include "OpenGL.h"

//-----------------------------------------------------------------------------
static char* readFile(const char* filename);
//-----------------------------------------------------------------------------
void GL::AttachShader(GLuint program, const char* filename, GLenum type) 
{
	GLuint shader;
	GLint hasCompiled;
	char* fileContents;	
    
	if (program == 0) 
    {
	    printf("Invalid program handle.");
		return;
	}
    
	// read contents froms file 
	fileContents = readFile(filename);
    
	if (fileContents == NULL) 
    {
	    printf("File not found.");
		return;
	}
    
	// create shader 
	shader = glCreateShader(type);
    
	if (shader == 0) 
    {
		printf("Could not create shader");
		free(fileContents);
		return;
	}
	
	// attach source code to shader object 
	glShaderSource(shader, 1, (const GLchar**) &fileContents, NULL);
	
	// compile shader and check for errors 
	glCompileShader(shader);
    free(fileContents);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &hasCompiled);
    
    if (!hasCompiled) {
		printf("Could not compile shader");
		glDeleteShader(shader);
		return;
	}
    
	// attach shader to the program 
	glAttachShader(program, shader);
}
//-----------------------------------------------------------------------------
void GL::BindAttribLocation(GLuint program, const char* attrName, 
                              GLuint index) 
{
	glBindAttribLocation(program, index, attrName);
}
//-----------------------------------------------------------------------------
void GL::BindFragDataLocation(GLuint program, const char* szColorBufName,
                                GLuint index) 
{
	glBindFragDataLocation(program, index, szColorBufName);
}
//-----------------------------------------------------------------------------
void GL::LinkProgram(GLuint program)
{
	GLint hasLinked;
	
	glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &hasLinked);
    
    if (!hasLinked) 
    {
		printf("Could not link program.");
		return;
    }
}
//-----------------------------------------------------------------------------
void GL::DumpLog(GLuint program) 
{
	GLint logLength;
	char* log;
	
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
	log = (char*) malloc((logLength + 1) * sizeof(char));
	glGetProgramInfoLog(program, logLength, NULL, log);
	log[logLength] = '\0';
	
	printf("%s\n", log);
	
	free(log); 
}
//-----------------------------------------------------------------------------
void GL::CreateBufferObject (GLuint& buffer, GLenum target, 
    GLsizeiptr size, const GLvoid* data, GLenum usage)
{
    glGenBuffers(1, &buffer);
    glBindBuffer(target, buffer);
    glBufferData(target, size, data, usage);
}
//-----------------------------------------------------------------------------
char* readFile(const char* filename) 
{
    FILE* pFile;
    long fileSize = 0;
    char* pContents;
    char ch;
    
    pFile = fopen(filename, "r");
    
    if (pFile == NULL) 
    {
        return NULL;
    }
    
    while (1) 
    {
        ch = fgetc(pFile);
        if (ch == EOF)
            break;
        ++fileSize;
    }
	rewind (pFile);
    pContents = (char*) malloc((fileSize + 1)*sizeof(char));
    
    if (pContents == NULL) 
    {
        fclose(pFile);
        return NULL;
    }
    
    fread(pContents, sizeof(char), fileSize, pFile);
    pContents[fileSize] = '\0';
    fclose(pFile);
    
    return pContents;
}
//-----------------------------------------------------------------------------
