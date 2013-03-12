#include <iostream>
#include <cstdlib>
#include <memory>
#include "VideoWriter\VideoWriter.h"
#include "OpenGL\OpenGL.h"
#include "CUDA\cuda.h"
#include "ParticleSystem.h"
#include "PointRenderer.h"
#include "WCSPHSolver.h"

#define PI 3.14159265358979323846
#define WIDTH  800
#define HEIGHT 800

ParticleSystem* gFluidParticles;
ParticleSystem* gFluidParticlesHigh;
ParticleSystem* gBoundaryParticles;
PointRenderer* gFluidRenderer;
PointRenderer* gFluidRendererHigh;
PointRenderer* gBoundaryRenderer;
WCSPHSolver* gSolver;
static VideoWriter gsVideoWriter("video.avi", WIDTH, HEIGHT);

void display ();
void keyboard (unsigned char key, int x, int y);

void init ();
void init2 ();
void tearDown ();
void saveScreenshot (const std::string& filename);

int main (int argc, char* argv[])
{  
    cudaGLSetGLDevice(0); 
    glutInit(&argc, argv);
    glutInitContextVersion(3, 3);
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
    glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("");
	glewExperimental = TRUE;
	glewInit();
    init();
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, 
        GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutMainLoop();
    tearDown();

    return 0;
}

void display () 
{
    static int i = 0;

    gSolver->AdvanceTS();
    //gSolver->AdvanceHigh(); 
    
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    gFluidRenderer->Render();
    gFluidRendererHigh->Render();
    gBoundaryRenderer->Render();

    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();

    if (i % 5 == 0)
    {
        gsVideoWriter.CaptureFrame();
        //std::cout << i << std::endl;
    }

    i++;
    
}

void keyboard (unsigned char key, int x, int y)
{
    switch (key) 
    {
        case 's':
            saveScreenshot("s.bmp");
            return;
	    default:
		    return; 
	}
}

void init ()
{
    // Dam break simulation with about 15000 fluid particles
    try
    {
        // set configuration parameters
        float restDensity = 1000.0f;
        float fluidVol = 100.0f*0.0025f*150.0f*0.0025f;
        unsigned int numParticles = 101*151;
        float particleMass = fluidVol*restDensity/numParticles;
        float effectiveRadius = std::sqrtf(fluidVol*25.0f/(numParticles*PI));
        float effectiveRadiusHigh = 0.5f*effectiveRadius;
        float taitCoeff = 1119.0714f;
        float speedSound = 88.1472f;
        float alpha = 0.04f;
        float tensionCoefficient = 0.08f;
        float timeStep = 0.00030f;

        gFluidParticles = CreateParticleBox
        (
            0.105f, 0.105f, 0.0025f, 101, 151, 
            particleMass
        );     
        gFluidParticlesHigh = new ParticleSystem
        (
            gFluidParticles->GetMaxParticles()*4,
            gFluidParticles->GetMass()/4.0f
        );
/*        gFluidParticlesHigh = CreateParticleBox
        (
            0.705f, 0.105f, 0.00125f, 101, 151, 
            particleMass/4.0f
        ); */ 
        gBoundaryParticles = CreateParticleBoxCanvas
        (
            0.1f, 0.1f, 0.0025f,321, 321, 3, 3, 
            1000.0f*0.25f*0.25f/(101.0f*151.0f)
        );
        gFluidRenderer = new PointRenderer
        (
            *gFluidParticles, 0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.5f, 1.0f, 1.0f
        );
        gFluidRendererHigh = new PointRenderer
        (
            *gFluidParticlesHigh, 0.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 0.5f, 0.0f, 1.0f        
        );
        gBoundaryRenderer = new PointRenderer
        (
            *gBoundaryParticles, 0.0f, 0.0f, 
            1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f
        );
        WCSPHConfig config
        (   
            0.0f, 0.0f, 1.0f, 1.0f, effectiveRadius, 
            effectiveRadiusHigh, restDensity, taitCoeff, speedSound, 
            alpha, tensionCoefficient, timeStep
        );     
        gSolver = new WCSPHSolver
        (
            config, *gFluidParticles, *gFluidParticlesHigh,*gBoundaryParticles
        );
        gSolver->Bind();      
    }
    catch (const std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
    }
    
}



//void init2 ()
//{
//    // Dam break simulation with about 60000 fluid particles
//    try
//    {
//        float restDensity = 1000.0f;
//        float fluidVol = 200.0f*0.00125f*300.0f*0.00125f;
//        unsigned int numParticles = 201*301;
//        float particleMass = fluidVol*restDensity/numParticles;
//        float effectiveRadius = std::sqrtf(fluidVol*30.0f/(numParticles*PI));
//        float taitCoeff = 1119.0714f;
//        float speedSound = 88.1472;
//        float alpha = 0.046f;
//        float tensionCoefficient = 0.08f;
//        float timeStep = 0.0002f;
//
//        gFluidParticles = CreateParticleBox(0.105f, 0.105f, 0.00125f, 201, 301, 
//            particleMass);
//        
//        gBoundaryParticles = CreateParticleBoxCanvas(0.1f, 0.1f, 0.00125f, 
//            641, 641, 4, 4, particleMass);
//
//        gFluidRenderer = new PointRenderer(*gFluidParticles, 0.0f, 0.0f, 1.0f,
//            1.0f, 0.0f, 0.6f, 0.8f, 1.0f);
//        gBoundaryRenderer = new PointRenderer(*gBoundaryParticles, 0.0f, 0.0f, 
//            1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
//
//        
//        WCSPHConfig config(0.0f, 0.0f, 1.0f, 1.0f, effectiveRadius, restDensity, 
//            taitCoeff, speedSound, alpha, tensionCoefficient, timeStep);
//
//        
//        
//        gSolver = new WCSPHSolver(config, *gFluidParticles, *gBoundaryParticles);
//        gSolver->Bind();  
//    }
//    catch (const std::runtime_error& e)
//    {
//        std::cout << e.what() << std::endl;
//    }
//}


void tearDown ()
{
    delete gFluidParticles;
    delete gBoundaryParticles;
    delete gFluidParticlesHigh;
    delete gFluidRenderer;
    delete gFluidRendererHigh;
    delete gBoundaryRenderer;
    delete gSolver;
}

void saveScreenshot (const std::string& filename)
{
    gsVideoWriter.SaveScreenshot(filename);
}