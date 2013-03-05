//-----------------------------------------------------------------------------
//  cuda.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 12.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once
#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>    
#include <stdexcept>
#include <string>
#include <sstream>


#define CUDA_SAFE_CALL(x) CUDA::SafeCall(x, __FILE__, __LINE__)

namespace CUDA
{

inline void SafeCall (cudaError_t err, const char* filename, unsigned int line)
{
    using namespace std;

    if (err == cudaSuccess)
    {
        return;
    }    
    stringstream str(stringstream::in | stringstream::out);

    str << "Exception thrown in FILE: " << filename << " LINE: " << line << endl;
    str << "Error Message: " << cudaGetErrorString(cudaGetLastError()) << endl;
    
    throw runtime_error(str.str());
}

template<typename T>
inline void SafeFree (T** ptr ) 
{
    if (*ptr != NULL) {
        cudaFree(*ptr);
        *ptr = NULL;
    }
}

template<typename T> 
inline void DumpArray (T* arr, unsigned int numElements, unsigned int offset = 0, 
    unsigned int stride = 1, unsigned int pauseAfter = 0)
{
    T* hostData = new T[numElements];
    
    CUDA_SAFE_CALL( cudaMemcpy(hostData, arr, sizeof(T)*numElements,
        cudaMemcpyDeviceToHost) );
    
    for (unsigned int i = 0; i < numElements; i++)
    {
        std::cout << i << " " << hostData[i*stride + offset] << std::endl;

        if (pauseAfter != 0)
        {
            if ((i % pauseAfter) == 0)
            {
                system("pause");
            }
        }
    }
    
    delete[] hostData;
}


class Timer
{
    enum 
    {
        CT_TAKING_TIME,
        CT_STOPPED
    };

public:
    Timer();
    ~Timer();
    void Start();
    void Stop();
    void DumpElapsed() const;

private:
    cudaEvent_t mStart;
    cudaEvent_t mStop;
    float mTime;
    int mState;
};


}
