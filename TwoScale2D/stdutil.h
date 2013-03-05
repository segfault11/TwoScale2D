//-----------------------------------------------------------------------------
//  ParticleSystem.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 12.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

#define DECL_DEFAULTS(x) x ();\
                         x (const x& orig);\
                         x& operator= (const x& rhs);


namespace UTIL
{

inline void ThrowException (const char* message, const char* filename, 
    unsigned int line)
{
    using namespace std;

    stringstream str(stringstream::in | stringstream::out);

    str << "Exception thrown in FILE: " << filename << " LINE: " << line << endl;
    str << "Error Message: " << message << endl;

    throw runtime_error(str.str());
}

template<typename T>
inline void SaveDelete (T** ptr)
{
    if (*ptr != NULL) {
        delete *ptr;
        *ptr = NULL;
    }
}

template<typename T>
inline void SaveDeleteArray (T** ptr)
{
    if (*ptr != NULL) {
        delete *ptr;
        *ptr = NULL;
    }
}

}