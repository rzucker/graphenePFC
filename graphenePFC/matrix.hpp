//
//  matrix.hpp
//  graphenePFC
//
//  Created by Rachel Zucker on 8/7/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#ifndef matrix_hpp
#define matrix_hpp
#include <vector>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <fstream>
#define PI 3.141592653589793

template <int R, int C>
class Matrix {
public:
    //Constructors
    
    Matrix(double init) {
        _storage.assign(R*C, init);
    }
    
    Matrix(double r0, double potentialAmplitude, double potentialShift, double potentialStretch) {
        _storage.reserve(R*C);
        // define helper numbers
        double cmax [] = {2.75075, 3.349208, 8258.11};
        double cmin [] = {1.63093, 3.347616, 8184.70};
        double atomicSpacing, zEq;
        atomicSpacing = 1.5 * r0 * potentialStretch;
        zEq = 3.31;
        double cixy [3];
        int i, ir, ic;
        double minPotential, maxPotential;
        minPotential = -68.7968;
        maxPotential = -68.1889;
        // iterate over the matrix, storing values
        for ( ir=0; ir<R; ++ir )
        {
            for ( ic=0; ic<C; ++ic )
            {
                for ( i=0; i<3; ++i )
                {
                    cixy[i] = cmax[i] - (cmax[i] - cmin[i]) * (2./9.) * (3. - (2. * cos(2 * PI * ir / atomicSpacing) * cos(2. * PI * (ic - potentialShift) / (sqrt(3.) * atomicSpacing)) + cos(4. * PI * (ic - potentialShift) / (sqrt(3.) * atomicSpacing)) ));
                }
                _storage[ir*C + ic] = potentialAmplitude * (((cixy[0] * exp(-zEq * cixy[1]) - cixy[2] / (zEq*zEq*zEq*zEq)) - minPotential) / (maxPotential - minPotential) - 0.5);
            }
        }
    }
    
    //Methods
    
    void set (int ir, int ic, double value) {
        _storage[ir * C + ic] = value;
    }
    
    double get (int ir, int ic) const {
        return _storage[ir * C + ic];
    }
    
    void writeToFile (const char* fileName) const {
        std::ofstream file (fileName);
        if (file.is_open())
        {
            file << "{";
            for (int ir=0; ir<R-1; ++ir )
            {
                file << " {";
                for (int ic=0; ic<C-1; ++ic )
                {
                    file << _storage[(ir * C) + ic] << ",";
                }
                file << _storage[(ir * C) + C-1] << "}," << std::endl;
            }
            file << " {";
            for (int ic=0; ic<C-1; ++ic )
            {
                file << _storage[((R-1) * C) + ic] << ",";
            }
            file << _storage[((R-1) * C) + C-1] << "}" << std::endl;
            file << "}";
            file.close();
            std::cout << "done writing matrix to file" << std::endl;
        }
        else std::cout << "Unable to open file" << std::endl;
    }
    
private:
    std::vector<double> _storage;
    // maybe replace this with a fixed size array on the heap, rather than dynamically sized. Use new and *'s?
};

 
#endif /* matrix_hpp */