//
//  main.cpp
//  graphenePFC
//
//  Created by Rachel Zucker on 7/8/16.
//

#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "matrix.hpp"
#include <fftw3.h>

#define PI 3.141592653589793
#define NR 2048
#define NC 2048

int main(int argc, const char * argv[]) {
    // define length scales
    double scaleUp, r0, a0;
    scaleUp = 3.57791754018;
    r0 = scaleUp * 2.12;
    a0 = scaleUp * 1.7321;
    double externalPotentialAmplitude = 1.0;
    
    // define appled potential
    Matrix<NR, NC> appliedPotential (r0, externalPotentialAmplitude, 0.0, 1.0);

    // begin initial condition
    // make AB and AC matrices
    double shiftAC = r0 * sqrt(3.) / 2.;
    Matrix<NR, NC> matrixAB (r0, 3.0, 2.0 * shiftAC, 1.0);
    Matrix<NR, NC> matrixAC (r0, 3.0, shiftAC, 1.0);
    
    // define circular zone
    double radialDistance;
    double domainCenter [2] = {NR/2., NC/2.};
    double maxRadius;
    if (NR < NC)
    {
        maxRadius = NR / 2.0;
    }
    else
        maxRadius = NC / 2.0;
 
    // make initial condition
    Matrix<NR, NC> initialCondition (0.0);

    for (int ir=0; ir<NR; ++ir )
    {
        for (int ic=0; ic<NC; ++ic )
        {
            radialDistance = sqrt((ir - domainCenter[0]) * (ir - domainCenter[0]) + (ic - domainCenter[1]) * (ic - domainCenter[1]));
            if (radialDistance < maxRadius * 0.5) {
                initialCondition.set(ir, ic, matrixAC.get(ir, ic));
            }
            else if (radialDistance < maxRadius * 0.7) {
                initialCondition.set(ir, ic, (rand() % 1000) / 1000. - 0.5);
            }
            else
                initialCondition.set(ir, ic, matrixAB.get(ir, ic));
        }
    }
    
    initialCondition.writeToFile("/Users/Rachel/Documents/graphenePFC/initalCondition.txt");
    return 0;
}

