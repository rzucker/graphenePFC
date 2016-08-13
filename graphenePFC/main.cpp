//
//  main.cpp
//  graphenePFC
//
//  Created by Rachel Zucker on 7/8/16.
//

#include <iostream>
#include <complex>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "matrix.hpp"
#include <boost/filesystem.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <fftw3.h>

#define PI 3.141592653589793
#define NR 128
#define NC 128

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
    Matrix<NR, NC> matrixAB (r0, 1.0, 2.0 * shiftAC, 1.0);
    Matrix<NR, NC> matrixAC (r0, 1.0, shiftAC, 1.0);
    
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
    Matrix<NR, NC> n_mat (0.0);

    for (int ir=0; ir<NR; ++ir )
    {
        for (int ic=0; ic<NC; ++ic )
        {
            radialDistance = sqrt((ir - domainCenter[0]) * (ir - domainCenter[0]) + (ic - domainCenter[1]) * (ic - domainCenter[1]));
            if (radialDistance < maxRadius * 0.5) {
                n_mat.set(ir, ic, matrixAC.get(ir, ic));
            }
            else if (radialDistance < maxRadius * 0.7) {
                n_mat.set(ir, ic, (rand() % 2000) / 1000. - 1.0);
            }
            else
                n_mat.set(ir, ic, matrixAB.get(ir, ic));
        }
    }
    
    // define cartesian k-vectors, same as in mathematica, with origin at (0,0) (not matrix center)
    double k_row [NR];
    double k_col [NC];
    double tmp;
    for (int ir=0; ir < NR; ++ir) {
        if (ir > NR/2.0) {
            tmp = (1.0 * ir) / NR - 1.0;
        }
        else {
            tmp = (1.0 * ir) / NR;
        }
        k_row[ir] = 2.0 * PI * tmp;
    }
    for (int ic=0; ic < NC; ++ic) {
        if (ic > NC/2.0) {
            tmp = (1.0 * ic) / NC - 1.0;
        }
        else {
            tmp = (1.0 * ic) / NC;
        }
        k_col[ic] = 2.0 * PI * tmp;
    }
    
    // define magnitude of k matrix = k_r and angular k matrix = k_th
    Matrix<NR, NC> k_r (0.0);
    Matrix<NR, NC> k_th (0.0);
    for (int ir=0; ir < NR; ++ir) {
        for (int ic=0; ic < NC; ++ic) {
            k_r.set(ir, ic, sqrt( k_row[ir] * k_row[ir] + k_col[ic] * k_col[ic] ) );
            if (ir == 0 && ic == 0) {
                k_th.set(0, 0, 0.);
            }
            else {
                tmp = atan2 (k_row[ir], k_col[ic]);
                if (tmp < 0.) {
                    k_th.set(ir, ic, tmp + (2.0 * PI));
                }
                else k_th.set(ir, ic, tmp);
            }
        }
    }
    
    // make the C2 matrix for 2-point correlations
    Matrix<NR, NC> c2hat (0.0);
    for (int ir=0; ir < NR; ++ir) {
        for (int ic=0; ic < NC; ++ic) {
            if (ir == 0 && ic == 0) {
                c2hat.set(0, 0, -6.);
            }
            else {
                tmp = boost::math::cyl_bessel_j(1, r0 * k_r.get(ir, ic)) / (r0 * k_r.get(ir, ic));
                if (fabs(tmp) < pow(10, -14)) {
                    tmp = 0.0;
                }
                c2hat.set(ir, ic, -12.0 * tmp);
            }
        }
    }
    // make the Cs1, Cs2 matrices for 3-point correlations
    // Cs1 and Cs2 still need to multiplied by i! No real component.
    Matrix<NR, NC> cs1hat (0.0);
    for (int ir=0; ir < NR; ++ir) {
        for (int ic=0; ic < NC; ++ic) {
            tmp = -2.5 * cos(3. * k_th.get(ir, ic)) * boost::math::cyl_bessel_j(3, a0 * k_r.get(ir, ic));
            if (fabs(tmp) < pow(10, -14)) {
                tmp = 0.0;
            }
            cs1hat.set(ir, ic, tmp);
        }
    }
    Matrix<NR, NC> cs2hat (0.0);
    for (int ir=0; ir < NR; ++ir) {
        for (int ic=0; ic < NC; ++ic) {
            tmp = -2.5 * sin(3. * k_th.get(ir, ic)) * boost::math::cyl_bessel_j(3, a0 * k_r.get(ir, ic));
            if (fabs(tmp) < pow(10, -14)) {
                tmp = 0.0;
            }
            cs2hat.set(ir, ic, tmp);
        }
    }

    // outside the time loop: allocation, plans
    
    double rescaleFFTs = 1.0 / sqrt(NR * NC * 1.0);
    double tmp1r, tmp2r;
    fftw_complex *n_hat, *dfdn2;
    // allocate input 2d array to be FFT'd, will be overwritten
    n_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NR * NC);
    dfdn2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NR * NC);
    
    fftw_complex *cs1n, *cs2n, *ncs1n, *ncs2n, *cs1ncs1n, *cs2ncs2n;
    cs1n = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NR * NC);
    cs2n = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NR * NC);
    ncs1n = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NR * NC);
    ncs2n = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NR * NC);
    cs1ncs1n = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NR * NC);
    cs2ncs2n = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NR * NC);
    
    // create a FFT plan, size NR x NC, input, output destination, direction of FFT,
    // and let FFTW guess what the best method is (fastest option, rather
    // than doing an exhaustive search)
    fftw_plan makeNHat, makeDfdn2, makecs1n, makecs2n, makencs1n, makencs2n, makecs1ncs1n, makecs2ncs2n;
    makeNHat = fftw_plan_dft_2d(NR, NC, n_hat, n_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
    makeDfdn2 = fftw_plan_dft_2d(NR, NC, dfdn2, dfdn2, FFTW_FORWARD, FFTW_ESTIMATE);
    
    makecs1n = fftw_plan_dft_2d(NR, NC, cs1n, cs1n, FFTW_FORWARD, FFTW_ESTIMATE);
    makecs2n = fftw_plan_dft_2d(NR, NC, cs2n, cs2n, FFTW_FORWARD, FFTW_ESTIMATE);
    makencs1n = fftw_plan_dft_2d(NR, NC, ncs1n, ncs1n, FFTW_BACKWARD, FFTW_ESTIMATE);
    makencs2n = fftw_plan_dft_2d(NR, NC, ncs2n, ncs2n, FFTW_BACKWARD, FFTW_ESTIMATE);
    makecs1ncs1n = fftw_plan_dft_2d(NR, NC, cs1ncs1n, cs1ncs1n, FFTW_FORWARD, FFTW_ESTIMATE);
    makecs2ncs2n = fftw_plan_dft_2d(NR, NC, cs2ncs2n, cs2ncs2n, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // make space for the three terms of dF/dN
    Matrix<NR, NC> dFdN1 (0.0);
    Matrix<NR, NC> dFdN2 (0.0);
    Matrix<NR, NC> dFdN3 (0.0);
    // create time variable
    // double time = 0.0;
    
    
    // begin time iteration
    
    // make dFdN1
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            tmp = n_mat.get(ir, ic);
            dFdN1.set(ir, ic, tmp - (0.5 * tmp * tmp) + (0.33333333333 * tmp * tmp *tmp));
        }
    }

    // define n_hat using current n_mat value
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            n_hat[ir * NC + ic][0] = n_mat.get(ir, ic);
            n_hat[ir * NC + ic][1] = 0.0;
        }
    }
    fftw_execute(makeNHat);
    
    // define dfdn2 using current n_hat value
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            tmp = c2hat.get(ir, ic);
            dfdn2[ir * NC + ic][0] = rescaleFFTs * n_hat[ir * NC + ic][0] * tmp;
            dfdn2[ir * NC + ic][1] = rescaleFFTs * n_hat[ir * NC + ic][1] * tmp;
        }
    }
    fftw_execute(makeDfdn2);
    // make dFdN2
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            dFdN2.set(ir, ic, -1.0 * dfdn2[ir * NC + ic][0]);
        }
    }
    
    // define IFFT[ Cs1 nhat]
    // (a + b i)(d i) = (-b d) + (a d) i
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            tmp = cs1hat.get(ir, ic);
            cs1n[ir * NC + ic][0] = -1.0 * rescaleFFTs * n_hat[ir * NC + ic][1] * tmp;
            cs1n[ir * NC + ic][1] = rescaleFFTs * n_hat[ir * NC + ic][0] * tmp;
        }
    }
    // define IFFT[ Cs2 nhat]
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            tmp = cs2hat.get(ir, ic);
            cs2n[ir * NC + ic][0] = -1.0 * rescaleFFTs * n_hat[ir * NC + ic][1] * tmp;
            cs2n[ir * NC + ic][1] = rescaleFFTs * n_hat[ir * NC + ic][0] * tmp;
        }
    }
    fftw_execute(makecs1n);
    fftw_execute(makecs2n);
    
    // define FFT[ n IFFT[Cs1 nhat]]
    // define FFT[ n IFFT[Cs2 nhat]]
    // (a)(c + d i) = (a c) + (a d) i
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            tmp = n_mat.get(ir, ic);
            ncs1n[ir * NC + ic][0] = rescaleFFTs * cs1n[ir * NC + ic][0] * tmp;
            ncs1n[ir * NC + ic][1] = rescaleFFTs * cs1n[ir * NC + ic][1] * tmp;
            ncs2n[ir * NC + ic][0] = rescaleFFTs * cs2n[ir * NC + ic][0] * tmp;
            ncs2n[ir * NC + ic][1] = rescaleFFTs * cs2n[ir * NC + ic][1] * tmp;
        }
    }
    fftw_execute(makencs1n);
    fftw_execute(makencs2n);
    
    // define IFFT[ Cs1 FFT[ n IFFT[ Cs1 nhat]]]
    // (a + b i)(d i) = (-b d) + (a d) i
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            tmp = cs1hat.get(ir, ic);
            cs1ncs1n[ir * NC + ic][0] = -1.0 * rescaleFFTs * ncs1n[ir * NC + ic][1] * tmp;
            cs1ncs1n[ir * NC + ic][1] = rescaleFFTs * ncs1n[ir * NC + ic][0] * tmp;
        }
    }
    // define IFFT[ Cs2 FFT[ n IFFT[ Cs2 nhat]]]
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            tmp = cs2hat.get(ir, ic);
            cs2ncs2n[ir * NC + ic][0] = -1.0 * rescaleFFTs * ncs2n[ir * NC + ic][1] * tmp;
            cs2ncs2n[ir * NC + ic][1] = rescaleFFTs * ncs2n[ir * NC + ic][0] * tmp;
        }
    }
    fftw_execute(makecs1ncs1n);
    fftw_execute(makecs2ncs2n);
    
    // make dFdN3
    for(int ir = 0; ir < NR; ++ir) {
        for(int ic = 0; ic < NC; ++ic) {
            tmp1r = (cs1n[ir * NC + ic][0] * cs1n[ir * NC + ic][0]) - (cs1n[ir * NC + ic][1] * cs1n[ir * NC + ic][1]);
            tmp2r = (cs2n[ir * NC + ic][0] * cs2n[ir * NC + ic][0]) - (cs2n[ir * NC + ic][1] * cs2n[ir * NC + ic][1]);
            tmp = rescaleFFTs * rescaleFFTs * (tmp1r + tmp2r) - 2.0 * rescaleFFTs * cs1ncs1n[ir * NC + ic][0] - 2.0 * rescaleFFTs * cs2ncs2n[ir * NC + ic][0];
            dFdN3.set(ir, ic, tmp);
        }
    }
    
    dFdN3.writeToFile("/Users/Rachel/Documents/graphenePFC/dfdn3.txt");
    
    // end time iteration
    
    fftw_destroy_plan(makeNHat);
    fftw_destroy_plan(makeDfdn2);
    fftw_free(n_hat);
    fftw_free(dfdn2);
    
    return 0;
}

