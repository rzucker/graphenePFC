//
//  graphene_functions.hpp
//  graphenePFC
//
//  Created by Rachel Zucker on 8/15/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#ifndef graphene_functions_hpp
#define graphene_functions_hpp

#include "matrix_types.hpp"

// an inital condition that produces an AC circle inside an AB matrix
void InitialCircle(matrix_t* n_mat, const double r0);
// an inital condition that produces stripes
void InitialShallowStripes(matrix_t* n_mat, const double r0);
void InitialSteepStripes(matrix_t* n_mat, const double r0);
void InitialDiagonalStripes(matrix_t* n_mat, const double r0);
void InitialVerticalStripes(matrix_t* n_mat, const double r0);
void InitialHorizontalStripes(matrix_t* n_mat, const double r0);
void InitialHorizontalDoubleStripes(matrix_t* n_mat, const double r0);
void InitialSmoothStripes(matrix_t* n_mat, const double r0, const double degrees);
void InitialSmoothStripesVert(matrix_t* n_mat, const double r0, const double degrees);
// an inital condition to make a perfect "A" sheet
void InitialAA(matrix_t* n_mat, const double r0);

// a function that can add a constant shift to a matrix, useful for conserved dynamics
void AddConstToMatrix(matrix_t* n_mat, const double amount);

// a function that produces the fourier-space vectors
// in cylindrical coordinates
void MakeKs(matrix_t* k_r, matrix_t* k_th);

// a function that produces the fourier transform of the coefficients
// C2, Cs1, and Cs2 from Seymour's paper
void MakeCs(matrix_t* c2, matrix_t* cs1, matrix_t* cs2, matrix_t* minus_k_squared, const double r0, const double a0);

// a function that writes a matrix to a file in the directory directoryString
// with the first line "the time is (time)" and a filename of the iteration number
void WriteMatrix(const double time, const double energy_val, const std::string directory_string, const matrix_t& mat);

void AFMTip(matrix_t* total_potential, const double radius, const double speed, const double force, const double angle, const double time, const double delay, const matrix_t& fixed_potential);

#endif /* graphene_functions_hpp */
