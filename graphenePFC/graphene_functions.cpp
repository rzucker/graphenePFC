//
//  graphene_functions.cpp
//  graphenePFC
//
//  Created by Rachel Zucker on 8/15/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#include "graphene_functions.hpp"
#include "matrix_types.hpp"
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <omp.h>

#define PI 3.141592653589793

// an inital condition that produces an AC circle inside an AB matrix
void InitialCircle(matrix_t* n_mat, const double r0) {
   
   // Seed the random number to a constant for repeatability.
   srand(0);
   // make AB and AC matrices
   // convention:
   // if the holeis over a hole: AA stacking
   // if the the hole is over a down site: AC stacking
   // if the hole is over an up site: AB stacking
   double shift_ac = r0 * sqrt(3.) / 2.;
   matrix_t perfect_ab(r0, 3.0, 2.0 * shift_ac, 1.0, 0.0);
   matrix_t perfect_ac(r0, 3.0, shift_ac, 1.0, 0.0);
   
   // define circular zone
   const double domain_center[2] = {NR / 2., NC / 2.};
   const double max_radius = std::min(NC, NR) / 2.0;
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         double radial_distance = sqrt((ir - domain_center[0]) * (ir - domain_center[0]) +
                               (ic - domain_center[1]) * (ic - domain_center[1]));
         if (radial_distance < max_radius * 0.35) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (radial_distance < max_radius * 0.40) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         }
      }
   }
}

/*
void InitialCircle(matrix_t* n_mat, const double r0, const double degrees=0.0) {
   
   // Seed the random number to a constant for repeatability.
   srand(0);
   // make AB and AC matrices
   // convention:
   // if the holeis over a hole: AA stacking
   // if the the hole is over a down site: AC stacking
   // if the hole is over an up site: AB stacking
   double shift_ac = r0 * sqrt(3.) / 2.;
   matrix_t perfect_ab(r0, 3.0, 2.0 * shift_ac, 1.0, 0.0);
   matrix_t perfect_ac(r0, 3.0, shift_ac, 1.0, 0.0, degrees);
   
   // define circular zone
   const double domain_center[2] = {NR / 2., NC / 2.};
   const double max_radius = std::min(NC, NR) / 2.0;
   
#pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
#pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         double radial_distance = sqrt((ir - domain_center[0]) * (ir - domain_center[0]) +
                                       (ic - domain_center[1]) * (ic - domain_center[1])
                                       );
         if (radial_distance < max_radius * 0.35) {
            (*n_mat).set(ir, ic, perfect_rot.get(ir, ic));
         } else if (radial_distance < max_radius * 0.4) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         }
      }
   }
}
*/

void InitialShallowStripes(matrix_t* n_mat, const double r0) {

   srand(0);
   double shift_ac = r0 * sqrt(3.) / 2.;
   matrix_t perfect_ab(r0, 3.0, 2.0 * shift_ac, 1.0, 0.0);
   matrix_t perfect_ac(r0, 3.0, shift_ac, 1.0, 0.0);
   
   double intercepts [] = {0.122205, 0.127795, 0.372205, 0.377795, 0.622205, 0.627795, \
0.872205, 0.877795, 1.1222, 1.1278, 1.3722, 1.3778}; // corrected, width = 0.005
   
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         if (ir < (-0.5) * ic + intercepts[0] * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-0.5) * ic + intercepts[1] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + intercepts[2] * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-0.5) * ic + intercepts[3] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + intercepts[4] * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-0.5) * ic + intercepts[5] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + intercepts[6] * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-0.5) * ic + intercepts[7] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + intercepts[8] * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-0.5) * ic + intercepts[9] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + intercepts[10] * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-0.5) * ic + intercepts[11] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         }
      }
   }
}

void InitialSteepStripes(matrix_t* n_mat, const double r0) {
   
   srand(0);
   double shift_ac = r0 * sqrt(3.) / 2.;
   matrix_t perfect_ab(r0, 3.0, 2.0 * shift_ac, 1.0, 0.0);
   matrix_t perfect_ac(r0, 3.0, shift_ac, 1.0, 0.0);
   
   double intercepts [] = {0.24441, 0.25559, 0.74441, 0.75559, 1.24441, 1.25559, 1.74441, \
1.75559, 2.24441, 2.25559, 2.74441, 2.75559}; // corrected, width = 0.005
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         if (ir < (-2.0) * ic + intercepts[0] * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-2.0) * ic + intercepts[1] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + intercepts[2] * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-2.0) * ic + intercepts[3] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + intercepts[4] * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-2.0) * ic + intercepts[5] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + intercepts[6] * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-2.0) * ic + intercepts[7] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + intercepts[8] * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-2.0) * ic + intercepts[9] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + intercepts[10] * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-2.0) * ic + intercepts[11] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         }
      }
   }
}

void InitialDiagonalStripes(matrix_t* n_mat, const double r0) {
   
   srand(0);
   double shift_ac = r0 * sqrt(3.) / 2.;
   matrix_t perfect_ab(r0, 3.0, 2.0 * shift_ac, 1.0, 0.0);
   matrix_t perfect_ac(r0, 3.0, shift_ac, 1.0, 0.0);
   
   double intercepts [] = {0.246464, 0.253536, 0.746464, 0.753536, 1.24646, 1.25354, 1.74646, \
1.75354}; // corrected, width = 0.005
   
    #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         if (ir < (-1.0) * ic + intercepts[0] * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-1.0) * ic + intercepts[1] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-1.0) * ic + intercepts[2] * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-1.0) * ic + intercepts[3] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-1.0) * ic + intercepts[4] * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-1.0) * ic + intercepts[5] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-1.0) * ic + intercepts[6] * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-1.0) * ic + intercepts[7] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         }
      }
   }
}

void InitialVerticalStripes(matrix_t* n_mat, const double r0) {
   
   srand(0);
   double shift_ac = r0 * sqrt(3.) / 2.;
   matrix_t perfect_ab(r0, 3.0, 2.0 * shift_ac, 1.0, 0.0);
   matrix_t perfect_ac(r0, 3.0, shift_ac, 1.0, 0.0);
   
   double intercepts [] = {0.2475, 0.2525, 0.7475, 0.7525};
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         if (ic < intercepts[0] * NC) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ic < intercepts[1] * NC) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ic < intercepts[2] * NC) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ic < intercepts[3] * NC) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         }
      }
   }
}

void InitialHorizontalStripes(matrix_t* n_mat, const double r0) {
   
   srand(0);
   double shift_ac = r0 * sqrt(3.) / 2.;
   matrix_t perfect_ab(r0, 3.0, 2.0 * shift_ac, 1.0, 0.0);
   matrix_t perfect_ac(r0, 3.0, shift_ac, 1.0, 0.0);
   
   double intercepts [] = {0.2475, 0.2525, 0.7475, 0.7525};
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         if (ir < intercepts[0] * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < intercepts[1] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < intercepts[2] * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < intercepts[3] * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         }
      }
   }
}

void InitialSmoothStripes(matrix_t* n_mat, const double r0, const double degrees) {
   
   // define helper numbers
   double const shift_ac = r0 * sqrt(3.) / 2.;
   double potential_shift = 2.0 * shift_ac; // start with AB
   double const intercepts [] = {0.2, 0.3, 0.7, 0.8};
   
   // repeated from matrix class constructor
   const double potential_amplitude = 3.0;
   const double atomic_spacing = 1.5 * r0;
   // values and functional form from DFT paper, Regguzoni et al.
   const double c_max[] = {2.75075, 3.349208, 8258.11};
   const double c_min[] = {1.63093, 3.347616, 8184.70};
   const double z_eq = 3.31;
   double const min_potential = -68.7968;
   double const max_potential = -68.1889;
   
   double potential_function_terms[3];
   double potential_value;
   // iterate over the matrix, storing values, gradually increasing shift
   for (int ir = 0; ir < NR; ++ir) {
      for (int ic = 0; ic < NC; ++ic) {

         double ic_rot = ic * cos(degrees * PI / 180.) - ir * sin(degrees * PI / 180.);
         double ir_rot = ir * cos(degrees * PI / 180.) + ic * sin(degrees * PI / 180.);
         
         if (ir < intercepts[0] * NR) {
            potential_shift = 2.0 * shift_ac;
         } else if (ir < intercepts[1] * NR) {
            potential_shift = (2.0 * shift_ac) - shift_ac * (ir - (NR * intercepts[0])) / (NR * (intercepts[1] - intercepts[0]));
         } else if (ir < intercepts[2] * NR) {
            potential_shift = shift_ac;
         } else if (ir < intercepts[3] * NR) {
            potential_shift = shift_ac + shift_ac * (ir - (NR * intercepts[2])) / (NR * (intercepts[3] - intercepts[2]));
         } else {
            potential_shift = 2.0 * shift_ac;
         }
         
         for (int i = 0; i < 3; ++i) {
            potential_function_terms[i] =
            c_max[i] -
            (c_max[i] - c_min[i]) * (2. / 9.) *
            (3. - (2. * cos(2 * PI * ic_rot / atomic_spacing) *
                   cos(2. * PI * (ir_rot - potential_shift) /
                       (sqrt(3.) * atomic_spacing)) +
                   cos(4. * PI * (ir_rot - potential_shift) /
                       (sqrt(3.) * atomic_spacing))));
         }
         
         potential_value =
         (potential_amplitude *
          (((potential_function_terms[0] *
             exp(-z_eq * potential_function_terms[1]) -
             potential_function_terms[2] / (z_eq * z_eq * z_eq * z_eq)) -
            min_potential) /
           (max_potential - min_potential) -
           0.5));
         
         (*n_mat).set(ir, ic, potential_value);
      }
   }
}


void InitialSmoothStripesVert(matrix_t* n_mat, const double r0, const double degrees) {
   
   // define helper numbers
   double const shift_ac = r0 * sqrt(3.) / 2.;
   double potential_shift = 2.0 * shift_ac; // start with AB
   double const intercepts [] = {0.2, 0.3, 0.7, 0.8};
   
   // repeated from matrix class constructor
   const double potential_amplitude = 3.0;
   const double atomic_spacing = 1.5 * r0;
   // values and functional form from DFT paper, Regguzoni et al.
   const double c_max[] = {2.75075, 3.349208, 8258.11};
   const double c_min[] = {1.63093, 3.347616, 8184.70};
   const double z_eq = 3.31;
   double const min_potential = -68.7968;
   double const max_potential = -68.1889;
   
   double potential_function_terms[3];
   double potential_value;
   // iterate over the matrix, storing values, gradually increasing shift
   for (int ir = 0; ir < NR; ++ir) {
      for (int ic = 0; ic < NC; ++ic) {
         
         double ic_rot = ic * cos(degrees * PI / 180.) - ir * sin(degrees * PI / 180.);
         double ir_rot = ir * cos(degrees * PI / 180.) + ic * sin(degrees * PI / 180.);
         
         if (ic < intercepts[0] * NC) {
            potential_shift = 2.0 * shift_ac;
         } else if (ic < intercepts[1] * NC) {
            potential_shift = (2.0 * shift_ac) - shift_ac * (ic - (NC * intercepts[0])) / (NC * (intercepts[1] - intercepts[0]));
         } else if (ic < intercepts[2] * NC) {
            potential_shift = shift_ac;
         } else if (ic < intercepts[3] * NC) {
            potential_shift = shift_ac + shift_ac * (ic - (NC * intercepts[2])) / (NC * (intercepts[3] - intercepts[2]));
         } else {
            potential_shift = 2.0 * shift_ac;
         }
         
         for (int i = 0; i < 3; ++i) {
            potential_function_terms[i] =
            c_max[i] -
            (c_max[i] - c_min[i]) * (2. / 9.) *
            (3. - (2. * cos(2 * PI * ic_rot / atomic_spacing) *
                   cos(2. * PI * (ir_rot - potential_shift) /
                       (sqrt(3.) * atomic_spacing)) +
                   cos(4. * PI * (ir_rot - potential_shift) /
                       (sqrt(3.) * atomic_spacing))));
         }
         
         potential_value =
         (potential_amplitude *
          (((potential_function_terms[0] *
             exp(-z_eq * potential_function_terms[1]) -
             potential_function_terms[2] / (z_eq * z_eq * z_eq * z_eq)) -
            min_potential) /
           (max_potential - min_potential) -
           0.5));
         
         (*n_mat).set(ir, ic, potential_value);
      }
   }
}


void InitialHorizontalDoubleStripes(matrix_t* n_mat, const double r0) {
   
   srand(0);
   double shift_ac = r0 * sqrt(3.) / 2.;
   matrix_t perfect_ab(r0, 3.0, 2.0 * shift_ac, 1.0, 0.0);
   matrix_t perfect_ac(r0, 3.0, shift_ac, 1.0, 0.0);
   
   #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
            for (int ic = 0; ic < NC; ++ic) {
               if (ir < 0.1 * NR) {
                  (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
               } else if (ir < 0.15 * NR) {
                  (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
               } else if (ir < 0.35 * NR) {
                  (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
               } else if (ir < 0.4 * NR) {
                  (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
               } else if (ir < 0.6 * NR) {
                  (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
               } else if (ir < 0.65 * NR) {
                  (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
               } else if (ir < 0.85 * NR) {
                  (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
               } else if (ir < 0.9 * NR) {
                  (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
               } else {
                  (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
               }
            }
      }
}

void InitialAA(matrix_t* n_mat, const double r0) {

   matrix_t perfect_aa(r0, 3.0, 0.0, 1.0, 0.0);

   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         (*n_mat).set(ir, ic, perfect_aa.get(ir, ic));
      }
   }
}

void AddConstToMatrix(matrix_t* n_mat, const double amount) {
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         double tmp = (*n_mat).get(ir, ic);
         (*n_mat).set(ir, ic, tmp + amount);
      }
   }
}

void MakeKs(matrix_t* k_r, matrix_t* k_th) {
   // make a vector of 2 PI x / L for the row and column sizes
   // these are cartesian k-vectors
   double k_row[NR];
   double k_col[NC];
   #pragma omp simd
   for (int ir = 0; ir < NR; ++ir) {
      double tmp;
      if (ir > NR / 2.0) {
         tmp = (1.0 * ir) / NR - 1.0;
      } else {
         tmp = (1.0 * ir) / NR;
      }
      k_row[ir] = 2.0 * PI * tmp;
   }
   #pragma omp simd
   for (int ic = 0; ic < NC; ++ic) {
      double tmp;
      if (ic > NC / 2.0) {
         tmp = (1.0 * ic) / NC - 1.0;
      } else {
         tmp = (1.0 * ic) / NC;
      }
      k_col[ic] = 2.0 * PI * tmp;
   }
   
   // define magnitude of k matrix = k_r and angular k matrix = k_th
   // ensure that k_th terms are non-negative
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         (*k_r).set(ir, ic,
                    sqrt((k_row[ir] * k_row[ir]) + (k_col[ic] * k_col[ic]) ) );
         if (ir == 0 && ic == 0) {
            (*k_th).set(0, 0, 0.);
         } else {
            double tmp = atan2(k_row[ir], k_col[ic]);
            if (tmp < 0.) {
               (*k_th).set(ir, ic, tmp + (2.0 * PI));
            } else {
               (*k_th).set(ir, ic, tmp);
            }
         }
      }
   }
}


void MakeCs(matrix_t* c2, matrix_t* cs1, matrix_t* cs2, matrix_t* minus_k_squared, const double r0, const double a0) {
   
   // define k-vectors, with origin at (0,0), not at matrix center.
   matrix_t k_r, k_th;
   MakeKs(&k_r, &k_th);
   
   // make the C2 matrix for 2-point correlations
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         (*minus_k_squared).set(ir, ic, -1. * k_r.get(ir, ic) * k_r.get(ir, ic));
         if (ir == 0 && ic == 0) {
            (*c2).set(0, 0, -6.);
         } else {
            double tmp = boost::math::cyl_bessel_j(1, r0 * k_r.get(ir, ic)) /
            (r0 * k_r.get(ir, ic));
            if (fabs(tmp) < pow(10, -14)) {
               tmp = 0.0;
            }
            (*c2).set(ir, ic, -12.0 * tmp);
         }
      }
   }
   // make the Cs1, Cs2 matrices for 3-point correlations
   // Cs1 and Cs2 still need to multiplied by i! No real component.
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         double tmp = -2.5 * cos(3. * k_th.get(ir, ic)) *
         boost::math::cyl_bessel_j(3, a0 * k_r.get(ir, ic));
         if (fabs(tmp) < pow(10, -14)) {
            tmp = 0.0;
         }
         (*cs1).set(ir, ic, tmp);
      }
   }
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         double tmp = -2.5 * sin(3. * k_th.get(ir, ic)) *
         boost::math::cyl_bessel_j(3, a0 * k_r.get(ir, ic));
         if (fabs(tmp) < pow(10, -14)) {
            tmp = 0.0;
         }
         (*cs2).set(ir, ic, tmp);
      }
   }
}

void WriteMatrix(const double time, const double energy_val, const std::string directory_string, const matrix_t& mat) {
   
   const std::string time_str = "{" + boost::lexical_cast<std::string>(energy_val) + "}, ";
   const std::string file_str =
   directory_string + boost::lexical_cast<std::string>(floor(time)) + ".txt";
   const char* time_char = time_str.c_str();
   const char* file_char = file_str.c_str();
   mat.WriteToFile(file_char, time_char);
  
}

// an inital condition that produces an AC circle inside an AB matrix
void AFMTip(matrix_t* total_potential, const double radius, const double speed, const double force, const double angle, const double time, const double delay, const matrix_t& fixed_potential) {

   // define center of tip as a function of time
   const double c_x = NR / 2. + cos(angle) * speed * (time - delay - NR / (2. * cos(angle)) );
   const double c_y = NC / 2. + sin(angle) * speed * (time - delay - NR / (2. * cos(angle)) );
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         double radial_distance = sqrt((ir - c_x) * (ir - c_x) + (ic - c_y) * (ic - c_y));
         double local_fixed_potential = fixed_potential.get(ir, ic);
         if (radial_distance <= radius) {
            (*total_potential).set(ir, ic, local_fixed_potential + force);
         } else {
            (*total_potential).set(ir, ic, local_fixed_potential);
         }
      }
   }
   
}



/*
void ReadMatrix(matrix_t* mat, const char* file_location) {
   auto ss = std::ostringstream{};
   ss << in.rdbuf();
   auto s = ss.str();
}
 */
