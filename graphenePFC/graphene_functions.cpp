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
         } else if (radial_distance < max_radius * 0.55) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         }
      }
   }
}

void InitialShallowStripes(matrix_t* n_mat, const double r0) {

   srand(0);
   double shift_ac = r0 * sqrt(3.) / 2.;
   matrix_t perfect_ab(r0, 3.0, 2.0 * shift_ac, 1.0, 0.0);
   matrix_t perfect_ac(r0, 3.0, shift_ac, 1.0, 0.0);
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         if (ir < (-0.5) * ic + 0.0690983 * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-0.5) * ic + 0.180902 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + 0.319098 * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-0.5) * ic + 0.430902 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + 0.569098 * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-0.5) * ic + 0.680902 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + 0.819098 * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-0.5) * ic + 0.930902 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + 1.0691 * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-0.5) * ic + 1.1809 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-0.5) * ic + 1.3191 * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-0.5) * ic + 1.4309 * NR) {
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
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         if (ir < (-2.0) * ic + 0.138197 * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-2.0) * ic + 0.361803 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + 0.638197 * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-2.0) * ic + 0.861803 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + 1.1382 * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-2.0) * ic + 1.3618 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + 1.6382 * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-2.0) * ic + 1.8618 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + 2.1382 * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < (-2.0) * ic + 2.3618 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < (-2.0) * ic + 2.6382 * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < (-2.0) * ic + 2.8616 * NR) {
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
   
   double intercepts [] = {0.179289, 0.320711, 0.679289, 0.820711, 1.17929, 1.32071, 1.67929, 1.82071};
   
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
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         if (ic < 0.2 * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ic < 0.3 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ic < 0.7 * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ic < 0.8 * NR) {
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
   
   #pragma omp parallel for
   for (int ir = 0; ir < NR; ++ir) {
      #pragma omp simd
      for (int ic = 0; ic < NC; ++ic) {
         if (ir < 0.2 * NR) {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         } else if (ir < 0.3 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else if (ir < 0.7 * NR) {
            (*n_mat).set(ir, ic, perfect_ac.get(ir, ic));
         } else if (ir < 0.8 * NR) {
            (*n_mat).set(ir, ic, (rand() % 2000) / 1000. - 1.0);
         } else {
            (*n_mat).set(ir, ic, perfect_ab.get(ir, ic));
         }
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
   std::fixed;
   const std::string time_str = "{" + boost::lexical_cast<std::string>(energy_val) + "}, ";
   const std::string file_str =
   directory_string + boost::lexical_cast<std::string>(floor(time)) + ".txt";
   const char* time_char = time_str.c_str();
   const char* file_char = file_str.c_str();
   mat.WriteToFile(file_char, time_char);
   std::scientific;
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
