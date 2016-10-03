//
//  matrix.hpp
//  graphenePFC
//
//  Created by Rachel Zucker on 8/7/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#ifndef matrix_hpp
#define matrix_hpp
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#define PI 3.141592653589793

template <int R, int C> class Matrix {
 public:
   // Constructors

   Matrix() { _storage.resize(R * C); }

Matrix(double r0, double potential_amplitude, double potential_shift,
       double potential_stretch, double add_constant) {
   _storage.resize(R * C);
   // define helper numbers
   const double c_max[] = {2.75075, 3.349208, 8258.11};
   const double c_min[] = {1.63093, 3.347616, 8184.70};
   const double atomic_spacing = 1.5 * r0 * potential_stretch;
   const double z_eq = 3.31;
   double min_potential = -68.7968;
   double max_potential = -68.1889;
   double potential_function_terms[3];
   // iterate over the matrix, storing values
   for (int ir = 0; ir < R; ++ir) {
      for (int ic = 0; ic < C; ++ic) {
         for (int i = 0; i < 3; ++i) {
            potential_function_terms[i] =
                c_max[i] -
                (c_max[i] - c_min[i]) * (2. / 9.) *
                    (3. - (2. * cos(2 * PI * ic / atomic_spacing) *
                               cos(2. * PI * (ir - potential_shift) /
                                   (sqrt(3.) * atomic_spacing)) +
                           cos(4. * PI * (ir - potential_shift) /
                               (sqrt(3.) * atomic_spacing))));
         }
         _storage[ir * C + ic] =
             (potential_amplitude *
              (((potential_function_terms[0] *
                     exp(-z_eq * potential_function_terms[1]) -
                 potential_function_terms[2] / (z_eq * z_eq * z_eq * z_eq)) -
                min_potential) /
                   (max_potential - min_potential) -
               0.5)) +
             add_constant;
      }
   }
}   
   // be able to ask matrix what it's size is
   static const int max_row = R;
   static const int max_col = C;

   // Methods

   void set(int ir, int ic, double value) { _storage[ir * C + ic] = value; }

   double get(int ir, int ic) const { return _storage[ir * C + ic]; }

   void WriteToFile(const char* file_name, const char* file_header) const {
      std::ofstream file(file_name);
      if (file.is_open()) {
         file << "{" << file_header << std::endl;
         for (int ir = 0; ir < R - 1; ++ir) {
            file << " {";
            for (int ic = 0; ic < C - 1; ++ic) {
               file << _storage[(ir * C) + ic] << ",";
            }
            file << _storage[(ir * C) + C - 1] << "}," << std::endl;
         }
         file << " {";
         for (int ic = 0; ic < C - 1; ++ic) {
            file << _storage[((R - 1) * C) + ic] << ",";
         }
         file << _storage[((R - 1) * C) + C - 1] << "}" << std::endl;
         file << "}";
         file.close();
         std::cout << "done writing " << file_name << std::endl;
      } else
         std::cout << "Unable to open file" << std::endl;
   }
   
   Matrix operator+(const double& add) {
      Matrix res;
      for (int ir = 0; ir < R; ++ir) {
         for (int ic = 0; ic < C; ++ic) {
            res.set(ir, ic, this->get(ir, ic) + add);
         }
      }
      return res;
   }
   
   Matrix operator*(const double& times) {
      Matrix res;
      for (int ir = 0; ir < R; ++ir) {
         for (int ic = 0; ic < C; ++ic) {
            res.set(ir, ic, this->get(ir, ic) * times);
         }
      }
      return res;
   }

   double MaxAbsValue() const {
      double max_abs_value = 0.0;
      for (int i = 0; i < R * C; ++i) {
         if (fabs(_storage[i]) > max_abs_value) {
            max_abs_value = fabs(_storage[i]);
         }
      }
      return max_abs_value;
   }

   double MaxValue() const {
      double max_value = _storage[0];
      for (int i = 1; i < R * C; ++i) {
         if (_storage[i] > max_value) {
            max_value = _storage[i];
         }
      }
      return max_value;
   }
   double MinValue() const {
      double min_value = _storage[0];
      for (int i = 0; i < R * C; ++i) {
         if (_storage[i] < min_value) {
            min_value = _storage[i];
         }
      }
      return min_value;
   }

 private:
   std::vector<double> _storage;
   // maybe replace this with a fixed size array on the heap, rather than
   // dynamically sized. Use new and *'s?
};

#endif /* matrix_hpp */