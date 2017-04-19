//
//  readMD.cpp
//  graphenePFC
//
//  Created by Rachel Zucker on 10/23/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#include <stdio.h>
#include <iostream>

void AnalyzeCoupledLayers(const matrix_t& top, const matrix_t& bottom, const double r0, const double time, const std::string directory_string_t, const std::string directory_string_b) {
   // omp_get_max_threads();
   // pad the matrices
   padded_matrix_t pad_t, pad_b;
   
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         double tmp_t = top.get( (ir - PAD(0) + NR) % NR, (ic - PAD(0) + NC) % NC);
         double tmp_b = bottom.get( (ir - PAD(0) + NR) % NR, (ic - PAD(0) + NC) % NC);
         pad_t.set(ir, ic, tmp_t);
         pad_b.set(ir, ic, tmp_b);
      }
   }
   
   // find local minima in pad by comparing with 8 nearest neighbor pixels
   const double global_max_t = top.MaxValue();
   const double global_max_b = bottom.MaxValue();
   const double global_min_t = top.MinValue();
   const double global_min_b = bottom.MinValue();
   const double max_minus_min_t = global_max_t - global_min_t;
   const double max_minus_min_b = global_max_b - global_min_b;
   // write 1 in minima/maxima matrix entries, 0 otherwise
   padded_matrix_t hole_t, hole_b, atoms_t, atoms_b;
   FindExtrema(pad_t, &hole_t, &atoms_t, max_minus_min_t, global_min_t);
   FindExtrema(pad_b, &hole_b, &atoms_b, max_minus_min_b, global_min_b);
   
   // split the atoms matrices into up and down
   padded_matrix_t up_t, up_b, down_t, down_b;
   MakeUpDownMatrices(&up_t, &down_t, atoms_t, r0);
   MakeUpDownMatrices(&up_b, &down_b, atoms_b, r0);
   
   // find stacking
   padded_matrix_t aa_t, ab_t, ac_t, aa_b, ab_b, ac_b;
   CoupledStacking(hole_t, hole_b, up_t, up_b, down_t, down_b, &aa_t, &ab_t, &ac_t, r0);
   CoupledStacking(hole_b, hole_t, up_b, up_t, down_b, down_t, &aa_b, &ab_b, &ac_b, r0);
   
   std::vector<PointAndColor> pts_and_colors_t, pts_and_colors_b;
   
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         if (atoms_t.get(ir, ic) != 0.) {
            Point atom = Point(ic, ir);
            std::array<double, 3> abc = {aa_t.get(ir, ic), ab_t.get(ir, ic), ac_t.get(ir, ic) };
            pts_and_colors_t.push_back(PointAndColor(atom, abc) );
         }
         if (atoms_b.get(ir, ic) != 0.) {
            Point atom = Point(ic, ir);
            std::array<double, 3> abc = {aa_b.get(ir, ic), ab_b.get(ir, ic), ac_b.get(ir, ic) };
            pts_and_colors_b.push_back(PointAndColor(atom, abc) );
         }
      }
   }
   
   WriteCoupled(time, directory_string_t, pts_and_colors_t);
   WriteCoupled(time, directory_string_b, pts_and_colors_b);
   
}


int main(int argc, const char* argv[]) {
   FILE* file_to_read = fopen("/Users/Rachel/Downloads/edgeTotalFlatNo.txt", "r");
   int atom, id;
   double pos_x, pos_y, pos_z, potential;
   while (!feof(file_to_read)) {
      int read_line = fscanf(file_to_read, "%i %i %lf %lf %lf %lf", &atom, &id, &pos_x, &pos_y, &pos_z, &potential);
      std::cout << "x, y, x: " << pos_x << ", " << pos_y << ", " << pos_z << std::endl;
   }
   return 0;
}
