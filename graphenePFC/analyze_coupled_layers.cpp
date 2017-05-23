//
//  analyze_coupled_layers.cpp
//  graphenePFC
//
//  Created by Rachel Zucker on 9/1/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#include "analyze_coupled_layers.hpp"
#include "postprocessing.hpp"
#include "matrix_types.hpp"
#include <boost/geometry.hpp>
#include <omp.h>

// NearestThreePoints returns the nearest_three_atoms in the vector possible_atoms
// to the point central_atom

void NearestThreePoints(std::vector<Point>* nearest_three_atoms, const std::vector<Point>& possible_neighbors) {
   
   int length = (int) possible_neighbors.size(); // why do I need (int)? Shouldn't .size() be returning an int?
   std::vector<ScalarAndId> all_data;
   for (int i = 0; i < length; ++i) {
      double tmp = sqrt( (possible_neighbors.at(i).x * possible_neighbors.at(i).x) + (possible_neighbors.at(i).y * possible_neighbors.at(i).y) );
      if ( tmp > 0.) {
         all_data.push_back(ScalarAndId(i, tmp, possible_neighbors.at(i).x, possible_neighbors.at(i).y));
      }
   }
   
   if (all_data.size() > 2) {
      std::sort(all_data.begin(), all_data.end(), SortByScalar);
      for (int j = 0; j < 3; ++j) {
         Point p = Point(all_data.at(j).x, all_data.at(j).y);
         (*nearest_three_atoms).push_back(p);
      }
   }
}

bool IsItAnUpAtom(const std::vector<Point>& triangle) {
   // if arctan > 0, point is above horizontal divider.
   std::vector<double> arctans(3);
   for (int i = 0; i < 3; ++i) {
      arctans[i] = atan2(triangle[i].y, triangle[i].x);
   }
   // if 1 atan is positive, it's an up site. if 2 atans are positive, it's a down site.
   std::sort(arctans.begin(), arctans.end());
   return arctans[1] < 0;
}

void FindNeighbors(const int row, const int col, std::vector<Point>* possible_neighbors, const padded_matrix_t& atom_sites, const double a0) {
   // find all atoms within a circle of fixed radius of the central atom
   double range = a0 * 2.;
   for (int i = floor(-range); i < ceil(range); ++i) {
      for (int j = floor(-range); j < ceil(range); ++j) {
         if ( sqrt( i * i + j * j) <= range ) {
            // wrap coordinates into matrix
            // neighbor positions are relative to the center point,
            // i.e., the center is shifted to (0,0)
            if (atom_sites.get( (row + i + PAD(PAD(NR)) ) % PAD(PAD(NR)), (col + j + PAD(PAD(NC))) % PAD(PAD(NC))) != 0.) {
               (*possible_neighbors).push_back(Point(j, i) );
            }
         }
      }
   }

}

void MakeUpDownMatrices(padded_matrix_t* up_sites, padded_matrix_t* down_sites,
                        const padded_matrix_t& atom_sites, const double a0) {
   // put 1's in up and down sites
   
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         // initialize up and down
         (*up_sites).set(ir, ic, 0.);
         (*down_sites).set(ir, ic, 0.);
         if (atom_sites.get(ir, ic) != 0.) {
            // find candidate nearest neighbors
            // neighbor coordinates are relative to center point coordinates,
            // i.e., center point is shifted to (0,0)
            std::vector<Point> possible_neighbor_atoms;
            FindNeighbors(ir, ic, &possible_neighbor_atoms, atom_sites, a0);
            if (possible_neighbor_atoms.size() > 2) {
               // using candidate neighbors, find the nearest 3 atoms
               std::vector<Point> triangle;
               NearestThreePoints(&triangle, possible_neighbor_atoms);
               // then decide whether they form a triangle that points up or
               // down
               if (triangle.size() == 3) {
                  if (IsItAnUpAtom(triangle)) {
                     (*up_sites).set(ir, ic, 1.);
                  } else {
                     (*down_sites).set(ir, ic, 1.);
                  }
               }
            }
         }
      }
   }
}


void CoupledStacking(const padded_matrix_t& hole_t, const padded_matrix_t& hole_b, const padded_matrix_t& up_t, const padded_matrix_t& up_b, const padded_matrix_t& down_t, const padded_matrix_t& down_b, padded_matrix_t* aa, padded_matrix_t* ab, padded_matrix_t* ac, const double a0) {
   
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         
         (*aa).set(ir, ic, 0.);
         (*ab).set(ir, ic, 0.);
         (*ac).set(ir, ic, 0.);
         double const bond_length = a0 / sqrt(3.);

         if (up_t.get(ir, ic) == 1.) {
            std::vector<Point> neighbor_holes, neighbor_ups, neighbor_downs;
            Point center = Point(0., 0.);
            // find the nearest holes, ups, and downs in the bottom layer to a given up atom in the top layer
            FindNeighbors(ir, ic, &neighbor_holes, hole_b, a0);
            FindNeighbors(ir, ic, &neighbor_ups, up_b, a0);
            FindNeighbors(ir, ic, &neighbor_downs, down_b, a0);
            // find the minimum distance between the up atom in the top layer and the 3 site types in the bottom layer
            if ( neighbor_holes.size() > 0) {
               double nearest_hole_dist = NearestPointDist(center, neighbor_holes);
               (*ac).set(ir, ic, nearest_hole_dist/(bond_length));
            } else {
               (*ac).set(ir, ic, 10 * bond_length);
            }
            if ( neighbor_ups.size() > 0) {
               double nearest_up_dist = NearestPointDist(center, neighbor_ups);
               (*aa).set(ir, ic, nearest_up_dist/(bond_length));
            } else {
               (*aa).set(ir, ic, 10 * bond_length);
            }
            if ( neighbor_downs.size() > 0) {
               double nearest_down_dist = NearestPointDist(center, neighbor_downs);
               (*ab).set(ir, ic, nearest_down_dist/(bond_length));
            } else {
               (*ab).set(ir, ic, 10 * bond_length);
            }
         }
         
         if (down_t.get(ir, ic) == 1.) {
            std::vector<Point> neighbor_holes, neighbor_ups, neighbor_downs;
            Point center = Point(0., 0.);
            // find the nearest holes, ups, and downs in the bottom layer to a given down atom in the top layer
            FindNeighbors(ir, ic, &neighbor_holes, hole_b, a0);
            FindNeighbors(ir, ic, &neighbor_ups, up_b, a0);
            FindNeighbors(ir, ic, &neighbor_downs, down_b, a0);
            // find the minimum distance between the down atom in the top layer and the 3 site types in the bottom layer
            if (neighbor_holes.size() > 0) {
               double nearest_hole_dist = NearestPointDist(center, neighbor_holes);
               (*ab).set(ir, ic, nearest_hole_dist/(bond_length));
            } else {
               (*ab).set(ir, ic, 10 * bond_length);
            }
            if (neighbor_ups.size() > 0) {
               double nearest_up_dist = NearestPointDist(center, neighbor_ups);
               (*ac).set(ir, ic, nearest_up_dist/(bond_length));
            } else {
               (*ac).set(ir, ic, 10 * bond_length);
            }
            if (neighbor_downs.size() > 0) {
               double nearest_down_dist = NearestPointDist(center, neighbor_downs);
               (*aa).set(ir, ic, nearest_down_dist/(bond_length));
            } else {
               (*aa).set(ir, ic, 10 * bond_length);
            }
         }
      }
   }
}

void WriteCoupled(const double time, const std::string directory_string, const std::vector<PointAndColor>& pc) {
   std::string file_str = directory_string + "a" + boost::lexical_cast<std::string>(floor(time)) + ".txt";
   const char* file_char = file_str.c_str();
   std::ofstream file(file_char);
   if (file.is_open()) {
      file << "{";
      for (auto& p : pc) {
         file << " {{" << p.atom.x << ", " << p.atom.y << "}, {" << p.abc[0] << ", " << p.abc[1] << ", " << p.abc[2] << "}}, " << std::endl;
      }
      file << " {} }";
      file.close();
      std::cout << "done writing " << file_char << std::endl;
   } else
      std::cout << "Unable to open file" << std::endl;
}



void AnalyzeCoupledLayers(const matrix_t& top, const matrix_t& bottom, const double a0, const double time, const std::string directory_string_t, const std::string directory_string_b) {
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
   MakeUpDownMatrices(&up_t, &down_t, atoms_t, a0);
   MakeUpDownMatrices(&up_b, &down_b, atoms_b, a0);
   
   // find stacking
   padded_matrix_t aa_t, ab_t, ac_t, aa_b, ab_b, ac_b;
   CoupledStacking(hole_t, hole_b, up_t, up_b, down_t, down_b, &aa_t, &ab_t, &ac_t, a0);
   CoupledStacking(hole_b, hole_t, up_b, up_t, down_b, down_t, &aa_b, &ab_b, &ac_b, a0);
   
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



