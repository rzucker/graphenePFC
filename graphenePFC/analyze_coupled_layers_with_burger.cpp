//
//  analyze_coupled_layers_with_burger.cpp
//  graphenePFC
//
//  Created by Rachel Zucker on 9/7/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#include "analyze_coupled_layers_with_burger.hpp"
#include "analyze_coupled_layers.hpp"
#include "postprocessing.hpp"
#include "matrix_types.hpp"
#include <boost/geometry.hpp>

struct FancyPoint {
   Point atom;
   std::array<double, 3> abc;
   Point burger;
   FancyPoint (Point position, std::array<double, 3> rgb, Point vector) : atom(position), abc(rgb), burger(vector) {}
};

Point NearestPointVector(const Point point, const std::vector<Point>& possible_neighbors) {
   // if point is up, compare with up
   int length = (int) possible_neighbors.size();
   std::vector<ScalarAndId> all_data;
   for (int i = 0; i < length; ++i) {
      double tmp = sqrt( ((possible_neighbors.at(i).x - point.x) * (possible_neighbors.at(i).x - point.x)) + ((possible_neighbors.at(i).y - point.y) * (possible_neighbors.at(i).y - point.y)) );
      if ( tmp > 0.) {
         all_data.push_back(ScalarAndId(i, tmp));
      }
   }
   std::sort(all_data.begin(), all_data.end(), SortByScalar);
   int index = all_data.at(0).id;
   Point raw_result = Point(possible_neighbors.at(index).x - point.x, possible_neighbors.at(index).y - point.y);
   double magnitude = sqrt( (raw_result.x * raw_result.x) + (raw_result.y * raw_result.y) );
   double angle = 0.;
   if(magnitude > 0){
      double arc_tan = atan2(raw_result.y, raw_result.x);
      angle = fmod((arc_tan + 2. * PI), (2. * PI / 3.));
   }
   return Point(magnitude * cos(angle), magnitude * sin(angle));
}

void CoupledStackingBurger(const padded_matrix_t& hole_t, const padded_matrix_t& hole_b, const padded_matrix_t& up_t, const padded_matrix_t& up_b, const padded_matrix_t& down_t, const padded_matrix_t& down_b, std::vector<FancyPoint>* all_atom_data, const double r0) {
   
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         
         if (up_t.get(ir, ic) == 1.) {
            std::vector<Point> neighbor_holes, neighbor_ups, neighbor_downs;
            Point center = Point(0., 0.);
            Point local_shift = Point(10 * r0, 10 * r0);
            double aa = 10 * r0;
            double ab = 10 * r0;
            double ac = 10 * r0;
            // find the nearest holes, ups, and downs in the bottom layer to a given up atom in the top layer
            FindNeighbors(ir, ic, &neighbor_holes, hole_b, r0);
            FindNeighbors(ir, ic, &neighbor_ups, up_b, r0);
            FindNeighbors(ir, ic, &neighbor_downs, down_b, r0);
            // find the minimum distance between the up atom in the top layer and the 3 site types in the bottom layer
            if ( neighbor_holes.size() > 0) {
               ac = NearestPointDist(center, neighbor_holes) /(0.75 * r0);
            }
            if ( neighbor_ups.size() > 0) {
               aa = NearestPointDist(center, neighbor_ups) /(0.75 * r0);
               local_shift = NearestPointVector(center, neighbor_ups);
            }
            if ( neighbor_downs.size() > 0) {
               ab = NearestPointDist(center, neighbor_downs) /(0.75 * r0);
            }
            (*all_atom_data).push_back(FancyPoint(Point(ic, ir), {aa, ab, ac}, local_shift));
         }
         
         
         if (down_t.get(ir, ic) == 1.) {
            std::vector<Point> neighbor_holes, neighbor_ups, neighbor_downs;
            Point center = Point(0., 0.);
            Point local_shift = Point(10 * r0, 10 * r0);
            double aa = 10 * r0;
            double ab = 10 * r0;
            double ac = 10 * r0;
            // find the nearest holes, ups, and downs in the bottom layer to a given up atom in the top layer
            FindNeighbors(ir, ic, &neighbor_holes, hole_b, r0);
            FindNeighbors(ir, ic, &neighbor_ups, up_b, r0);
            FindNeighbors(ir, ic, &neighbor_downs, down_b, r0);
            // find the minimum distance between the up atom in the top layer and the 3 site types in the bottom layer
            if ( neighbor_holes.size() > 0) {
               ab = NearestPointDist(center, neighbor_holes) /(0.75 * r0);
            }
            if ( neighbor_ups.size() > 0) {
               ac = NearestPointDist(center, neighbor_ups) /(0.75 * r0);
               local_shift = NearestPointVector(center, neighbor_ups);
            }
            if ( neighbor_downs.size() > 0) {
               aa = NearestPointDist(center, neighbor_downs) /(0.75 * r0);
               local_shift = NearestPointVector(center, neighbor_downs);
            }
            (*all_atom_data).push_back(FancyPoint(Point(ic, ir), {aa, ab, ac}, local_shift));
         }
         
      }
   }
}

void WriteCoupledBurger(const double time, const std::string directory_string, const std::vector<FancyPoint>& pcb) {
   std::string file_str = directory_string + "points" + boost::lexical_cast<std::string>(floor(time)) + ".txt";
   const char* file_char = file_str.c_str();
   std::ofstream file(file_char);
   if (file.is_open()) {
      file << "{";
      for (auto& p : pcb) {
         file << " {{" << p.atom.x << ", " << p.atom.y << "}, {" << p.abc[0] << ", " << p.abc[1] << ", " << p.abc[2] << "}, {" << p.burger.x << ", " << p.burger.y << "}}" << std::endl;
      }
      file << " {} }";
      file.close();
      std::cout << "done writing " << file_char << std::endl;
   } else
      std::cout << "Unable to open file" << std::endl;
}

void AnalyzeCoupledBurger(const matrix_t& top, const matrix_t& bottom, const double r0, const double time, const std::string directory_string_t, const std::string directory_string_b) {
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
   std::vector<FancyPoint> atom_data_t, atom_data_b;
   CoupledStackingBurger(hole_t, hole_b, up_t, up_b, down_t, down_b, &atom_data_t, r0);
   CoupledStackingBurger(hole_b, hole_t, up_b, up_t, down_b, down_t, &atom_data_b, r0);
   
   // initialize a vector of polygons
   std::vector<Polygon> polygons_t, polygons_b;
   int polygon_counter_t = 0;
   int polygon_counter_b = 0;
   // populate it with white polygons centered on each hole
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         if (hole_t.get(ir, ic) == 1.) {
            // initialize with sequential ID, color, and center point
            polygons_t.push_back(Polygon(polygon_counter_t, {1., 1., 1.}, Point(ic, ir) ) );
            polygon_counter_t += 1;
         }
         if (hole_b.get(ir, ic) == 1.) {
            // initialize with sequential ID, color, and center point
            polygons_b.push_back(Polygon(polygon_counter_b, {1., 1., 1.}, Point(ic, ir) ) );
            polygon_counter_b += 1;
         }
      }
   }
   // find the nearby atoms composing the polygon
   MakePolygons(&polygons_t, atoms_t, r0);
   MakePolygons(&polygons_b, atoms_b, r0);
   
   WriteCoupledBurger(time, directory_string_t, atom_data_t);
   WriteCoupledBurger(time, directory_string_b, atom_data_b);
   
   WritePolygonFile(time, directory_string_t, polygons_t);
   WritePolygonFile(time, directory_string_b, polygons_b);
   
}
