//
//  postprocessing.cpp
//  graphenePFC
//
//  Created by Rachel Zucker on 8/15/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#include "postprocessing.hpp"
#include "matrix_types.hpp"
#include <cmath>
#include <boost/geometry.hpp>
#include <omp.h>

#define PI 3.141592653589793

typedef boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian> BoostPoint;
typedef boost::geometry::model::polygon<BoostPoint> BoostPolygon;

bool SortByScalar(const ScalarAndId& a, const ScalarAndId& b) {
   return a.scalar < b.scalar;
};

// this finds local minima that are within 20% of the lowest value in the
// matrix, and local maxima withing 20% of the highest value,
// populating matrix with 1's at the extrema, 0's elsewhere

void FindExtrema(const padded_matrix_t& mat, padded_matrix_t* minima, padded_matrix_t* maxima, const double max_minus_min, const double global_min) {
   #pragma omp parallel for
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         (*minima).set(ir, ic, 0.0);
         (*maxima).set(ir, ic, 0.0);
      }
   }
   #pragma omp parallel for
   for (int ir = 1; ir < PAD(PAD(NR)) - 1; ++ir) {
      for (int ic = 1; ic < PAD(PAD(NC)) - 1; ++ic) {
         
         double neighbor_values[3][3];
         for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
               neighbor_values[i][j] = mat.get(ir + i - 1, ic + j - 1);
            }
         }
         double min = neighbor_values[0][0];
         double max = neighbor_values[0][0];
         for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
               if (neighbor_values[i][j] < min) {
                  min = neighbor_values[i][j];
               }
               if (neighbor_values[i][j] > max) {
                  max = neighbor_values[i][j];
               }
            }
         }
         
         if (neighbor_values[1][1] == min && min < (0.2 * max_minus_min + global_min)) {
            (*minima).set(ir, ic, 1.);
         }
         if (neighbor_values[1][1] == max && max > (0.8 * max_minus_min + global_min)) {
            (*maxima).set(ir, ic, 1.);
         }
      }
   }
}

// this populates the 6 argument vectors with the coordinates of
// hole sites, up sites, and down sites near the origin unit cell

void PotentialSites(const double r0, std::vector<Point>* hole, std::vector<Point>* up, std::vector<Point>* down) {
   
   // locate the up, down and hole sites in the origin unit cell
   // convention:
   // if the hole is over a hole: AA stacking
   // if the the hole is over a down site: AC stacking
   // if the hole is over an up site: AB stacking
   double sqrt3 = sqrt(3.);
   double holes_in_uc[4][2] = {{0., 0.},
                          {1.5, 0.},
                          {0.75, 0.75 * sqrt3},
                          {2.25, 0.75 * sqrt3}};
   double downs_in_uc[4][2] = {{0., 0.5 * sqrt3},
                          {1.5, 0.5 * sqrt3},
                          {0.75, 1.25 * sqrt3},
                          {2.25, 1.25 * sqrt3}};
   double ups_in_uc[4][2] = {{0., sqrt3},
                        {1.5, sqrt3},
                        {0.75, 1.75 * sqrt3},
                        {2.25, 1.75 * sqrt3}};
   // create vectors of site positions in the 3x3 grid of cells near the origin
   // unit cell
   int current_index = 0;
   #pragma omp parallel for
   for (int site = 0; site < 4; ++site) {
      for (int uc_row_index = -1; uc_row_index < 2; ++uc_row_index) {
         for (int uc_col_index = -1; uc_col_index < 2; ++uc_col_index) {
            (*hole)[current_index].x = r0 * holes_in_uc[site][0] + r0 * 1.5 * uc_col_index;
            (*hole)[current_index].y = r0 * holes_in_uc[site][1] + r0 * 1.5 * sqrt3 * uc_row_index;
            (*up)[current_index].x = r0 * ups_in_uc[site][0] + r0 * 1.5 * uc_col_index;
            (*up)[current_index].y = r0 * ups_in_uc[site][1] + r0 * 1.5 * sqrt3 * uc_row_index;
            (*down)[current_index].x = r0 * downs_in_uc[site][0] + r0 * 1.5 * uc_col_index;
            (*down)[current_index].y = r0 * downs_in_uc[site][1] + r0 * 1.5 * sqrt3 * uc_row_index;
            current_index += 1;
         }
      }
   }
}

// nearestPointDist returns the minimum distance between the point pt
// and a point in the vector candidates

double NearestPointDist(const Point pt,
                        const std::vector<Point>& candidates) {
   
   double minimum_distance = sqrt((pt.x - candidates[0].x) * (pt.x - candidates[0].x) +
                                 (pt.y - candidates[0].y) * (pt.y - candidates[0].y));
   for (int i = 0; i < candidates.size(); ++i) {
      double ith_distance = sqrt((pt.x - candidates[i].x) * (pt.x - candidates[i].x) +
                        (pt.y - candidates[i].y) * (pt.y - candidates[i].y));
      if (ith_distance < minimum_distance) {
         minimum_distance = ith_distance;
      }
   }
   return minimum_distance;
}

// findStacking populates the matrix "stacking" with 0 for irrelevant points,
// 1 if the hole is near a potential hole, 2 if the hole is near a potential up site,
// and 3 if the hole is near a potential down site.

void FindStacking(const padded_matrix_t& minima, padded_matrix_t* color_red,
                  padded_matrix_t* color_green, padded_matrix_t* color_blue, const double r0) {

   // generate lists of hole, up, and down sites near the origin unit cell with
   // corners (0,0) and (3/2 r0) * (sqrt(3), 1)
   std::vector<Point> hole, up, down;
   // their size = 4 sites / UC * 3x3 UC block = 36 sites
   for (int i = 0; i < 36; ++i) {
      hole.push_back(Point(0,0) );
      up.push_back(Point(0,0) );
      down.push_back(Point(0,0) );
   }
   PotentialSites(r0, &hole, &up, &down);

   // determine the stacking of a local minimum
   double sqrt3 = sqrt(3.0);
   #pragma omp parallel for
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         
         (*color_red).set(ir, ic, 0.);
         (*color_green).set(ir, ic, 0.);
         (*color_blue).set(ir, ic, 0.);
         
         if (minima.get(ir, ic) == 1.) {
            // fold the matrix coordinate back into the unit cell rectangle
            Point pt(fmod((ic - PAD(0) + NC) % NC, 1.5 * r0) , fmod((ir - PAD(0) + NR) % NR, 1.5 * r0 * sqrt3) );
            // find the minimum distance to each type of site
            // the 3 site types become the 3 R,G,B values in a color vector to represent how the graphene layers are stacked relative to each other
            double nearest_hole_dist = NearestPointDist(pt, hole);
            double nearest_up_dist = NearestPointDist(pt, up);
            double nearest_down_dist = NearestPointDist(pt, down);
            (*color_red).set(ir, ic, nearest_hole_dist);
            (*color_green).set(ir, ic, nearest_up_dist);
            (*color_blue).set(ir, ic, nearest_down_dist);
         }
      }
   }
}

void MakePolygons(std::vector<Polygon>* all_image_polygons, const padded_matrix_t& maxima, const double r0) {
   
   // for each carbon ring, find the bounding atoms
   
      for (auto& poly : (*all_image_polygons)) {
      // find atoms within 2 r0 units of the center of the ring
      int cx = poly.center.x;
      int cy = poly.center.y;
      std::vector<Point> possible_ring_atoms;
      for (int i = floor(-(r0 * 3.)); i < ceil((r0 * 3.)); ++i) {
         for (int j = floor(-(r0 * 3.)); j < ceil((r0 * 3.)); ++j) {
            if ( sqrt( i * i + j * j) <= (r0 * 3.) ) {
               // wrap coordinates into matrix
               // neighbor positions are relative to the center point,
               // i.e., the center is shifted to (0,0)
               if (maxima.get((cy + i + PAD(PAD(NC)) ) % PAD(PAD(NC)), (cx + j + PAD(PAD(NR)) ) % PAD(PAD(NR))) != 0.) {
                  possible_ring_atoms.push_back(Point(j, i) );
               }
            }
         }
      }

      // find the dual of the atoms near the ring
      std::vector<Point> possible_ring_atom_duals;
      for (auto& pt : possible_ring_atoms) {
         double norm_squared = pt.x * pt.x + pt.y * pt.y;
         possible_ring_atom_duals.push_back(Point(pt.x / norm_squared, pt.y / norm_squared) );
      }
      // find the convex hull of the dual of the atoms
      // to eliminate atoms not on the ring
      BoostPolygon hull_me;
      for (int i = 0; i < possible_ring_atom_duals.size(); ++i) {
         hull_me.outer().push_back(BoostPoint(possible_ring_atom_duals[i].x, possible_ring_atom_duals[i].y));
      }
      BoostPolygon hull;
      boost::geometry::convex_hull(hull_me, hull);
      // write boost polygon to a vector of points
      std::vector<Point> actual_ring_atom_duals;
      for ( auto i = hull.outer().begin(); i != hull.outer().end(); ++i ) {
         actual_ring_atom_duals.push_back(Point(boost::geometry::get<0>( *i ), boost::geometry::get<1>( *i )) );
      }
      std::vector<Point> actual_ring_atoms;
      int num_vertices = 0;
      for (auto& pt : actual_ring_atom_duals) {
         double norm_squared = pt.x * pt.x + pt.y * pt.y;
         actual_ring_atoms.push_back(Point(cx + (pt.x / norm_squared), cy + (pt.y / norm_squared)) );
         num_vertices += 1;
      }
      poly.points = actual_ring_atoms;
      poly.num_sides = num_vertices - 1; // subtract 1 because first point in list = last point, to close the shape
   }
}

void WritePolygonFile(const double time, const std::string directory_string, const std::vector<Polygon>& polys) {
   std::string file_str = directory_string + "poly" + boost::lexical_cast<std::string>(floor(time)) + ".txt";
   const char* file_char = file_str.c_str();
   std::ofstream file(file_char);
   if (file.is_open()) {
      file << "{";
      for (auto& p : polys) {
         file << " {" << p.num_sides << ", {" << p.rgb[0] << ", " << p.rgb[1] << ", " << p.rgb[2] << "}, {";
         file << p.center.x << ", " << p.center.y << "}, {";
         for (auto& v : p.points) {
            file << "{" << v.x << ", " << v.y << "}, ";
         }
         file << "{}} }," << std::endl;
      }
      file << " {} }";
      file.close();
      std::cout << "done writing " << file_char << std::endl;
   } else
      std::cout << "Unable to open file" << std::endl;
}

void Analyze(const matrix_t& mat, const double r0, const double time, const std::string directory_string) {
   // pad the matrix
   padded_matrix_t pad;
   
   #pragma omp parallel for
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         double tmp = mat.get( (ir - PAD(0) + NR) % NR, (ic - PAD(0) + NC) % NC);
         pad.set(ir, ic, tmp);
      }
   }
   
   // find local minima in pad by comparing with 8 nearest neighbor pixels
   const double global_max = mat.MaxValue();
   const double global_min = mat.MinValue();
   const double max_minus_min = global_max - global_min;
   // write 1 in minima matrix entries, 0 otherwise
   padded_matrix_t minima, maxima;
   FindExtrema(pad, &minima, &maxima, max_minus_min, global_min);
   
   // find stacking color from minima
   padded_matrix_t color_red, color_green, color_blue;
   FindStacking(minima, &color_red, &color_green, &color_blue, r0);
   
   // initialize a vector of polygons
   std::vector<Polygon> polygons_for_image;
   
   #pragma omp parallel for
   for (int ir = 0; ir < PAD(PAD(NR)); ++ir) {
      for (int ic = 0; ic < PAD(PAD(NC)); ++ic) {
         if (minima.get(ir, ic) == 1.) {
            // initialize with sequential ID, color, and center point
            std::array<double, 3> cols = {color_red.get(ir, ic) / r0, color_green.get(ir, ic) / r0, color_blue.get(ir, ic) / r0};
            Point pt = Point(ic, ir);
            polygons_for_image.push_back(Polygon(cols, pt ) );
         }
      }
   }
   
   MakePolygons(&polygons_for_image, maxima, r0);
   
   // write polygons to a file
   WritePolygonFile(time, directory_string, polygons_for_image);
   
}
