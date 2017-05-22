//
//  postprocessing.hpp
//  graphenePFC
//
//  Created by Rachel Zucker on 8/15/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#ifndef postprocessing_hpp
#define postprocessing_hpp

#include "matrix_types.hpp"
#include <array>

struct Point {
public:
   double x;
   double y;
   Point () {
   }
   Point (double c, double r) {
      x = c;
      y = r;
   }
   Point (const Point& p) {
      x = p.x;
      y = p.y;
   }
};

struct Polygon {
public:
   int num_sides;
   std::array<double, 3> rgb;
   Point center;
   std::vector<Point> points;
   
   Polygon (std::array<double, 3> rgbvals, Point centercoords) {
      rgb = rgbvals;
      center = centercoords;
   }
};

struct ScalarAndId {
   int id;
   double scalar;
   double x;
   double y;
   ScalarAndId(int i, double s) {
      id= i;
      scalar = s;
   };
   ScalarAndId(int i, double s, double xp, double yp) {
      id= i;
      scalar = s;
      x = xp;
      y = yp;
   };
};

bool SortByScalar(const ScalarAndId& a, const ScalarAndId& b);

// complete post-processing analysis:
// produce an image that shows stacking of the two layers
void Analyze(const matrix_t& mat, const double a0, const double time, const std::string directory_string);

void FindExtrema(const padded_matrix_t& mat, padded_matrix_t* minima, padded_matrix_t* maxima, const double max_minus_min, const double global_min);

double NearestPointDist(const Point pt, const std::vector<Point>& candidates);

void MakePolygons(std::vector<Polygon>* all_image_polygons, const padded_matrix_t& maxima, const double a0);
void WritePolygonFile(const double time, const std::string directory_string, const std::vector<Polygon>& polys);

#endif /* postprocessing_hpp */
