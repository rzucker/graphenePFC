//
//  matrix_types.hpp
//  graphenePFC
//
//  Created by Rachel Zucker on 8/18/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#ifndef matrix_types_hpp
#define matrix_types_hpp

#include "matrix.hpp"
#define NR 208
// R = armchair direction
#define NC 280
// C = zigzag direction
#define SCALEUP 11.547
// this is the bond length, in pixels
#define PAD(x) (x+20)

typedef Matrix<NR, NC> matrix_t;
typedef Matrix<PAD(PAD(NR)), PAD(PAD(NC))> padded_matrix_t;

#endif /* matrix_types_hpp */

