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
#define NR 2314
#define NC 64
#define SCALEUP 4.42987
#define PAD(x) (x+20)

typedef Matrix<NR, NC> matrix_t;
typedef Matrix<PAD(PAD(NR)), PAD(PAD(NC))> padded_matrix_t;

#endif /* matrix_types_hpp */

// 256:
// 6.7086
//
// 512:
// 3.57792
//
// 1024:
// 4.12837
//
// 2048:
// 3.83348
//
// 8192:
// 4.10861
//
// 15 pixels per lattice const:
// 4.71698
