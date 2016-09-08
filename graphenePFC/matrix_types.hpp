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
#define NR 512
#define NC 512
#define SCALEUP 3.57792
#define PAD(x) (x+20)

typedef Matrix<NR, NC> matrix_t;
typedef Matrix<PAD(PAD(NR)), PAD(PAD(NC))> padded_matrix_t;

#endif /* matrix_types_hpp */
