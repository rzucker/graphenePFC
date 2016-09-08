//
//  analyze_coupled_layers_with_burger.hpp
//  graphenePFC
//
//  Created by Rachel Zucker on 9/7/16.
//  Copyright © 2016 Rachel Zucker. All rights reserved.
//

#ifndef analyze_coupled_layers_with_burger_hpp
#define analyze_coupled_layers_with_burger_hpp

#include <stdio.h>
#include "matrix_types.hpp"

void AnalyzeCoupledBurger(const matrix_t& top, const matrix_t& bottom, const double r0, const double time, const std::string directory_string_t, const std::string directory_string_b);

#endif /* analyze_coupled_layers_with_burger_hpp */
