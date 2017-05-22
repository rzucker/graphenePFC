//
//  graphene.cpp
//  graphenePFC
//
//  Created by Rachel Zucker on 7/8/16.
//

#include "matrix_types.hpp"
#include "graphene_functions.hpp"
#include "postprocessing.hpp"
#include "analyze_coupled_layers.hpp"
#include "analyze_coupled_layers_with_burger.hpp"
#include <iostream>
#include <omp.h>
#include <fftw3.h>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>

#define PI 3.141592653589793

// int argc, const char* argv[]
int main() {
   double external_potential_amplitude = 0.009;
   const std::string directory_string = "/Users/Rachel/Documents/graphenePFC/test/fix0";

   const double write_at_these_times[] = {0., 100., 500., 1000., 1500., 2000., 2500., 3000., 3500., 4000., 5000., 6000., 7000., 8000., 9000., 10000.};
   const double max_time = 10000.;
   const double chemical_potential = 0.0;
   const double max_timestep = 1.;
   const double angle = 60.0;

   
   int number_threads = omp_get_max_threads();
   std::cout << "Number of threads available is " << number_threads << std::endl;
   omp_set_num_threads(4);
   #pragma omp parallel
   {
      std::cout << "Number of threads used is " << omp_get_num_threads() << std::endl;
   }
   struct timeval start, stop;
   gettimeofday(&start, NULL);

   const std::string directory_string_fix = directory_string + "Fix"; // write location, fixed layer
   const std::string directory_string_energy_matrix = directory_string + "EMat"; // write location, energy matrix
   const std::string directory_string_energy_list = directory_string + "EList"; // write location, list of total energies
  
   // define length scales
   const double r0 = SCALEUP * 2.12356; // this is 1.22604 times a0, for fourier space
   const double a0 = SCALEUP * 1.73205; // this is Sqrt[3], the primative lattice const, scaleup is the bond length
   
   gettimeofday(&stop, NULL);
   double setup_secs = ((stop.tv_sec  - start.tv_sec) * 1000000u +
            stop.tv_usec - start.tv_usec) / 1.e6;
   
   gettimeofday(&start, NULL);
   // define appled potential, shift chemical_potentialst be zero for inital condition and analyze functions to work properly
   matrix_t applied_potential(a0, -1.0 * external_potential_amplitude, 0.0,
                              1.0, 0.0, angle);
   
   // make initial condition, all IC functions take the same arguments
   matrix_t n_mat;
   InitialPerfect(&n_mat, a0);
   
   // convention:
   // if the hole is over a hole: AA stacking
   // if the the hole is over a down site: AC stacking
   // if the hole is over an up site: AB stacking
   
   // make the FFT of the C2, Cs1, and Cs2 for the calcultion of the 2-point and 3-point correlations
   // these are already in fourier space, not real space.
   matrix_t c2, cs1, cs2, minus_k_squared;
   MakeCs(&c2, &cs1, &cs2, &minus_k_squared, r0, a0);
   gettimeofday(&stop, NULL);
   double initialize_secs = ((stop.tv_sec  - start.tv_sec) * 1000000u +
            stop.tv_usec - start.tv_usec) / 1.e6;
   
   gettimeofday(&start, NULL);
   // outside the time loop: FFTW allocation, plans
   double rescale_fft = 1.0 / sqrt(NR * NC * 1.0); // corrects FFT magnitudes
   const double onethird = 1.0 / 3.0;
   
   // initialize multithreading support, must be first fftw call
   int fftw_init_threads();
   
   // allocate input 2d array to be FFT'd, will be overwritten by FFTW
   fftw_complex *n_hat, *two_pt_correlations, *cs1_n, *cs2_n, *n_cs1_n, *n_cs2_n, *cs1_n_cs1_n, *cs2_n_cs2_n;
   n_hat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NR * NC);
   two_pt_correlations = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NR * NC);
   cs1_n = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NR * NC);
   cs2_n = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NR * NC);
   n_cs1_n = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NR * NC);
   n_cs2_n = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NR * NC);
   cs1_n_cs1_n = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NR * NC);
   cs2_n_cs2_n = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NR * NC);
   
   // subsequent plans will be multithreaded
   fftw_plan_with_nthreads(number_threads);
   // create a FFT plan, size NR x NC, input, output destination, direction of
   // FFT, and let FFTW guess what the best method is (fastest option, rather than doing an exhaustive search)
   fftw_plan make_n_hat, make_two_pt_corrs, make_cs1_n, make_cs2_n, make_n_cs1_n, make_n_cs2_n,
   make_cs1_n_cs1_n, make_cs2_n_cs2_n;
   make_n_hat =
   fftw_plan_dft_2d(NR, NC, n_hat, n_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
   make_two_pt_corrs =
   fftw_plan_dft_2d(NR, NC, two_pt_correlations, two_pt_correlations, FFTW_FORWARD, FFTW_ESTIMATE);
   make_cs1_n = fftw_plan_dft_2d(NR, NC, cs1_n, cs1_n, FFTW_FORWARD, FFTW_ESTIMATE);
   make_cs2_n = fftw_plan_dft_2d(NR, NC, cs2_n, cs2_n, FFTW_FORWARD, FFTW_ESTIMATE);
   make_n_cs1_n =
   fftw_plan_dft_2d(NR, NC, n_cs1_n, n_cs1_n, FFTW_BACKWARD, FFTW_ESTIMATE);
   make_n_cs2_n =
   fftw_plan_dft_2d(NR, NC, n_cs2_n, n_cs2_n, FFTW_BACKWARD, FFTW_ESTIMATE);
   make_cs1_n_cs1_n = fftw_plan_dft_2d(NR, NC, cs1_n_cs1_n, cs1_n_cs1_n, FFTW_FORWARD,
                                       FFTW_ESTIMATE);
   make_cs2_n_cs2_n = fftw_plan_dft_2d(NR, NC, cs2_n_cs2_n, cs2_n_cs2_n, FFTW_FORWARD,
                                       FFTW_ESTIMATE);
   
   // make matrices for the three terms of dF/dN, and dN/dt
   matrix_t dfdn_1;
   matrix_t dfdn_2;
   matrix_t dfdn_3;
   matrix_t dndt;
   
   // create time variable
   double time = 0.;
   double old_time = -0.;
   int print_counter = 0;
   int iteration = 0;
   gettimeofday(&stop, NULL);
   double fftw_init_secs = ((stop.tv_sec  - start.tv_sec) * 1000000u +
            stop.tv_usec - start.tv_usec) / 1.e6;
   std::vector<double> energy_list;
   std::vector<double> time_list;
   
   double conditional_secs, one_pt_secs, two_pt_secs, three_pt_secs, time_advance_secs;
   
   for (; time < max_time; ++iteration) {
      
      if (iteration % 100 == 0) {
         std::cout << "time: " << time << ", iteration: " << iteration << ", dt: " << time - old_time << std::endl;
      }
      
      gettimeofday(&start, NULL);
      // begin 1-point correlation terms, dfdn_1
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double tmp = n_mat.get(ir, ic);
            dfdn_1.set(ir, ic, tmp - (0.5 * tmp * tmp) +
                       (tmp * tmp * tmp * onethird));
         }
      }
      
      // 1-point correlations complete
      gettimeofday(&stop, NULL);
      one_pt_secs = ((stop.tv_sec  - start.tv_sec) * 1000000u +
            stop.tv_usec - start.tv_usec) / 1.e6;
      
      gettimeofday(&start, NULL);
      // begin 2-point correlation terms, dfdn_2
      // define n_hat = FFT[n_mat] using current n_mat value
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            n_hat[ir * NC + ic][0] = n_mat.get(ir, ic);
            n_hat[ir * NC + ic][1] = 0.0;
         }
      }
      
      fftw_execute(make_n_hat);
      
      // define two_pt_correlations using current n_hat value
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double tmp = c2.get(ir, ic);
            two_pt_correlations[ir * NC + ic][0] = rescale_fft * n_hat[ir * NC + ic][0] * tmp;
            two_pt_correlations[ir * NC + ic][1] = rescale_fft * n_hat[ir * NC + ic][1] * tmp;
         }
      }
      
      fftw_execute(make_two_pt_corrs);
      
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            dfdn_2.set(ir, ic, -1.0 * rescale_fft * two_pt_correlations[ir * NC + ic][0]);
         }
      }
      
      // 2-point correlations complete
      gettimeofday(&stop, NULL);
      two_pt_secs = ((stop.tv_sec  - start.tv_sec) * 1000000u +
            stop.tv_usec - start.tv_usec) / 1.e6;
      
      // begin 3-point correlation terms, dfdn_3
      gettimeofday(&start, NULL);
      // define IFFT[ Cs1 nhat]
      // don't forget: (a + b i)(d i) = (-b d) + (a d) i
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double tmp = cs1.get(ir, ic);
            cs1_n[ir * NC + ic][0] =
            -1.0 * rescale_fft * n_hat[ir * NC + ic][1] * tmp;
            cs1_n[ir * NC + ic][1] = rescale_fft * n_hat[ir * NC + ic][0] * tmp;
         }
      }
      
      // define IFFT[ Cs2 nhat]
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double tmp = cs2.get(ir, ic);
            cs2_n[ir * NC + ic][0] =
            -1.0 * rescale_fft * n_hat[ir * NC + ic][1] * tmp;
            cs2_n[ir * NC + ic][1] = rescale_fft * n_hat[ir * NC + ic][0] * tmp;
         }
      }
      
      fftw_execute(make_cs1_n);
      fftw_execute(make_cs2_n);
      
      // define FFT[ n IFFT[Cs1 nhat]]
      // define FFT[ n IFFT[Cs2 nhat]]
      // don't forget: (a)(c + d i) = (a c) + (a d) i
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double tmp = n_mat.get(ir, ic);
            n_cs1_n[ir * NC + ic][0] = rescale_fft * cs1_n[ir * NC + ic][0] * tmp;
            n_cs1_n[ir * NC + ic][1] = rescale_fft * cs1_n[ir * NC + ic][1] * tmp;
            n_cs2_n[ir * NC + ic][0] = rescale_fft * cs2_n[ir * NC + ic][0] * tmp;
            n_cs2_n[ir * NC + ic][1] = rescale_fft * cs2_n[ir * NC + ic][1] * tmp;
         }
      }
      
      fftw_execute(make_n_cs1_n);
      fftw_execute(make_n_cs2_n);
      
      // define IFFT[ Cs1 FFT[ n IFFT[ Cs1 nhat]]]
      // don'tforget: (a + b i)(d i) = (-b d) + (a d) i
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double tmp = cs1.get(ir, ic);
            cs1_n_cs1_n[ir * NC + ic][0] =
            -1.0 * rescale_fft * n_cs1_n[ir * NC + ic][1] * tmp;
            cs1_n_cs1_n[ir * NC + ic][1] =
            rescale_fft * n_cs1_n[ir * NC + ic][0] * tmp;
         }
      }
      
      // define IFFT[ Cs2 FFT[ n IFFT[ Cs2 nhat]]]
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double tmp;
            tmp = cs2.get(ir, ic);
            cs2_n_cs2_n[ir * NC + ic][0] =
            -1.0 * rescale_fft * n_cs2_n[ir * NC + ic][1] * tmp;
            cs2_n_cs2_n[ir * NC + ic][1] =
            rescale_fft * n_cs2_n[ir * NC + ic][0] * tmp;
         }
      }
      
      fftw_execute(make_cs1_n_cs1_n);
      fftw_execute(make_cs2_n_cs2_n);
      
      // make dfdn_3
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double tmp1r = (cs1_n[ir * NC + ic][0] * cs1_n[ir * NC + ic][0]) -
            (cs1_n[ir * NC + ic][1] * cs1_n[ir * NC + ic][1]);
            double tmp2r = (cs2_n[ir * NC + ic][0] * cs2_n[ir * NC + ic][0]) -
            (cs2_n[ir * NC + ic][1] * cs2_n[ir * NC + ic][1]);
            double tmp = rescale_fft * rescale_fft * (tmp1r + tmp2r) -
            2.0 * rescale_fft * cs1_n_cs1_n[ir * NC + ic][0] -
            2.0 * rescale_fft * cs2_n_cs2_n[ir * NC + ic][0];
            dfdn_3.set(ir, ic, tmp);
         }
      }
      
      // 3-point correlations complete
      gettimeofday(&stop, NULL);
      three_pt_secs = ((stop.tv_sec  - start.tv_sec) * 1000000u +
            stop.tv_usec - start.tv_usec) / 1.e6;
   
      gettimeofday(&start, NULL);
      matrix_t energy;
      double total_energy = 0.;
      #pragma omp parallel for reduction(+ : total_energy)
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double local_n = n_mat.get(ir, ic);
            double local_n_squared = local_n * local_n;
            double one_pt_corrs = local_n_squared / 2. -
                            (local_n_squared * local_n) / 6. +
                            (local_n_squared * local_n_squared) / 12.;
            double two_pt_corrs =
          -0.5 * local_n * rescale_fft * two_pt_correlations[0][ir * NC + ic];
            double three_pt_corrs = -1. * onethird * local_n *
                              (pow(rescale_fft * cs1_n[ir * NC + ic][0], 2) +
                               pow(rescale_fft * cs2_n[ir * NC + ic][0], 2));
            energy.set(ir, ic, -1. * (one_pt_corrs + two_pt_corrs + three_pt_corrs -
                             chemical_potential * local_n));
            total_energy += -1. * (one_pt_corrs + two_pt_corrs + three_pt_corrs -
                      chemical_potential * local_n);
         }
      }
      energy_list.push_back(total_energy);
      time_list.push_back(time);
      
      // write the output files, if necessary
      if (time >= write_at_these_times[print_counter] && old_time <= write_at_these_times[print_counter]) {
         WriteMatrix(time, total_energy, directory_string, n_mat);
         WriteMatrix(time, total_energy, directory_string_fix, applied_potential * -3.0);
         AnalyzeCoupledBurger(n_mat, applied_potential * -3.0, a0, time, directory_string, directory_string_fix);
         WriteMatrix(time, iteration, directory_string_energy_matrix, energy);
         print_counter += 1;
      }
      
      gettimeofday(&stop, NULL);
      conditional_secs = ((stop.tv_sec  - start.tv_sec) * 1000000u +
            stop.tv_usec - start.tv_usec) / 1.e6;
      
      gettimeofday(&start, NULL);
      // define change in matrix with time, dN/dt
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double self_potential = -(dfdn_1.get(ir, ic) + dfdn_2.get(ir, ic) -
                                      onethird * dfdn_3.get(ir, ic));
            dndt.set(ir, ic, self_potential + chemical_potential + applied_potential.get(ir, ic));
         }
      }
      
      // find adaptive timestep dt
      double dt = max_timestep / dndt.MaxAbsValue();
      // update n_mat, n_(i+1) = n_i + dt * d(n_i)/dt
      #pragma omp parallel for
      for (int ir = 0; ir < NR; ++ir) {
         for (int ic = 0; ic < NC; ++ic) {
            double tmp = n_mat.get(ir, ic);
            n_mat.set(ir, ic, tmp + dt * dndt.get(ir, ic));
         }
      }
      
      // update time
      old_time = time;
      time += dt;
      gettimeofday(&stop, NULL);
      time_advance_secs = ((stop.tv_sec  - start.tv_sec) * 1000000u +
            stop.tv_usec - start.tv_usec) / 1.e6;
      
      if (iteration % 100 == 0) {
         std::cout << "Operation times:" << std::endl;
         std::cout << "energy + conditionals: " << conditional_secs << ", 1-pt correlations: " << one_pt_secs << ", 2-pt correlations: " << two_pt_secs << ", 3-pt correlations: " << three_pt_secs << ", advance timestep: " << time_advance_secs << std::endl;
      }
   }
   // end time iteration
   
   // clean up allocations for FFTs
   fftw_destroy_plan(make_n_hat);
   fftw_destroy_plan(make_two_pt_corrs);
   fftw_destroy_plan(make_cs1_n);
   fftw_destroy_plan(make_cs2_n);
   fftw_destroy_plan(make_n_cs1_n);
   fftw_destroy_plan(make_n_cs2_n);
   fftw_destroy_plan(make_cs1_n_cs1_n);
   fftw_destroy_plan(make_cs2_n_cs2_n);
   
   fftw_cleanup_threads();
   fftw_free(n_hat);
   fftw_free(two_pt_correlations);
   fftw_free(cs1_n);
   fftw_free(cs2_n);
   fftw_free(n_cs1_n);
   fftw_free(n_cs2_n);
   fftw_free(cs1_n_cs1_n);
   fftw_free(cs2_n_cs2_n);
   
   gettimeofday(&start, NULL);
   
   // write the final timestep to a file
   WriteMatrix(time, iteration, directory_string, n_mat);
   WriteMatrix(time, iteration, directory_string_fix, applied_potential * -3.0);
   AnalyzeCoupledBurger(n_mat, applied_potential * -3.0, a0, time, directory_string, directory_string_fix);
   
   // write the total energy vector to a file
   std::string file_str = directory_string_energy_list + ".txt";
   const char* file_char = file_str.c_str();
   std::ofstream file(file_char);
   if (file.is_open()) {
      file << "{";
      for (int i = 0; i < energy_list.size(); ++i) {
         file << " {" << energy_list.at(i) << ", " << time_list.at(i) << "}, ";
      }
      file << " {} }";
      file.close();
      std::cout << "done writing " << file_char << std::endl;
   } else {
      std::cout << "Unable to open file" << std::endl;
   }
   
   // all calculations complete!
   gettimeofday(&stop, NULL);
   double final_analysis_secs = ((stop.tv_sec  - start.tv_sec) * 1000000u +
            stop.tv_usec - start.tv_usec) / 1.e6;
   
   std::cout << "shared operation times:" << std::endl;
   std::cout << "setup: " << setup_secs << ", initialize matrices: " << initialize_secs << ", intialize fftw: " << fftw_init_secs << ", final analysis: " << final_analysis_secs << std::endl;
   
   return 0;
}
