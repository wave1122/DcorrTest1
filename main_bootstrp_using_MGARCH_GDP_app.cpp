#include <iostream>
#include <fstream>
#include <iomanip>   // format manipulation
#include <string>
#include <sstream>
#include <cstdlib>
#include <math.h>
#include <cmath>
#include <gsl/gsl_math.h>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_bspline.h>
#include <vector> // C++ vector class
#include <algorithm>
#include <functional>
#include <gsl/gsl_randist.h>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/lexical_cast.hpp>
#include <gsl/gsl_rng.h>
#include <unistd.h>
#include <filein.h>
#include <limits>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <chrono>
//#include <windows.h>
#include <omp.h>
#include <matrix_ops2.h>
#include <dist_corr.h>
#include <kernel.h>
#include <power.h>
#include <tests.h>
#include <nl_dgp.h>
#include <nongaussian_reg.h>
#include <dep_tests.h>
#include <nongaussian_dist_corr.h>
#include <VAR_gLasso.h>
//#include <student_reg.h>

#define CHUNK 1



using namespace std;

//void (*aktfgv)(double *,double *,int *,int *,void *,Matrix&);

int main(void) {

	//start the timer%%%%%
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    auto time = std::chrono::high_resolution_clock::now();
    auto timelast = time;

    /*gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, 143523);
    int T, T_max, d_x, d_y, number_sampl, nblocks, bsize, bandwidth, seed;
    double alpha = 1., ddep, pwr = 0.;
    number_sampl = 500;
    T = 50;
    T_max = 500;
    d_x = 4;
    d_y = 4;
    nblocks = 300;
    bandwidth = 5;
    bsize = 30;
    ddep = 0.;*/


    #if 0
    unsigned long seed = 134235235;
    gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    int T = 200;
    int N_x = 3, N_y = 4;
	int poly_degree = 4, nbreak = 5, L_T = 2;
	int ncoeffs = nbreak + poly_degree - 2;
	int T0 = T- L_T;
    Matrix X(T,N_x), Y(T,N_y), beta_x(N_x*L_T*ncoeffs,N_x), beta_y(N_y*L_T*ncoeffs,N_y);

    //generate X
    X(1,1) = gsl_ran_gaussian (r, 0.5);
    X(1,3) = gsl_ran_gaussian(r, 1.5);
    for (auto t = 2; t <= T; ++t) {
        X(t,1) = 0.8*X(t-1,1) + gsl_ran_ugaussian (r);
        X(t,3) = 0.1*sin(X(t-1,3)) + gsl_ran_ugaussian(r);
    }
    X(1,2) = gsl_ran_gaussian (r, 0.5);
    X(2,2) = gsl_ran_gaussian (r, 1.);
	for (auto t = 3; t <= T; ++t) {
		//X(t,2) = 0.8*log(1+3*pow(X(t-1,2), 2.)) - 0.6*log(1+3*pow(X(t-2,2), 2.)) + 1.5*sin(M_PI_2*X(t-1,1)) - 1.*sin(M_PI_4*X(t-2,1)) + gsl_ran_ugaussian(r);
		X(t,2) = 0.8*log(1+3*pow(X(t-1,2), 2.)) - 0.6*log(1+3*pow(X(t-2,2), 2.)) - 1.*sin(M_PI_4*X(t-2,1)) + gsl_ran_ugaussian(r);
	}

	Matrix X2(T,1);
	for (auto t = 1; t <= T; ++t)
		X2(t) = X(t,2);

	//generate Y
	Y(1,1) = gsl_ran_gaussian (r, 0.5);
	Y(1,3) = gsl_ran_gaussian (r, 1.5);
	Y(1,4) = gsl_ran_gaussian (r, 0.8);
	for (auto t = 2; t <= T; ++t) {
		Y(t,1) = 0.2*Y(t-1,1) + gsl_ran_ugaussian (r);
		Y(t,3) = 0.1*cos(Y(t-1,3)) + gsl_ran_ugaussian(r);
          Y(t,4) = 0.7*exp(-fabs(Y(t-1,4))) + gsl_ran_ugaussian(r);
	}
	Y(1,2) = gsl_ran_gaussian (r, 0.5);
	Y(2,2) = gsl_ran_gaussian (r, 1.);
	for (auto t = 3; t <= T; ++t) {
		Y(t,2) = 0.8*log(1+3*pow(Y(t-1,2), 2.)) - 0.6*log(1+3*pow(Y(t-2,2), 2.)) + 1.5*sin(M_PI_2*Y(t-1,1)) - 1.*sin(M_PI_4*Y(t-2,1)) + gsl_ran_ugaussian(r);
	}

	do_Norm(X); //normalize data to the 0-1 range
	do_Norm(Y);

	Matrix X0(T0,N_x), Y0(T0,N_y);
	for (auto t = L_T + 1; t <= T; ++t) {
		for (auto i = 1; i <= N_x; ++i)
			X0(t-L_T,i) = X(t,i);
	}
	for (auto t = L_T+1; t <= T; ++t) {
		for (auto i = 1; i <= N_y; ++i)
			Y0(t-L_T,i) = Y(t,i);
	}

	//assign initial values for all the slope parameters
	Matrix beta_init(N_x*L_T*ncoeffs,1);
	for (auto i = 1; i <= N_x*L_T*ncoeffs; ++i) {
		beta_init(i) = gsl_ran_ugaussian(r);
		for (auto j = 1; j <= N_x; ++j) {
			beta_x(i,j) = gsl_ran_ugaussian(r);
		}
	}
	for (auto i = 1; i <= N_y*L_T*ncoeffs; ++i) {
		for (auto j = 1; j <= N_y; ++j) {
			beta_y(i,j) = gsl_ran_ugaussian(r);
		}
	}


     gsl_bspline_workspace *bw; //define workspace for B-splines
     bw = gsl_bspline_alloc(poly_degree, nbreak);
     //use uniform breakpoints on [0, 1]
     gsl_bspline_knots_uniform(0.0, 1.0, bw);
    //calculate the knots
	//for (auto k = 1; k <= ncoeffs+poly_degree; ++k )
        //cout << "knot = " << k <<  ": " << gsl_vector_get(bw->knots, k-1) << endl;
     Matrix BS_x(N_x*L_T*ncoeffs,T0), BS_y(N_y*L_T*ncoeffs,T0), ave_BS_x(N_x*L_T*ncoeffs,1), ave_BS_y(N_y*L_T*ncoeffs,1);
     gsl_vector *B = gsl_vector_alloc (ncoeffs);
     for (auto  j= 1; j <= N_x; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = L_T+1; t <= T; ++t) {
                         gsl_bspline_eval(X(t-ell,j), B, bw); //evaluate all B-spline basis functions at X(t-ell,j)
                         for (auto k = 1; k <= ncoeffs; ++k) {
						BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t-L_T) = poly_degree * gsl_vector_get(B, k-1) / (nbreak-1); //normalize all the B-splines basis functions
						//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
                         }
               }
          }
     }
      for (auto  j= 1; j <= N_y; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = L_T+1; t <= T; ++t) {
                         gsl_bspline_eval(Y(t-ell,j), B, bw); //evaluate all B-spline basis functions at Y(t-ell,j)
                         for (auto k = 1; k <= ncoeffs; ++k) {
						BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t-L_T) = poly_degree * gsl_vector_get(B, k-1) / (nbreak-1); //normalize all the B-splines basis functions
						//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
                         }
               }
          }
     }
     gsl_bspline_free (bw); //free memory
     gsl_vector_free (B);
     ave_BS_x.set(0.);
     for (auto  j= 1; j <= N_x; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= ncoeffs; ++k) {
				for (auto t = 1; t <= T0; ++t) {
					ave_BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) += BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) / T0; //calculate the temporal averages for all B-spline basis functions (BS_x)
				}
			}
          }
     }
     for (auto  j= 1; j <= N_x; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = 1; t <= T0; ++t) {
				for (auto k = 1; k <= ncoeffs; ++k) {
					//calculate the centered B-spline basis functions for BS_x
					BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) = BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) - ave_BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k);
					//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
				}
               }
          }
     }
      ave_BS_y.set(0.);
     for (auto  j= 1; j <= N_y; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= ncoeffs; ++k) {
				for (auto t = 1; t <= T0; ++t) {
					ave_BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) += BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) / T0; //calculate the temporal averages for all B-spline basis functions (BS_x)
				}
			}
          }
     }
     for (auto  j= 1; j <= N_y; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = 1; t <= T0; ++t) {
				for (auto k = 1; k <= ncoeffs; ++k) {
					//calculate the centered B-spline basis functions for BS_x
					BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) = BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) - ave_BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k);
					//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
				}
               }
          }
     }



	/*int max_iter = 500, ngrid = 5;
	auto min_tol = 1e-4, lambda_glasso_rate = pow(T0*log(N_x*L_T*ncoeffs), 0.5), opt_BIC = 0., opt_EBIC = 0., lb = 0.2, ub = 2.;
	//Matrix opt_tuning(2,1);
	auto lambda1 = pow(T0*log(N_x*L_T*ncoeffs), 0.5), lambda2 = pow(T0, 0.5);
	ofstream  glasso_out, ic_out;
	glasso_out.open ("lasso.txt", ios::out);
	ic_out.open ("ic.txt", ios::out);
	VAR_gLASSO::calcul_adap_gLasso (beta_init, X2, X, L_T, poly_degree, nbreak, lambda1, lambda2, max_iter, min_tol, glasso_out);
	//opt_tuning = VAR_gLASSO::calcul_IC (opt_BIC, opt_EBIC, beta_init, Y, BS, lambda_glasso_rate, lb, ub, ngrid, N, L_T, max_iter, min_tol, ic_out, glasso_out);
	glasso_out.close();
	ic_out.close();*/


	ofstream  reg_out;
	reg_out.open ("euclidean_reg1.txt", ios::out);
	int lag_smooth = 5;
	auto stat = NGDist_corr::do_Test< bartlett_kernel> (X0, Y0, BS_x, BS_y, beta_x, beta_y, lag_smooth, 1.5);
	cout << "the value of the test statistic = " << stat << endl;
	reg_out.close();
     gsl_rng_free (r); //free memory

     #endif

	#if 0
     //compare two data matrices
	ifstream inStream1;//import the first dataset
	inStream1.open ("out_stat1.txt", ios::in);
    if (!inStream1)
        return(EXIT_FAILURE);
    vector<vector<string>* > data1;
    loadCSV(inStream1, data1);
    int T = data1.size(); //Rows
    int N = (*data1[1]).size(); //Columns
    Matrix _X(T,N);
    for (int t = 1; t <= T; t++)
    {
        for (int i = 1; i <= N; i++)
        {
            _X(t,i) = hex2d((*data1[t-1])[i-1]);
        }
    }
    for (vector<vector<string>*>::iterator p1 = data1.begin( ); p1 != data1.end( ); ++p1)
    {
      delete *p1;
    }
    inStream1.close();

    ifstream inStream2;//import the second dataset
	inStream2.open ("out_stat.txt", ios::in);
    if (!inStream2)
        return(EXIT_FAILURE);
    vector<vector<string>* > data2;
    loadCSV(inStream2, data2);
    T = data2.size(); //Rows
    N = (*data2[1]).size(); //Columns
    Matrix _Y(T,N);
    for (int t = 1; t <= T; t++)
    {
        for (int i = 1; i <= N; i++)
        {
            _Y(t,i) = hex2d((*data2[t-1])[i-1]);
        }
    }
    for (vector<vector<string>*>::iterator p2 = data2.begin( ); p2 != data2.end( ); ++p2)
    {
      delete *p2;
    }
    inStream2.close();
    bool res = true;
    res = compare(_X,_Y);
    cout << "res = " << res << endl;
    #endif


    # if 0
    int TL = 2;
    int T = 100;
	int number_sampl = 500;
    Matrix X(T,1), Y(T,1), x1(TL,2), x2(TL,2), x0(1,2);
    Matrix alpha(2,1), beta(2,1), lambda(2,1), sigma(2,1);
    alpha(1) = 0.01;
    alpha(2) = 0.04;
    beta(1) = 0.7;
    beta(2) = 0.4;
    lambda(1) = 0.;
    lambda(2) = -1;
	sigma(1) = 1.5; // cf. Dgp::gen_MixedAR in dgp.h
	sigma(2) = 1.5; // cf. Dgp::gen_MixedAR in dgp.h
	double rho = 1.54 * sigma(1); // set rho equal to 1.54 for zero correlation
    double cdf = 0.95;
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    int lag_smooth = 20;
    double cv = 0.;
    Matrix reject_rate(2, 1);
    output_filename = "./Results/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
	size_filename = "./Results/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    power_filename = "./Results/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    Power obj_power;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, cdf, seed, size_out);
    output << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    cout << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, 1.645, seed, pwr_out);
	output << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    cout << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //*****************************************************************************************************************************************//
    lag_smooth = 10;
    output_filename = "./Results/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
	size_filename = "./Results/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    power_filename = "./Results/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, cdf, seed, size_out);
    output << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    cout << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, 1.645, seed, pwr_out);
	output << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    cout << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //***************************************************************************************************************************************//
    lag_smooth = 5;
    output_filename = "./Results/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
	size_filename = "./Results/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    power_filename = "./Results/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + "_pvalue=" + boost::lexical_cast<std::string>(1-cdf) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, cdf, seed, size_out);
    output << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    cout << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, 1.645, seed, pwr_out);
	output << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    cout << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
    # endif

	# if 0
	 int TL = 2;
     int T = 100;
     int N = 500;
	 int num_B = 1000;
	 double expn = 1.5, rho12 = -0.5, rho13 = 0.2;
	 int choose_alt = 1;
     unsigned long seed = 9436434;
     int lag_smooth = 5; //set M_T = 5, 10, 20, and 25
     Matrix asymp_CV(2,1), empir_REJF(2,1), empir_CV(2,1), asymp_REJF(2,1), alpha_X(5,1), alpha_Y(5,2), epsilon_t(N,1), epsilon_s(N,1), eta1_t(N,1), eta1_s(N,1), eta2_t(N,1),
                 eta2_s(N,1), delta(3,1);
     asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
     asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
     alpha_X(1) = 0.01; //define the AR coefficients
     alpha_X(2) = 0.7;
     alpha_X(3) = 0.1;
     alpha_X(4) = -0.6;
     alpha_X(5) = 0.3;
     alpha_Y(1,1) = 0.04;
     alpha_Y(2,1) = 0.4;
     alpha_Y(3,1) = 0.5;
     alpha_Y(4,1) = 0.1;
     alpha_Y(5,1) = -0.6;
     alpha_Y(1,2) = 0.01;
     alpha_Y(2,2) = -0.2;
     alpha_Y(3,2) = 0.1;
     alpha_Y(4,2) = 0.05;
     alpha_Y(5,2) = 0.4;

     delta(1) = 0.1;
     delta(2) = 0.9;
     delta(3) = -0.1;
     NL_Dgp::gen_RANV<NL_Dgp::gen_TriSN> (epsilon_t, eta1_t, eta2_t, delta, 0., 0., 0, 13435); //generate independent skewed-normal random errors
     NL_Dgp::gen_RANV<NL_Dgp::gen_TriSN> (epsilon_s, eta1_s, eta2_s, delta, 0., 0., 0, 53435);

	int size_alpha_X = alpha_X.nRow(), size_alpha_Y = alpha_Y.nRow();
	Matrix X(T,1), Y1(T,1), Y2(T,1), alpha_X_hat(size_alpha_X,1), alpha_Y1_hat(size_alpha_Y,1), alpha_Y2_hat(size_alpha_Y,1), alpha_Y_hat(size_alpha_Y,2),
			  resid_X(T-2,1), resid_Y1(T-2,1), resid_Y2(T-2,1);

     NL_Dgp::gen_TAR<NL_Dgp::gen_TriSN>  (X, Y1, Y2, alpha_X, alpha_Y, delta, rho12, rho13, choose_alt, seed); //generate data

	//then use these samples to estimate the AR coefficients
	NL_Dgp::est_TAR (resid_X, alpha_X_hat, X);
	NL_Dgp::est_TAR (resid_Y1, alpha_Y1_hat, Y1);
	NL_Dgp::est_TAR (resid_Y2, alpha_Y2_hat, Y2);
	for (int j = 1; j <= size_alpha_Y; j++) {
		alpha_Y_hat(j,1) = alpha_Y1_hat(j); //collect all the OLS estimates in a matrix
		alpha_Y_hat(j,2) = alpha_Y2_hat(j);
	}

	Matrix cstat_bootstrap(num_B, 1);
	double cstat = 0.;
	Matrix xi_x(num_B, T), xi_y(num_B, T);
	gsl_rng *r = nullptr;
     const gsl_rng_type *gen; //random number generator
     gsl_rng_env_setup();
     gen = gsl_rng_taus;
     r = gsl_rng_alloc(gen);
     gsl_rng_set(r, 145325);
	for (auto i = 1; i <= num_B; i++) {
			for (auto t = 1; t <= T; t++) {
				xi_x(i,t) = gsl_ran_ugaussian (r);
				xi_y(i,t) = gsl_ran_ugaussian (r);
			}
	}
	gsl_rng_free (r); //free memory

	auto pvalue = NGDist_corr::calcul_Pvalue <bartlett_kernel, NGReg::cmean_TAR> (cstat, cstat_bootstrap, X, Y1, Y2, TL, lag_smooth, alpha_X_hat, alpha_Y_hat,
																									epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t, eta2_s, xi_x, xi_y, expn);

	ofstream  bootstrap;
	bootstrap.open ("boot_stat.txt", ios::out);
	for (int i = 1; i <= num_B; i++)
			bootstrap << cstat_bootstrap(i) << endl;
	bootstrap << "the value of the statistics = " << cstat << endl;
	cout << "p-value = " << pvalue<< endl;
	bootstrap << "p-value = " << pvalue<< endl;
     bootstrap.close();
     # endif

	//# if 0
	/**************************************************** IMPORT THE FIRST DATASET *************************************************************/
     ifstream readX; //GDP
	readX.open ("E:\\Copy\\SCRIPTS\\Exogeneity\\Application\\data\\GDP\\US2\\gdp_resid1.csv", ios::in);
	if (!readX)
        return(EXIT_FAILURE);
    vector<vector<string>* > matX;
    loadCSV(readX, matX);
    int T = matX.size(); //Rows
    int N = 1; //number of indices
    int colX = (*matX[1]).size(); //Columns
    Matrix X(T, N), cmean_X(T, N), resid_X(T, N), csigma_X(T, colX - 3*N) ;

    for (int t = 1; t <= T; t++) {
		for (int i = 1; i <= N; i++) {
			X(t,i) = hex2d((*matX[t-1])[i-1]);
			cmean_X(t, i) = hex2d((*matX[t-1])[N+i-1]);
			resid_X(t, i) = hex2d((*matX[t-1])[2*N+i-1]);
         }
         for (int i = 3*N+1; i <= colX; i++) {
			csigma_X(t, i-3*N) = hex2d((*matX[t-1])[i-1]);
         }
    }

    for (vector<vector<string>*>::iterator p1 = matX.begin( ); p1 != matX.end( ); ++p1)
    {
		delete *p1;
    }
    readX.close();

    /***********************************************************************************************************************************************/
    /*************************************************** IMPORT THE SECOND DATASET **************************************************************/
     ifstream readY; //yield spread, business loan demand, and NFCI
	readY.open ("E:\\Copy\\SCRIPTS\\Exogeneity\\Application\\data\\GDP\\US2\\fin_resid1.csv", ios::in);
	if (!readY)
        return(EXIT_FAILURE);
    vector<vector<string>* > matY;
    loadCSV(readY, matY);
    T = matY.size(); //Rows
    N = 3; //number of features
    int colY = (*matY[1]).size(); //Columns
    Matrix Y(T, N), cmean_Y(T, N), resid_Y(T, N), csigma_Y(T, colY - 3*N) ;

    for (int t = 1; t <= T; t++) {
		for (int i = 1; i <= N; i++) {
			Y(t,i) = hex2d((*matY[t-1])[i-1]);
			cmean_Y(t, i) = hex2d((*matY[t-1])[N+i-1]);
			resid_Y(t, i) = hex2d((*matY[t-1])[2*N+i-1]);
         }
         for (int i = 3*N+1; i <= colY; i++) {
			csigma_Y(t, i-3*N) = hex2d((*matY[t-1])[i-1]);
         }
    }

    for (vector<vector<string>*>::iterator p1 = matY.begin( ); p1 != matY.end( ); ++p1)
    {
		delete *p1;
    }
    readY.close();
 /************************************************************************************************************************************************/

    int num_B = 1000; //number of bootstrap samples to be generated
    std::vector<int> lag_smooth = {5, 10, 15, 20, 25}; //set a kernel bandwidth: 5, 10, 15, 20, 25, 30, 35, 40, 45
    double expn = 1.5; //set an exponent for the distance correlation
    Matrix xi_x(num_B, T), xi_y(num_B, T); //generate two independent sequences of i.i.d. standard normal random variables
	gsl_rng *r = nullptr;
     const gsl_rng_type *gen; //random number generator
     gsl_rng_env_setup();
     gen = gsl_rng_taus;
     r = gsl_rng_alloc(gen);
     unsigned long seed = 1342353; // 145325;
     gsl_rng_set(r, seed);
	for (auto i = 1; i <= num_B; i++) {
			for (auto t = 1; t <= T; t++) {
				xi_x(i,t) = gsl_ran_ugaussian (r);
				xi_y(i,t) = gsl_ran_ugaussian (r);
			}
	}
	gsl_rng_free (r); //free memory

	auto cstat = 0., pvalue = 0.;
	Matrix cstat_bootstrap(num_B, 1);
	ofstream  output; //open an output stream

	for (auto ii : lag_smooth) {
			string output_filename = "E:\\Copy\\SCRIPTS\\Exogeneity\\Application\\output\\GDP\\US2\\results_T=" + std::to_string(T) + "_lag_smooth=" + std::to_string(ii) + "_expn="
												+ std::to_string(expn) +  ".txt";
			output.open (output_filename.c_str(), ios::out);
			 pvalue = NGDist_corr::calcul_Pvalue <QS_kernel> (cstat, cstat_bootstrap, X, Y, cmean_X, cmean_Y, csigma_X, csigma_Y, resid_X, resid_Y, ii, xi_x,  xi_y, expn);
			output << "p-value = " << pvalue << endl;
			output << "the value of the distance-based statistic = " << cstat << endl;
			output << "The bootstrap statistics: " << endl;
			for (int i = 1; i <= num_B; i++)
				output << cstat_bootstrap(i) << endl;
			output.close();
	}
	//# endif

    //please do not comment out the lines below.
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    time = std::chrono::high_resolution_clock::now();
    //output << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //cout << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //output << "This program took " << std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count() << " seconds to run.\n";
    auto duration =  std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count();
    cout << "This program took " << duration << " seconds (" << duration/60. << " minutes) to run." << endl;
    //output.close ();
    //pwr_out.close ();
    //gsl_rng_free (r);
    system("PAUSE");
    return 0;
}






