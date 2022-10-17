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

	int L_T = 10; //set the maximum number of autoregression lags
	int poly_degree = 4; //set the degree of B-spline polynomials
	int nbreak = 40; //set the number of break points in [0, 1]
     int ncoeffs = nbreak + poly_degree - 2;

     int max_iter = 200; //set a maximum number of iterations (max_iter) for the Block Coordinate Descent Algorithm used implement the group LASSO
     //set a lower bound and a upper bound for the scaling constant of the optimal penalty parameter in the group LASSO objective function
	auto lb = 0.001, ub = 0.5; //set lb = 0.001 and ub = 0.5 for X
     int ngrid = 200; //set a number of grid points for [lb, ub]
    	auto min_tol = 1e-4; //set a minimum tolerance degree for the Block Coordinate Descent Algorithm

    	auto expn = 1.5; //set an exponent for the Euclidean distance
	int lag_smooth = 5; //set a minimum kernel bandwidth for the proposed test statistic
	int lag_smooth_max = 25; //set a maximum kernel bandwidth for the proposed test statistic

    	unsigned long seed = 134235235; //a seed to generate random numbers
	gsl_rng *r = nullptr;
	const gsl_rng_type *gen; //a random number generator
	gsl_rng_env_setup();
	gen = gsl_rng_taus;
	r = gsl_rng_alloc(gen);
	gsl_rng_set(r, seed);

    /* IMPORT DATA MATRICES*/
	/*ifstream import_X; //import data on the first vector of time series
	import_X.open ("X.csv", ios::in); //there are multiple columns of data
	if (!import_X)
		return(EXIT_FAILURE);
	vector<vector<string>* > data_X; //define an array pointer
	loadCSV(import_X, data_X);
	int T_x = data_X.size(); //Rows
	int N_x = (*data_X[1]).size(); //Columns
	Matrix X(T_x,N_x);
	for (int t = 1; t <= T_x; t++) {
		for (int i = 1; i <= N_x; i++) {
			X(t,i) = hex2d((*data_X[t-1])[i-1]);
		}
	}
	for (vector<vector<string>*>::iterator p1 = data_X.begin( ); p1 != data_X.end( ); ++p1) {
		delete *p1; //free memory
	}
	import_X.close();

	ifstream import_Y; //import data on the second vector of time series
	import_Y.open ("Y.csv", ios::in); //there are multiple columns of data
	if (!import_Y)
		return(EXIT_FAILURE);
	vector<vector<string>* > data_Y; //define an array pointer
	loadCSV(import_Y, data_Y);
	int T = data_Y.size(); //Rows
	ASSERT (T == T_x); //to make sure that both X and Y have the same number of rows
	int N_y = (*data_Y[1]).size(); //Columns
	Matrix Y(T,N_y);
	for (int t = 1; t <= T; t++) {
		for (int i = 1; i <= N_y; i++) {
			Y(t,i) = hex2d((*data_Y[t-1])[i-1]);
		}
	}
	for (vector<vector<string>*>::iterator p2 = data_Y.begin( ); p2 != data_Y.end( ); ++p2) {
		delete *p2; //free memory
	}
	import_Y.close();*/

	//Import a csv file of oil prices
	ifstream import_X;
	import_X.open ("./Application/oilprices_2017.csv", ios::in); //there is only one column of data
	if (!import_X)
		return(EXIT_FAILURE);
	vector<string> data_X;
	loadCSV(import_X, data_X);
	int T_x = data_X.size() - 1; //number of rows to calculate returns
	int N_x = 1;
	Matrix X(T_x,N_x);
	for (auto t = 1; t <= T_x; ++t)
		//X(t) = 100*(hex2d(data_X[t]) - hex2d(data_X[t-1])) / hex2d(data_X[t-1]); //calculate simple oil returns
		X(t) = 100*(log(hex2d(data_X[t])) - log(hex2d(data_X[t-1]))); //calculate log oil returns
	import_X.close(); //close the input stream

	//Import a csv file of stock prices: use 'stockprices_2017.csv' for the U.S. stock market indices and 'data_MSCI_GCC_2017.csv' for the two major GCC countries stock indices
	ifstream import_Y;
	import_Y.open ("./Application/data_MSCI_GCC_2017.csv", ios::in); //there are multiple columns of data
	if (!import_Y)
		return(EXIT_FAILURE);
	vector<vector<string>* > data_Y; //an array pointer
	loadCSV(import_Y, data_Y);
	int T = data_Y.size() - 1; //number of rows
	ASSERT (T == T_x); //to make sure that both X and Y have the same number of rows
	int N_y = (*data_Y[1]).size(); //number of columns
	Matrix Y(T,N_y);
	for (auto t = 1; t <= T; ++t)
		for (auto j = 1; j <= N_y; ++j)
			//Y(t,j) = 100*(hex2d((*data_Y[t])[j-1]) - hex2d((*data_Y[t-1])[j-1])) / hex2d((*data_Y[t-1])[j-1]); //calculate simple stock returns
			Y(t,j) = 100*(log(hex2d((*data_Y[t])[j-1])) - log(hex2d((*data_Y[t-1])[j-1]))); //calculate log stock returns
	for (vector<vector<string>*>::iterator p2 = data_Y.begin( ); p2 != data_Y.end( ); ++p2)
		delete *p2; //free memory
	import_Y.close(); //close the input stream

	//Write returns to file
	/*
	ofstream returns;
	string returns_filename = "./Application/US_returns_2017.csv";
	returns.open (returns_filename.c_str(), ios::out);
	returns << "S&P 500" << " , " << "NYSE Composite" << " , " << "NASDAQ" << " , " << "OK WTI Oil" << endl;
	for (auto t = 1; t <= T; ++t) {
		for (auto j = 1; j <= N_y; ++j) {
			returns << Y(t,j) << " , ";
		}
		returns << X(t) << "\n";
	}
	returns.close(); //close the output stream
	*/
	ofstream returns;
	string returns_filename = "./Application/GCC_returns_2017.csv";
	returns.open (returns_filename.c_str(), ios::out);
	returns << "MSCI EM GCC Countries" << " , " << "MSCI FM GCC Countries" << " , " << "OK WTI Oil" << endl;
	for (auto t = 1; t <= T; ++t) {
		for (auto j = 1; j <= N_y; ++j) {
			returns << Y(t,j) << " , ";
		}
		returns << X(t) << "\n";
	}
	returns.close(); //close the output stream


	int T0 = T- L_T;

	/* Define B-Spline bases for X and Y */
	Matrix Xbs(T,N_x), Ybs(T,N_y);
	Xbs = X; //copy X and Y to Xbs and Ybs respectively
	Ybs = Y;
	do_Norm(Xbs); //normalize data to the 0-1 range
     do_Norm(Ybs);
	gsl_bspline_workspace *bw; //define workspace for B-splines
     bw = gsl_bspline_alloc(poly_degree, nbreak);
     //use uniform breakpoints on [0, 1]
     gsl_bspline_knots_uniform(0., 1., bw);
    //calculate the knots
	//for (auto k = 1; k <= ncoeffs+poly_degree; ++k )
        //cout << "knot = " << k <<  ": " << gsl_vector_get(bw->knots, k-1) << endl;
     Matrix BS_x(N_x*L_T*ncoeffs,T0), BS_y(N_y*L_T*ncoeffs,T0), ave_BS_x(N_x*L_T*ncoeffs,1), ave_BS_y(N_y*L_T*ncoeffs,1);
     gsl_vector *B = gsl_vector_alloc (ncoeffs); //a GSL array pointer
     for (auto  j= 1; j <= N_x; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = L_T+1; t <= T; ++t) {
                         gsl_bspline_eval(Xbs(t-ell,j), B, bw); //evaluate all B-spline basis functions at Xbs(t-ell,j)
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
                         gsl_bspline_eval(Ybs(t-ell,j), B, bw); //evaluate all B-spline basis functions at Ybs(t-ell,j)
                         for (auto k = 1; k <= ncoeffs; ++k) {
						BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t-L_T) = poly_degree * gsl_vector_get(B, k-1) / (nbreak-1); //normalize all the B-splines basis functions
						//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
                         }
               }
          }
     }
     gsl_bspline_free (bw); //free the B-splines workspace
     gsl_vector_free (B); //free memory

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
					//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) << endl;
				}
               }
          }
     }
     ofstream main_out; //define the main output stream
     main_out.open ("./Application/output/main_output.txt", ios::out);

	auto opt_BIC = 0., opt_EBIC = 0.;
	int indx = 0;
	Matrix opt_tuning(2,1);
	auto lambda_aglasso_rate = pow(T0, 0.5); //the penalty parameter for the adaptive group LASSO

	//#if 0
     /* ESTIMATE the B-SPLINE VAR MODEL for X */
     cout << "Start the LASSO procedures to estimate the B-spline VAR model for X. . ." << endl;
     cout << "Refer to X_glasso_est.txt, X_glasso_ic.txt, X_adap_glasso_est.txt, X_adap_glasso_ic.txt, and main_output.txt for all outputs. " << endl;
     double lambda_glasso_rate_x = pow(T0*log(N_x*L_T*ncoeffs), 0.5); //the penalty parameter for the group LASSO

     Matrix mu_X(N_x,1), X0(T0,N_x);
     for (auto t = L_T + 1; t <= T; ++t) {
		for (auto i = 1; i <= N_x; ++i)
			X0(t-L_T,i) = X(t,i); //obtain the data on the response for the first vector of time series
	}
     mu_X = mean(X0);
     for (auto t = 1; t <= T0; ++t) {
		for (auto i = 1; i <= N_x; ++i)
			X0(t,i) = X0(t,i) - mu_X(i); //re-center X0
     }

     int beta_x_nR = N_x*L_T*ncoeffs;
     Matrix beta_init_x(beta_x_nR,1), resp_X(T0,1);
     for (auto i = 1; i <= beta_x_nR; ++i)
		beta_init_x(i) = gsl_ran_ugaussian(r); //set initial values for the B-spline (BS_x) coefficients

	ofstream  X_glasso_out, X_ic_out, X_adap_glasso_out, X_adap_ic_out; //define output streams
	X_glasso_out.open ("./Application/output/X_glasso_est.txt", ios::out);
	X_ic_out.open ("./Application/output/X_glasso_ic.txt", ios::out);
	X_adap_glasso_out.open ("./Application/output/X_adap_glasso_est.txt", ios::out);
	X_adap_ic_out.open ("./Application/output/X_adap_glasso_ic.txt", ios::out);

	Matrix beta_X(beta_x_nR,N_x),  beta_X_i(beta_x_nR,1), beta_X_jl(ncoeffs,1);
	Matrix norm_beta_X(N_x,L_T), BS_xx(beta_x_nR,T0);

	main_out << "******************************* Start the LASSO procedures to estimate the B-spline VAR model for X *******************************************" << endl;
	indx = 1;
     do {
		cout << "Estimating the " << indx << "-th equation using the group LASSO: " << endl;
		main_out <<  "Estimating the " << indx << "-th equation using the group LASSO: " << endl;
		X_glasso_out << "Estimating the " << indx << "-th equation using the group LASSO: " << endl;
		X_ic_out << "Estimating the " << indx << "-th equation using the group LASSO: " << endl;
		for (auto t = 1; t <= T0; ++t)
			resp_X(t) = X0(t, indx); //copy the column number `indx' of X0 to resp_X

		//calculate penalty paramters
		X_glasso_out << "caculating the optimal penalty parameters. . ." << endl;
		X_ic_out << "caculating the optimal penalty parameters. . ." << endl;
		opt_tuning = VAR_gLASSO::calcul_IC (opt_BIC, opt_EBIC, beta_init_x, resp_X, BS_x, lambda_glasso_rate_x, lb, ub, ngrid, N_x, L_T, max_iter, min_tol, X_ic_out, X_glasso_out);
		main_out << "optimal tuning parameters using the BIC and the EBIC for the group LASSO estimator are " << opt_tuning(1) << " and " << opt_tuning(2) << " respectively." << endl;
		main_out << "optimal values of the BIC and the EBIC for the group LASSO estimator are " << opt_BIC << " and " << opt_EBIC << " respectively." << endl;

		//call the group LASSO routine to estimate this model
		beta_X_i = beta_init_x;
		X_glasso_out << "caculating the group LASSO estimates using an optimal penalty parameter. . ." << endl;
		VAR_gLASSO::calcul_gLasso (beta_X_i, resp_X, BS_x, N_x, L_T, opt_tuning(1), max_iter, min_tol, X_glasso_out);
		main_out << "the group LASSO estimates are: " << endl;
		for (auto j = 1; j <= N_x; ++j)
			for (auto ell = 1; ell <= L_T; ++ell)
				for (auto k = 1; k <= ncoeffs; ++k)
					main_out << "j = " << j << ", ell = " << ell << ", k = " << k << ": " <<  beta_X_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) << endl;

		//given the group LASSO estimates above, transform the B-spline basis functions to implement the adaptive group LASSO estimator
		for (auto  j= 1; j <= N_x; ++j) {
			for (auto ell = 1; ell <= L_T; ++ell) {
				for (auto k = 1; k <= ncoeffs; ++k)
					beta_X_jl(k) = beta_X_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k);
				norm_beta_X(j,ell) = ENorm(beta_X_jl); //calculate the Euclidean norm of beta_X_jl
				for (auto t = 1; t <= T0; ++t) {
					for (auto k = 1; k <= ncoeffs; ++k) {
						//multiply B-spline basis functions with the Euclidean norms of the group LASSO estimates in each (j,ell) block
						BS_xx((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) = BS_x((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) * norm_beta_X(j,ell);
						//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
					}
				}
			}
		}

		main_out <<  "Estimating the " << indx << "-th equation using the adaptive group LASSO: " << endl;
		X_adap_glasso_out << "Estimating the " << indx << "-th equation using the adaptive group LASSO: " << endl;
		X_adap_ic_out << "Estimating the " << indx << "-th equation using the adaptive group LASSO: " << endl;

		//calculate penalty parameters
		X_adap_glasso_out << "caculating the optimal penalty parameters. . ." << endl;
		X_adap_ic_out << "caculating the optimal penalty parameters. . ." << endl;
		opt_tuning = VAR_gLASSO::calcul_IC (opt_BIC, opt_EBIC, beta_init_x, resp_X, BS_xx, lambda_aglasso_rate, lb, ub, ngrid, N_x, L_T, max_iter, min_tol, X_adap_ic_out, X_adap_glasso_out);
		main_out << "optimal tuning parameters using the BIC and the EBIC for the adaptive group LASSO estimator are " << opt_tuning(1) << " and " << opt_tuning(2) << " respectively." << endl;
		main_out << "optimal BIC and EBIC for the adaptive group LASSO estimator are " << opt_BIC << " and " << opt_EBIC << " respectively." << endl;

		//given an optimal penalty parameter, call the adaptive group LASSO routine to estimate this model
		beta_X_i = beta_init_x; //re-set the initial parameters
		X_adap_glasso_out << "caculating the adaptive group LASSO estimates using an optimal penalty parameter. . ." << endl;
		VAR_gLASSO::calcul_gLasso (beta_X_i, resp_X, BS_xx, N_x, L_T, opt_tuning(1), max_iter, min_tol, X_adap_glasso_out);
		main_out << "the adaptive group LASSO estimates are: " << endl;
		for (auto j = 1; j <= N_x; ++j)
			for (auto ell = 1; ell <= L_T; ++ell)
				for (auto k = 1; k <= ncoeffs; ++k) {
					//multiply the obtained B-spline coefficients with the Euclidean norms of the group LASSO estimates in each (j,ell) block to yield
					//the estimated slope coefficients of the original B-spline basis functions
					beta_X_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) = beta_X_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) * norm_beta_X(j,ell);
					main_out << "j = " << j << ", ell = " << ell << ", k = " << k << ": " <<  beta_X_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) << endl;
				}
		for (auto i = 1; i <= beta_x_nR; ++i) {
			beta_X(i, indx) = beta_X_i(i); //copy beta_X_i to the column number `indx' in beta_X
		}

		indx += 1;
     } while (indx <= N_x);
	X_glasso_out.close();
	X_ic_out.close();
	X_adap_glasso_out.close();
	X_adap_ic_out.close();

	main_out << "********************************************* LASSO estimation of the B-spline VAR model for X has finished *****************************************" << endl;
	//#endif

	 /* ESTIMATE the B-SPLINE VAR MODEL for Y */
     cout << "Start the LASSO procedures to estimate the B-spline VAR model for Y. . ." << endl;
     cout << "Refer to Y_glasso_est.txt, Y_glasso_ic.txt, Y_adap_glasso_est.txt, Y_adap_glasso_ic.txt, and main_output.txt for all outputs. " << endl;
     double lambda_glasso_rate_y = pow(T0*log(N_y*L_T*ncoeffs), 0.5); //the penalty parameter for the group LASSO

     Matrix mu_Y(N_y,1), Y0(T0,N_y);
     for (auto t = L_T + 1; t <= T; ++t) {
		for (auto i = 1; i <= N_y; ++i)
			Y0(t-L_T,i) = Y(t,i); //obtain the data on the response for the first vector of time series
	}
     mu_Y = mean(Y0);
     for (auto t = 1; t <= T0; ++t) {
		for (auto i = 1; i <= N_x; ++i)
			Y0(t,i) = Y0(t,i) - mu_Y(i); //re-center Y0
     }

     int beta_y_nR = N_y*L_T*ncoeffs;
     Matrix beta_init_y(beta_y_nR,1), resp_Y(T0,1);
     for (auto i = 1; i <= beta_y_nR; ++i)
		beta_init_y(i) = gsl_ran_ugaussian(r); //set initial values for the B-spline (BS_y) coefficients
	gsl_rng_free (r); //free the random generator

	ofstream  Y_glasso_out, Y_ic_out, Y_adap_glasso_out, Y_adap_ic_out; //define output streams
	Y_glasso_out.open ("./Application/output/Y_glasso_est.txt", ios::out);
	Y_ic_out.open ("./Application/output/Y_glasso_ic.txt", ios::out);
	Y_adap_glasso_out.open ("./Application/output/Y_adap_glasso_est.txt", ios::out);
	Y_adap_ic_out.open ("./Application/output/Y_adap_glasso_ic.txt", ios::out);

	Matrix beta_Y(beta_y_nR,N_y), beta_Y_i(beta_y_nR,1), beta_Y_jl(ncoeffs,1);
	Matrix norm_beta_Y(N_y,L_T), BS_yy(beta_y_nR,T0);

	main_out << "********************************* Start the LASSO procedures to estimate the B-spline VAR model for Y *******************************************" << endl;
     indx = 1; //re-set the counter
     do {
		cout << "Estimating the " << indx << "-th equation using the group LASSO: " << endl;
		main_out <<  "Estimating the " << indx << "-th equation using the group LASSO: " << endl;
		Y_glasso_out << "Estimating the " << indx << "-th equation using the group LASSO: " << endl;
		Y_ic_out <<  "Estimating the " << indx << "-th equation using the group LASSO: " << endl;
		for (auto t = 1; t <= T0; ++t)
			resp_Y(t) = Y0(t, indx); //copy the column number `indx' of Y0 to resp_Y

		//calculate penalty paramters
		Y_glasso_out << "caculating the optimal penalty parameters. . ." << endl;
		Y_ic_out << "caculating the optimal penalty parameters. . ." << endl;
		opt_tuning = VAR_gLASSO::calcul_IC (opt_BIC, opt_EBIC, beta_init_y, resp_Y, BS_y, lambda_glasso_rate_y, lb, ub, ngrid, N_y, L_T, max_iter, min_tol, Y_ic_out, Y_glasso_out);
		main_out << "optimal tuning parameters using the BIC and the EBIC for the group LASSO estimator are " << opt_tuning(1) << " and " << opt_tuning(2) << " respectively." << endl;
		main_out << "optimal values of the BIC and the EBIC for the group LASSO estimator are " << opt_BIC << " and " << opt_EBIC << " respectively." << endl;

		//call the group LASSO routine to estimate this model
		beta_Y_i = beta_init_y;
		Y_glasso_out << "caculating the group LASSO estimates using an optimal penalty parameter. . ." << endl;
		VAR_gLASSO::calcul_gLasso (beta_Y_i, resp_Y, BS_y, N_y, L_T, opt_tuning(1), max_iter, min_tol, Y_glasso_out);
		main_out << "the group LASSO estimates are: " << endl;
		for (auto j = 1; j <= N_y; ++j)
			for (auto ell = 1; ell <= L_T; ++ell)
				for (auto k = 1; k <= ncoeffs; ++k)
					main_out << "j = " << j << ", ell = " << ell << ", k = " << k << ": " <<  beta_Y_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) << endl;

		//given the group LASSO estimates above, transform the B-spline basis functions to implement the adaptive group LASSO estimator
		for (auto  j= 1; j <= N_y; ++j) {
			for (auto ell = 1; ell <= L_T; ++ell) {
				for (auto k = 1; k <= ncoeffs; ++k)
					beta_Y_jl(k) = beta_Y_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k);
				norm_beta_Y(j,ell) = ENorm(beta_Y_jl);
				for (auto t = 1; t <= T0; ++t) {
					for (auto k = 1; k <= ncoeffs; ++k) {
						//multiply B-spline basis functions with the Euclidean norms of the group LASSO estimates in each (j,ell) block
						BS_yy((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) = BS_y((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) * norm_beta_Y(j,ell);
						//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
					}
				}
			}
		}

		main_out <<  "Estimating the " << indx << "-th equation using the adaptive group LASSO: " << endl;
		Y_adap_glasso_out << "Estimating the " << indx << "-th equation using the adaptive group LASSO: " << endl;
		Y_adap_ic_out << "Estimating the " << indx << "-th equation using the adaptive group LASSO: " << endl;

		//calculate penalty parameters
		Y_adap_glasso_out << "caculating the optimal penalty parameters. . ." << endl;
		Y_adap_ic_out << "caculating the optimal penalty parameters. . ." << endl;
		opt_tuning = VAR_gLASSO::calcul_IC (opt_BIC, opt_EBIC, beta_init_y, resp_Y, BS_yy, lambda_aglasso_rate, lb, ub, ngrid, N_y, L_T, max_iter, min_tol, Y_adap_ic_out, Y_adap_glasso_out);
		main_out << "optimal tuning parameters using the BIC and the EBIC for the adaptive group LASSO estimator are " << opt_tuning(1) << " and " << opt_tuning(2) << " respectively." << endl;
		main_out << "optimal BIC and EBIC for the adaptive group LASSO estimator are " << opt_BIC << " and " << opt_EBIC << " respectively." << endl;

		//given an optimal penalty parameter, call the adaptive group LASSO routine to estimate this model
		beta_Y_i = beta_init_y;
		Y_adap_glasso_out << "caculating the adaptive group LASSO estimates using an optimal penalty parameter. . ." << endl;
		VAR_gLASSO::calcul_gLasso (beta_Y_i, resp_Y, BS_yy, N_y, L_T, opt_tuning(1), max_iter, min_tol, Y_adap_glasso_out);
		main_out << "the adaptive group LASSO estimates are: " << endl;
		for (auto j = 1; j <= N_y; ++j)
			for (auto ell = 1; ell <= L_T; ++ell)
				for (auto k = 1; k <= ncoeffs; ++k) {
					//multiply the obtained B-spline coefficients with the Euclidean norms of the group LASSO estimates in each (j,ell) block to yield
					//the estimated slope coefficients of the original B-spline basis functions
					beta_Y_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) = beta_Y_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) * norm_beta_Y(j,ell);
					main_out << "j = " << j << ", ell = " << ell << ", k = " << k << ": " <<  beta_Y_i((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) << endl;
				}
		for (auto i = 1; i <= beta_y_nR; ++i) {
			beta_Y(i, indx) = beta_Y_i(i); //copy beta_Y_i to the column number `indx' in beta_X
		}

		indx += 1;
     } while (indx <= N_y);
	Y_glasso_out.close();
	Y_ic_out.close();
	Y_adap_glasso_out.close();
	Y_adap_ic_out.close();
	main_out << "********************************************* LASSO estimation of the B-spline VAR model for Y has finished *****************************************" << endl;

	//#if 0
	cout << "************************************************** Start to calculate the proposed test statistic *******************************************************" << endl;
	main_out << "************************************************** Start to calculate the proposed test statistic *******************************************************" << endl;
	auto stat = 0.;
	/* CALCULATE THE PROPOSED TEST STATISTIC FOR DIFFERENT KERNELS & BANDWIDTHS */
	main_out << "Using the Bartlett kernel: " << endl;
	cout << "Using the Bartlett kernel: " << endl;
	do {
		stat = NGDist_corr::do_Test <bartlett_kernel> (X0, Y0, BS_x, BS_y, beta_X, beta_Y, lag_smooth, expn);
		main_out << "the value of the test statistic for the bandwidth = " << lag_smooth << " is " << stat << endl;
		cout << "the value of the test statistic for the bandwidth = " << lag_smooth << " is " << stat << endl;
		lag_smooth += 5;
	} while (lag_smooth <= lag_smooth_max);

	main_out << "Using the Daniell kernel: " << endl;
	cout << "Using the Daniell kernel: " << endl;
	lag_smooth = 5;
	do {
		stat = NGDist_corr::do_Test <daniell_kernel> (X0, Y0, BS_x, BS_y, beta_X, beta_Y, lag_smooth, expn);
		main_out << "the value of the test statistic for the bandwidth = " << lag_smooth << " is " << stat << endl;
		cout << "the value of the test statistic for the bandwidth = " << lag_smooth << " is " << stat << endl;
		lag_smooth += 5;
	} while (lag_smooth <= lag_smooth_max);

	main_out << "Using the QS kernel: " << endl;
	cout << "Using the QS kernel: " << endl;
	lag_smooth = 5;
	do {
		stat = NGDist_corr::do_Test <QS_kernel> (X0, Y0, BS_x, BS_y, beta_X, beta_Y, lag_smooth, expn);
		main_out << "the value of the test statistic for the bandwidth = " << lag_smooth << " is " << stat << endl;
		cout << "the value of the test statistic for the bandwidth = " << lag_smooth << " is " << stat << endl;
		lag_smooth += 5;
	} while (lag_smooth <= lag_smooth_max);

	main_out.close(); //close the main output stream

	cout << "**************************************************** Calculation of the proposed test statistic has finished *********************************************************" << endl;
	main_out << "************************************************** Calculation of the proposed test statistic has finished *******************************************************" << endl;
	//#endif

    //please do not comment out the lines below.
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    time = std::chrono::high_resolution_clock::now();
    //output << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //cout << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //output << "This program took " << std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count() << " seconds to run.\n";
    auto run_time = std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count();
    cout << "This program took " << run_time << " seconds (" << run_time/60. << " minutes) to run." << endl;

    #if defined(_WIN64) || defined(_WIN32)
    system("PAUSE");
    #endif
    return 0;
}






