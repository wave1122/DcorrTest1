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
#include <dgp.h>
//#include <dep_tests.h>

#define CHUNK 1



using namespace std;

//void (*aktfgv)(double *,double *,int *,int *,void *,Matrix&);

int main(void) {

	//start the timer%%%%%
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    auto time = std::chrono::high_resolution_clock::now();
    auto timelast = time;
    /*Matrix alpha(3,1), beta(3,1), lambda(3,1), sigma(2,1);
    alpha(1) = 0.01;
    alpha(2) = 0.04;
    alpha(3) = 0.02;
    beta(1) = 0.7;
    beta(2) = 0.4;
    beta(3) = 0.5;
    lambda(1) = 0.;
    lambda(2) = -1;
    lambda(3) = 0.;
	sigma(1) = 1.5; // cf. Dgp::gen_MixedAR in dgp.h 
	sigma(2) = 1.5; // cf. Dgp::gen_MixedAR in dgp.h 
	int T = 50, TL = 2, lag_smooth = 10;
	Matrix X(T, 1), Y1(T, 1), Y2(T, 1), Y(T, 2), resid(T, 1), std_dev(3, 1), cv(2, 1);
	double rho = 0.5;
    Power pwr_obj;
    ofstream size_out;
    size_out.open ("multivariate_test.txt", ios::out);
    cv = pwr_obj.cValue <daniell_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, 5235231, size_out);
    cout << cv(1) << " , " << cv(2) << endl;
    size_out.close();*/
    /*for (int t = 1; t <= T; ++t) {
    	Y(t,1) = Y1(t);
    	Y(t,2) = Y2(t);
	}
    Dist_corr dist_corr_obj (X,Y);
    cout << dist_corr_obj.do_Test <daniell_kernel> (TL, lag_smooth, 0.5, alpha, beta, lambda, sigma, rho, 523532523) << endl;*/
    //DWORD start = GetTickCount();
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
    # if 0 //begin comment
    /************************************************************************************************************************************************************/
	/**************************************************** Start the Simulations for Other Tests *****************************************************************/
    int T = 100;
    Matrix X(T,1), Y(T,1);
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
	unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    Dep_tests dep_obj;
	Matrix cv_Hong(2, 1), cv_Haugh(2, 1), asymp_cv_Hong(2, 1), asymp_cv_Haugh(2, 1);
	Matrix empir_REJF_Hong(2, 1), asymp_REJF_Hong(2, 1), empir_REJF_Haugh(2, 1), asymp_REJF_Haugh(2, 1);
	asymp_cv_Hong(1) = 1.645;//5%-critical value from the standard normal distribution
    asymp_cv_Hong(2) = 1.28;//10%-critical value from the standard normal distribution
    
    //*************************************************************using the Bartlett kernel**********************************************************************
    int bandW = 5;
    output_filename = "./Results/Others/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <bartlett_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, bartlett_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //************************************************************************************************************************************************************
    bandW = 10;
    output_filename = "./Results/Others/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <bartlett_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, bartlett_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //***********************************************************************************************************************************************************
    bandW = 20;
    output_filename = "./Results/Others/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <bartlett_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, bartlett_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //************************************************************************************************************************************************************
    bandW = 25;
    output_filename = "./Results/Others/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <bartlett_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, bartlett_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    //***********************************************************using the Daniell kernel*************************************************************************
    
	bandW = 5;
    output_filename = "./Results/Others/DN/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/DN/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/DN/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <daniell_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, daniell_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //************************************************************************************************************************************************************
    bandW = 10;
    output_filename = "./Results/Others/DN/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/DN/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/DN/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <daniell_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, daniell_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //***********************************************************************************************************************************************************
    bandW = 20;
    output_filename = "./Results/Others/DN/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/DN/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/DN/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <daniell_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, daniell_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //************************************************************************************************************************************************************
    bandW = 25;
    output_filename = "./Results/Others/DN/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/DN/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/DN/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <daniell_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, daniell_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //******************************************************************using the QS kernel***********************************************************************
    bandW = 5;
    output_filename = "./Results/Others/QS/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/QS/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/QS/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <QS_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, QS_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //************************************************************************************************************************************************************
    bandW = 10;
    output_filename = "./Results/Others/QS/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/QS/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/QS/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <QS_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, QS_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //***********************************************************************************************************************************************************
    bandW = 20;
    output_filename = "./Results/Others/QS/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/QS/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/QS/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <QS_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, QS_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    
    //************************************************************************************************************************************************************
    bandW = 25;
    output_filename = "./Results/Others/QS/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/Others/QS/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/Others/QS/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
	output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values
    dep_obj.cValue <QS_kernel> (cv_Hong, cv_Haugh, T, bandW, alpha, beta, lambda, seed, size_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Hong(1) << " , " << cv_Hong(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << cv_Haugh(1) << " , " << cv_Haugh(2) << ")" << endl;
    output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Hong(1) << " , " << asymp_cv_Hong(2) << ")" << endl;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << ":  " 
	       << "(" << asymp_cv_Haugh(1) << " , " << asymp_cv_Haugh(2) << ")" << endl;
	//calculate rejection rates
    dep_obj.power_f <Dgp, &Dgp::gen_MixedAR, QS_kernel> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_Haugh, asymp_REJF_Haugh, 500, T, bandW, 
                         alpha, beta, lambda, rho, cv_Hong, asymp_cv_Hong, cv_Haugh, asymp_cv_Haugh, seed, pwr_out);
    output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2) << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2) << ")" << endl;
    cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << empir_REJF_Haugh(1) << " , " << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for lag_smooth = " << bandW << ": " << "(" << asymp_REJF_Haugh(1) << " , " << asymp_REJF_Haugh(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
	/************************************************ Finish the Simulations for Other Tests ********************************************************************/
    /************************************************************************************************************************************************************/
	#endif //end comment
	
	
	/**************************************************** Start the Simulations for the Proposed Test: Bivariate Case *******************************************/
    /************************************************************************************************************************************************************/
	int TL = 2;
    int T = 100;
	int number_sampl = 500;
    Matrix alpha(3,1), beta(3,1), lambda(3,1), sigma(2,1);
    alpha(1) = 0.01;
    alpha(2) = 0.04;
    alpha(3) = 0.02;
    beta(1) = 0.7;
    beta(2) = 0.4;
    beta(3) = -0.5;
    lambda(1) = 0.;
    lambda(2) = -1;
    lambda(3) = -0.8;
	sigma(1) = 1.5; // cf. Dgp::gen_AR1 in dgp.h 
	sigma(2) = 1.5; // cf. Dgp::gen_AR1 in dgp.h 
	double rho = 0.2, threshold = 1.54;//set threshold equal to 1.54 for zero correlation
	//double rho = 0.8;
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    Power obj_power;
    int lag_smooth = 5;
    Matrix cv(2, 1), asymp_cv(2, 1), empir_REJF(2, 1), asymp_REJF(2, 1);
    asymp_cv(1) = 1.645;//5%-critical value from the standard normal distribution
    asymp_cv(2) = 1.28;//10%-critical value from the standard normal distribution
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_AR1, bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, threshold, 
	                   cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
    
	//============================================================================================================================================================
	lag_smooth = 10;
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_AR1, bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, threshold, 
	                   cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
    
    //============================================================================================================================================================
    lag_smooth = 20;
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_AR1, bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, threshold, 
	                   cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
    
    //============================================================================================================================================================
    lag_smooth = 25;
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_threshold="
	                  + boost::lexical_cast<std::string>(threshold) + "_bivariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_AR1, bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, threshold, 
	                   cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
	/***************************************** Finish the Simulations for the Proposed Test: Bivariate Case *****************************************************/
    /************************************************************************************************************************************************************/
	
	

	
	#if 0 //begin comment
    /* Import a column of data */
    ifstream data;
    data.open("./Results/BL/size_out_T=200_lag_smooth=25_rho=2.3100000000000001.txt", ios::in);
    if (!data)
        return(EXIT_FAILURE);
    vector<string> data1;
    loadCSV(data, data1);
    int N = data1.size(); //Rows
    Matrix X(N, 1);
    for (auto i = 1; i <= N; i++)
        X(i) = hex2d(data1[i-1]);
    data.close();
    int count10 = 0, count5 = 0;
    for (auto i = 1; i <= N; i++) {
    	if (X(i) >= 1.28) ++count10;
    	if (X(i) >= 1.645) ++count5;
	}
	cout << "N = " << N << endl;
	cout << "10% rejection rate = " << ((double) count10/N) << endl;
    cout << "5% rejection rate = " << ((double) count5/N) << endl;
    //#endif //end comment
    //#if 0 //begin comment
    /**************************************************** Start the Simulations for the Proposed Test: Univariate Case ******************************************/
    /************************************************************************************************************************************************************/
	int TL = 2;
    int T = 200;
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
	//double rho = 0.8;
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    int lag_smooth = 5;
    Matrix cv(2, 1), asymp_cv(2, 1), empir_REJF(2, 1), asymp_REJF(2, 1);
    asymp_cv(1) = 1.645;//5%-critical value from the standard normal distribution
    asymp_cv(2) = 1.28;//10%-critical value from the standard normal distribution
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    Power obj_power;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //*****************************************************************************************************************************************//
    lag_smooth = 10;
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //***************************************************************************************************************************************//
    lag_smooth = 20;
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
    
    //***************************************************************************************************************************************//
    lag_smooth = 25;
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_rho="
	                  + boost::lexical_cast<std::string>(rho) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
    /***************************************** Finish the Simulations for the Proposed Test: Univariate Case ****************************************************/
    /************************************************************************************************************************************************************/
	#endif //end comment
	/*Matrix alpha_x(2,1), alpha_y(2,1), beta_x(2,1), beta_y(2,1);
	alpha_x(1) = 0.001;
	alpha_x(2) = -0.002;
	beta_x(1) = 0.4;
	beta_x(2) = 0.1;
	alpha_y(1) = -0.001;
	alpha_y(2) = 0.002;
	beta_y(1) = 0.9;
	beta_y(2) = 0.78;
    Dgp dgp_obj (alpha_x, alpha_y, beta_x, beta_y);
    //Dgp dgp_obj;
    //dgp_obj.gen_MixedAR (X, Y, alpha, beta, lambda, rho, 543590823);
    Matrix X(50,2), Y(50,2);
    dgp_obj.gen_AR1 (X, Y, 0., 55235);*/
    /*int i = 1, TL = 10;
    while (i <= TL) {
    	x1(i,1) = X(TL-i+1,1);
    	x1(i,2) = X(TL-i+1,2);
    	x2(i,1) = X(TL-i+4,1);
    	x2(i,2) = X(TL-i+4,2);
    	++i;
	}
	x0(1,1) = 0.5;
	x0(1,2) = 0.01;*/
	//NReg nreg_obj;
	//NReg nreg_obj;
	//int lag = -2;
	//int lag_smooth = 10;
	/*int number_sampl = 500;
	Matrix sigma(2,1);
	sigma(1) = sqrt(2);
	sigma(2) = sqrt(0.5 + pow(rho, 2.) * 1.5);
	string output_filename, power_filename;
    power_filename =  "test_power_statistics.txt";
    ofstream pwr_out;
    pwr_out.open (power_filename.c_str(), ios::out);
    Power obj_power;
    double reject_rate = 0.;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, 0.95, 645345, pwr_out);
	cout << reject_rate << endl;*/
	//cout << "rho = " << rho <<": The value of the test statistic is " << dist_corr_obj.do_Test <bartlett_kernel> (TL, lag_smooth, alpha, beta, lambda, sigma)<< endl;
    //cout << "the denominator = " << nreg_obj.weight <epanechnikov_kernel> (X, x1, TL, 0.03) << endl;
    //cout << "the distance covariance test = " << nreg_obj.do_Test <bartlett_kernel,epanechnikov_kernel> (X, Y, 10, 10, 0.03) << endl;
    //cout << "the regression function = " << nreg_obj.var_Ux_ts <triangle_kernel> (X, x1, x2, x0, x0, TL, 0.03) << endl;
    //please do not comment out the lines below.
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    time = std::chrono::high_resolution_clock::now();
    //output << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //cout << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //output << "This program took " << std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count() << " seconds to run.\n";
    cout << "This program took " << std::chrono::duration_cast <std::chrono::minutes> (time-timelast).count() << " minutes to run.\n";
    //output.close ();
    //pwr_out.close ();
    //gsl_rng_free (r);
    #if defined(_WIN64) || defined(_WIN32)
    system("PAUSE");
    #endif
    return 0;
}






