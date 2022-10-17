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
//#include <dgp.h>
//#include <student_reg.h>

#define CHUNK 1



using namespace std;

//void (*aktfgv)(double *,double *,int *,int *,void *,Matrix&);

int main(void) {

	//start the timer%%%%%
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    auto time = std::chrono::high_resolution_clock::now();
    auto timelast = time;
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
    //#if 0 /* the following code blocks will be commented out until #endif */
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
    double nu = 2.5;
	sigma(1) = 1.5; // cf. Dgp::gen_MixedAR in dgp.h 
	sigma(2) = 1.5; // cf. Dgp::gen_MixedAR in dgp.h 
    double cdf = 0.95;
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    int lag_smooth = 20;
    double cv = 0.;
    Matrix reject_rate(2, 1);
    output_filename = "./Results/Student_file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_nu="
	                  + boost::lexical_cast<std::string>(nu) + "_sigma=(" + std::to_string(sigma(1)) + "," + std::to_string(sigma(2)) + ")"
					  + "_pvalue=" + std::to_string(1-cdf) + ".txt";
	size_filename = "./Results/Student_size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_nu="
	                  + boost::lexical_cast<std::string>(nu) + "_sigma=(" + std::to_string(sigma(1)) + "," + std::to_string(sigma(2)) + ")"
					  + "_pvalue=" + std::to_string(1-cdf) + ".txt";
    power_filename = "./Results/Student_power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_nu="
	                  + boost::lexical_cast<std::string>(nu) + "_sigma=(" + std::to_string(sigma(1)) + "," + std::to_string(sigma(2)) + ")"
					  + "_pvalue=" + std::to_string(1-cdf) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    Power obj_power;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, nu, sigma, cdf, seed, size_out);
    output << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    cout << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, nu, sigma, cv, 1.645, seed, pwr_out);
	output << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    cout << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //*****************************************************************************************************************************************//
    lag_smooth = 10;
    output_filename = "./Results/Student_file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_nu="
	                  + boost::lexical_cast<std::string>(nu) + "_sigma=(" + std::to_string(sigma(1)) + "," + std::to_string(sigma(2)) + ")"
					  + "_pvalue=" + std::to_string(1-cdf) + ".txt";
	size_filename = "./Results/Student_size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_nu="
	                  + boost::lexical_cast<std::string>(nu) + "_sigma=(" + std::to_string(sigma(1)) + "," + std::to_string(sigma(2)) + ")"
					  + "_pvalue=" + std::to_string(1-cdf) + ".txt";
    power_filename = "./Results/Student_power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_nu="
	                  + boost::lexical_cast<std::string>(nu) + "_sigma=(" + std::to_string(sigma(1)) + "," + std::to_string(sigma(2)) + ")"
					  + "_pvalue=" + std::to_string(1-cdf) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, nu, sigma, cdf, seed, size_out);
    output << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    cout << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, nu, sigma, cv, 1.645, seed, pwr_out);
	output << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    cout << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();


    //***************************************************************************************************************************************//
    lag_smooth = 5;
    output_filename = "./Results/Student_file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_nu="
	                  + boost::lexical_cast<std::string>(nu) + "_sigma=(" + std::to_string(sigma(1)) + "," + std::to_string(sigma(2)) + ")"
					  + "_pvalue=" + std::to_string(1-cdf) + ".txt";
	size_filename = "./Results/Student_size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_nu="
	                  + boost::lexical_cast<std::string>(nu) + "_sigma=(" + std::to_string(sigma(1)) + "," + std::to_string(sigma(2)) + ")"
					  + "_pvalue=" + std::to_string(1-cdf) + ".txt";
    power_filename = "./Results/Student_power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_nu="
	                  + boost::lexical_cast<std::string>(nu) + "_sigma=(" + std::to_string(sigma(1)) + "," + std::to_string(sigma(2)) + ")"
					  + "_pvalue=" + std::to_string(1-cdf) + ".txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, nu, sigma, cdf, seed, size_out);
    output << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    cout << "Empirical critical value for lag_smooth = " << lag_smooth << ": " << cv << endl;
    reject_rate = obj_power.power_f <bartlett_kernel> (number_sampl, T, TL, lag_smooth, alpha, beta, lambda, nu, sigma, cv, 1.645, seed, pwr_out);
	output << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    cout << "Empirical and asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << reject_rate(1) << " , " << reject_rate(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
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
    //cout << "the distance covariance test = " << nreg_obj.do_Test <QS_kernel,epanechnikov_kernel> (X, Y, 10, 10, 0.03) << endl;
    //cout << "the regression function = " << nreg_obj.var_Ux_ts <triangle_kernel> (X, x1, x2, x0, x0, TL, 0.03) << endl;
    //please do not comment out the lines below.
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    //#endif /* the code blocks above this line will be commented out */
    time = std::chrono::high_resolution_clock::now();
    //output << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //cout << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //output << "This program took " << std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count() << " seconds to run.\n";
    cout << "This program took " << std::chrono::duration_cast <std::chrono::minutes> (time-timelast).count() << " minutes to run.\n";
    //output.close ();
    //pwr_out.close ();
    //gsl_rng_free (r);
    system("PAUSE");
    return 0;
}






