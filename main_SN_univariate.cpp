#include <iostream>
#include <fstream>
#include <iomanip>   // format manipulation
#include <string>
#include <sstream>
#include <cstdlib>
#include <math.h>
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_test.h>
#include <gsl/gsl_multimin.h>
#include <vector> // C++ vector class
#include <algorithm>
#include <functional>
#include <gsl/gsl_randist.h>
//#include <boost/numeric/conversion/cast.hpp>
//#include <boost/lexical_cast.hpp>
#include <gsl/gsl_rng.h>
#include <unistd.h>
#include <filein.h>
//#include <csvreader.h>
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
//#include <tests.h>
#include <dgp.h>
#include <nl_dgp.h>
#include <nongaussian_reg.h>
#include <dep_tests.h>
#include <nongaussian_dist_corr.h>
#include <nongaussian_power.h>


using namespace std;

//void (*aktfgv)(double *,double *,int *,int *,void *,Matrix&);

int main(void) {

	//start the timer%%%%%
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    auto time = std::chrono::high_resolution_clock::now();
    auto timelast = time;

    #if 0 //begin comment
    auto T = 100, N = 1000, TL = 2, lag_smooth = 5;
    Matrix X(T,1), Y1(T,1), Y2(T,1), resid(T-2,1), slope(5,1), alpha_X(6,1), alpha_Y(6,1);
    alpha_X(1) = 0.01;
    alpha_X(2) = 0.5;
    alpha_X(3) = -0.2;
    alpha_X(4) = -0.1;
    alpha_X(5) = 0.2;
    alpha_X(6) = 0.1;
    alpha_Y(1,1) = 0.04;
    alpha_Y(2,1) = 0.7;
    alpha_Y(3,1) = -0.1;
    alpha_Y(4,1) = -0.3;
    alpha_Y(5,1) = 0.1;
    alpha_Y(6,1) = 0.1;
    /*alpha_Y(1,2) = 0.05;
    alpha_Y(2,2) = 0.8;
    alpha_Y(3,2) = -0.1;
    alpha_Y(4,2) = -0.5;
    alpha_Y(5,2) = -0.1;
    alpha_Y(6,2) = 0.1;*/
    double rho12 = 0.4, rho13 = -0.4, res_X = 0., res_Y = 0.;
	Matrix lag_t1(3,1), lag_t2(3,1), xy_t(3,1), xy_s(3,1), lag_s1(3,1), lag_s2(3,1), delta(3,1);
	lag_t1(1) = 0.85;
	lag_t1(2) = 0.7;
	lag_t1(3) = -0.6;
	lag_t2(1) = -0.5;
	lag_t2(2) = 0.53;
	lag_t2(3) = 0.8;
	lag_s1(1) = -0.85;
	lag_s1(2) = 0.5;
	lag_s1(3) = -0.8;
	lag_s2(1) = 0.23;
	lag_s2(2) = -0.95;
	lag_s2(3) = 0.8;
	xy_t(1) = 0.1;
	xy_t(2) = 0.8;
	xy_t(3) = -0.7;
	xy_s(1) = -0.5;
	xy_s(2) = 0.6;
	xy_s(3) = 1.5;
	delta(1) = 0.7;
	delta(2) = 0.5;
	delta(3) = -0.6;
    //NGReg::breg <NL_Dgp::gen_TriSN, NGReg::cmean_TAR> (&res_X, &res_Y, lag_t1, lag_t2, lag_s1, lag_s2, alpha_X, alpha_Y, delta, rho12, rho13, 1.5, r, 1325325);
    //NGReg::var_U_ts <NL_Dgp::gen_SN, NGReg::cmean_TAR> (&res_X, &res_Y, xy_t, xy_s, lag_t1, lag_t2, lag_s1, lag_s2, alpha_X, alpha_Y,0.7, -0.5, 1.5, r, 1342525);
    //NGReg::var_U_ts <NL_Dgp::gen_SN, NGReg::cmean_BL> (res_X, res_Y, xy_t, xy_s, lag_t1, lag_t2, lag_s1, lag_s2, alpha_X, alpha_Y, delta(1), 1.5, 1352543564);
	//cout << res_X << " , " << res_Y << endl;
	//generate data
	/*unsigned long seed = 6436;
	Matrix data_X(T,1), data_Y1(T,1), data_Y2(T,1), epsilon_t(N,1), eta1_t(N,1), eta2_t(N,1), epsilon_s(N,1), eta1_s(N,1), eta2_s(N,1);
    NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriSN> (data_X, data_Y1, data_Y2, alpha_X, alpha_Y, delta, rho12, rho13, 1, seed);
    NL_Dgp::gen_RANV<NL_Dgp::gen_TriSN> (epsilon_t, eta1_t, eta2_t, delta, 0., 0., 0, 13435);
    NL_Dgp::gen_RANV<NL_Dgp::gen_TriSN> (epsilon_s, eta1_s, eta2_s, delta, 0., 0., 0, 53435);
    cout << epsilon_t(1) << " , " << epsilon_s(1) << " , " << eta1_t(1) << " , " << eta1_s(1) << " , " << eta2_t(1) << " , " << eta2_s(1) << endl;
    auto kernel_QDSum = 0., kernel_QRSum = 0.;
    NGDist_corr::integrate_Kernel <daniell_kernel> (kernel_QDSum, kernel_QRSum);
	cout << NGDist_corr::do_Test<daniell_kernel, NGReg::cmean_BL> (data_X, data_Y1, data_Y2, TL, lag_smooth, kernel_QRSum, alpha_X, alpha_Y, epsilon_t,
	                                                               epsilon_s, eta1_t, eta1_s, eta2_t, eta2_s, 1.5) << endl;*/



	Matrix asymp_REJF(2,1), empir_CV(2,1), asymp_CV(2,1), epsilon_t(N,1), epsilon_s(N,1), eta_t(N,1), eta_s(N,1);
	auto number_sampl = 500;
	auto expn = 1.5;
	asymp_CV(1) = 1.645;//5%-critical value from the standard normal distribution
    asymp_CV(2) = 1.28;//10%-critical value from the standard normal distribution
    ofstream size_out;
    string size_out_filename = "./temp/size.txt";
	size_out.open (size_out_filename.c_str(), ios::out);
	NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_t, eta_t, delta(1), 0., 0, 13435); //generate random errors
    NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_s, eta_s, delta(1), 0., 0, 53435);
    NGPower::cValue<bartlett_kernel,NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>,NGReg::cmean_BL, NL_Dgp::est_BL> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                           alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
																										   delta(1), expn, asymp_CV, 1234235, size_out);
    size_out.close();
    /*auto T = 100;
    Matrix X(T, 1), Y(T, 1), Y2(T, 1), resid(T-2,1), slope(5,1), alpha_X(5, 1), alpha_Y(5, 2), delta(3,1);
    alpha_X(1) = 0.01;
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
    delta(1) = 0.7;
    delta(2) = 0.5;
    delta(3) = -0.6;
    double rho12 = -0.2056, rho13 = 0.2671;
    gsl_rng *r = nullptr;
	int choose_alt = 2;
    //NL_Dgp::gen_TAR<NL_Dgp::gen_TriSN>(X, Y1, Y2, alpha_X, alpha_Y, delta, rho12, rho13, choose_alt, 1242432);
    NL_Dgp::gen_TAR<NL_Dgp::gen_SN>(X, Y, alpha_X, alpha_Y, delta(1), rho12, choose_alt, 12424);
    NL_Dgp::est_TAR (resid, slope, Y);
    ofstream output;
	output.open ("./test_output.txt", ios::out);
    for (int t = 1; t <= X.nRow(); t++)
        output << X(t) << " , " << Y(t) << endl;
    output << "----------------------------------------------------------------------\n";
    for (int t = 1; t <= resid.nRow(); ++t)
        output << resid(t) << endl;
    output << "slope: " << endl;
    output << slope(1) << endl;
    output << slope(2) << endl;
    output << slope(3) << endl;
    output << slope(4) << endl;
    output << slope(5) << endl;
    output.close();
    gsl_rng_free (r); //free memory*/
    #endif //end comment

    #if 0 //begin comment
    /***********************************************************************************************************************************************************/
    /*********************************************************** Start Calculating the Proposed Test Using Real Data *******************************************/
	//Import a csv file of oil prices
    ifstream oildata;
    oildata.open ("./Application/oilprices_1977-1978.csv", ios::in);
    if (!oildata)
        return(EXIT_FAILURE);
    vector<string> oildata1;
    loadCSV(oildata, oildata1);
    int T = oildata1.size() - 1; //Rows
    Matrix X(T, 1);
    for (auto i = 1; i <= T; i++)
        X(i) = (hex2d(oildata1[i]) - hex2d (oildata1[i-1])) / hex2d (oildata1[i-1]);//calculate oil returns
    oildata.close();
    //Import a csv file of stock prices
    ifstream stockdata;
    stockdata.open ("./Application/stockprices_1977-1978.csv", ios::in);
    if (!stockdata)
        return(EXIT_FAILURE);
    vector<vector<string>* > stockdata1;
    loadCSV(stockdata, stockdata1);
    int nC = (*stockdata1[1]).size(); //Columns
    Matrix Y(T, nC);
    for (auto i = 1; i <= T; ++i)
    {
        for (auto j = 1; j <= nC; ++j)
        {
            Y(i,j) = (hex2d ((*stockdata1[i])[j-1]) - hex2d ((*stockdata1[i-1])[j-1])) / hex2d ((*stockdata1[i-1])[j-1]);//calculate stock returns
        }
    }
    for (vector<vector<string>*>::iterator p = stockdata1.begin( ); p != stockdata1.end( ); ++p)
    {
        delete *p;
    }
    stockdata.close();
    //Write returns to file
    ofstream returns;
    string returns_filename = "./Application/returns_1977-1978.csv";
	returns.open (returns_filename.c_str(), ios::out);
	returns << "S&P 500" << " , " << "NYSE Composite" << " , " << "NASDAQ" << " , " << "OK WTI Oil" << endl;
	for (auto i = 1; i <= T; ++i)
    {
        for (auto j = 1; j <= nC; ++j)
        {
            returns << Y(i,j) << " , ";
        }
        returns << X(i) << "\n";
    }
	returns.close();
    /*ofstream returns;
    string returns_filename = "./Application/returns.csv";
	returns.open (returns_filename.c_str(), ios::out);
	returns << "S&P 500" << " , " << "NYSE Composite" << " , " << "NASDAQ" << " , " << "OK WTI Oil Spot Price" << endl;
	for (auto i = 1; i <= T; ++i)
    {
        for (auto j = 1; j <= nC; ++j)
        {
            returns << Y(i,j) << " , ";
        }
        returns << X(i) << "\n";
    }
	returns.close();*/

    int lag_len = 20; //AR lag length
    int lag_smooth = 20; //a bandwidth for the proposed statistic
    Dgp dgp_obj;
    Matrix resid_X(T-lag_len, 1), slope_X(lag_len+1, 1), Yj(T, 1), resid_Yj(T-lag_len, 1), slope_Yj(lag_len+1, 1), resid_Y(T-lag_len, nC);
    double SSR = 0.;
    dgp_obj.est_AR (resid_X, slope_X, SSR, X, lag_len);//fit X to an AR(lag_len) process, and obtain a sequence of residuals
    for (auto j = 1; j <= nC; ++j) {
    	for (auto t = 1; t <= T; ++t) {
    		Yj(t) = Y(t, j);
		}
		dgp_obj.est_AR (resid_Yj, slope_Yj, SSR, Yj, lag_len);//fit Yj to an AR(lag_len) process, and obtain a sequence of residuals
		for (auto t = 1; t <= T-lag_len; ++t) {
			resid_Y(t, j) = resid_Yj(t);
		}
	}
	ofstream output;
    string output_filename = "./Application/file_out_T=" + std::to_string(T) + "_dy=" + std::to_string(nC) + "_lag_len=" + std::to_string(lag_len)
	                         + "_lag_smooth=" + std::to_string(lag_smooth) +  ".txt";
	output.open (output_filename.c_str(), ios::out);
	output << "El Himdi & Roy's (1997) (HR) test: " << endl;
	cout << "El Himdi & Roy's (1997) (HR) test: " << endl;
	Dep_tests dep_obj;
	int bandw = 10; //a bandwidth for El Himdi & Roy's (1997) statistic
	do {
	    output << "bandwidth = " << bandw << ": HR = " << dep_obj.do_ElHimdiRoyTest (bandw, resid_X, resid_Y) << endl;
	    bandw += 2;
	} while (bandw <= 30);
    int TL = 2;//truncation lag for kernel regressions
    double kernel_QDSum = 0., kernel_QRSum = 0., bandw_reg = pow(T, -0.6);//a bandwidth for nonparametric regressions
    Dist_corr obj_dc (X,Y);
    obj_dc.integrate_Kernel <bartlett_kernel> (&kernel_QDSum, &kernel_QRSum);//integrating a kernel function
    output << "The proposed distance correlation (DC) test: " << endl;
    cout << "The proposed distance correlation (DC) test: " << endl;
    do {
	    output << "truncation lag = " << TL << ": DC = " << obj_dc.do_Test <bartlett_kernel,epanechnikov_kernel> (TL, lag_smooth, kernel_QRSum, bandw_reg)  << endl;
	    TL += 1;
	} while (TL <= 15);
	output.close();

    /*CSVReader reader("./Application/data.csv");
	// Get the data from CSV File
	std::vector<std::vector<std::string> > dataList = reader.getData();
	// Print the content of row by row on screen
	for(std::vector<std::string> vec : dataList)
	{
		for(std::string data : vec)
		{
			std::cout<<data << " , ";
		}
		std::cout<<std::endl;
	}*/
    /********************************************************** Finish Calculating the Proposed Test Using Real Data ********************************************/
    /************************************************************************************************************************************************************/
    #endif //end comment

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
	int T = 50, bandW = 10;
	double rho = 0.1;
	Dgp dgp_obj;
	Matrix X(T, 1), Y1(T, 1), Y2(T, 1), resid_X(T-2, 1), slope_X(3, 1), resid_Y1(T-2, 1), slope_Y1(3, 1), resid_Y2(T-2, 1), slope_Y2(3, 1), resid_Y(T-2, 2);
	dgp_obj.gen_CMixedAR (X, Y1, alpha, beta, lambda, rho, 525235);//draw two independent Gaussian random samples for X and Y := (Y1, Y2)
	dgp_obj.gen_Resid (resid_X, slope_X, X);//estimate the d.g.p of X
	dgp_obj.gen_Resid (resid_Y1, slope_Y1, Y1);//estimate the d.g.p of Y1
	dgp_obj.gen_Resid (resid_Y2, slope_Y2, Y2);//estimate the d.g.p of Y2
	for (auto t = 1; t <= T-2; ++t) {
		resid_Y(t, 1) = resid_Y1(t);
		resid_Y(t, 2) = resid_Y2(t);
	}
	cout << X(10) << " , " << Y1(10) << endl;
	cout << Dep_tests:: do_ECFTest<bartlett_kernel> (bandW, X, Y1) << endl; */
	//Dep_tests dep_obj;
	//cout << dep_obj.do_ElHimdiRoyTest (bandW, resid_X, resid_Y) << endl;//calculate El Himdi & Roy's test statistic
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
	/**************************************************** Start the Simulations for Other Tests: The Univariate Case ********************************************/
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
	/************************************************ Finish the Simulations for Other Tests: The Univariate Case ***********************************************/
    /************************************************************************************************************************************************************/
	#endif //end comment
	/*
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
	double rho = 0.2, threshold = 0.;//set threshold equal to 1.54 for zero correlation
	int T = 200;
	Matrix X(T, 1), Y(T, 2), Y1(T, 1), Y2(T, 1);
	Dgp dgp_obj;
	dgp_obj.gen_AR1 (X, Y1, Y2, alpha, beta, lambda, sigma, rho, threshold, 5353125);
	for (int t = 1; t <= T; ++t) {
		Y(t,1) = Y1(t);
		Y(t,2) = Y2(t);
	}
	Dist_corr dc_obj(X, Y);
	Matrix sigma1(3, 1);
	sigma1(1) = 1.5;
	sigma1(2) = 0.6;
	sigma1(3) = 0.9;
	cout << dc_obj.do_Test <daniell_kernel> (2, 10, 0.50, alpha, beta, lambda, sigma1, rho, 3512523) << endl;
	*/
	#if 0 //begin comment
	/**************************************************** Start the Simulations for the Proposed Test: The Bivariate Case ***************************************/
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
	//double rho = 0.2, threshold = 1.54; //set threshold equal to 1.54 for zero correlation
	double rho = 0.2, corr = 0.8; //set the correlation betwen X and Y1 to 0.8
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    Power obj_power;
    Matrix cv(2, 1), asymp_cv(2, 1), empir_REJF(2, 1), asymp_REJF(2, 1);
    asymp_cv(1) = 1.645;//5%-critical value from the standard normal distribution
    asymp_cv(2) = 1.28;//10%-critical value from the standard normal distribution
    int lag_smooth = 5;

    //******************************************************* Using the Bartlett kernel **************************************************************************
    lag_smooth = 5;
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
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
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
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
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
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
    output_filename = "./Results/BL/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/BL/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/BL/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <bartlett_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, bartlett_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
	                   cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //************************************************* Using the Daniell kernel *********************************************************************************
    lag_smooth = 5;
    output_filename = "./Results/DN/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/DN/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/DN/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <daniell_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, daniell_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
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
    output_filename = "./Results/DN/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/DN/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/DN/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <daniell_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, daniell_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
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
    output_filename = "./Results/DN/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/DN/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/DN/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <daniell_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, daniell_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
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
    output_filename = "./Results/DN/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/DN/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/DN/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <daniell_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, daniell_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
	                   cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //*********************************************************** Using the QS kernel ***************************************************************************
    lag_smooth = 5;
    output_filename = "./Results/QS/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/QS/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/QS/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <QS_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, QS_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
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
    output_filename = "./Results/QS/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/QS/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/QS/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <QS_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, QS_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
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
    output_filename = "./Results/QS/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/QS/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/QS/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <QS_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, QS_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
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
    output_filename = "./Results/QS/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/QS/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/QS/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    cv = obj_power.cValue <QS_kernel> (T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    obj_power.power_f <Dgp, &Dgp::gen_CAR1, QS_kernel> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth, alpha, beta, lambda, sigma, rho, corr,
	                   cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();
	/***************************************** Finish the Simulations for the Proposed Test: The Bivariate Case *************************************************/
    /************************************************************************************************************************************************************/
	#endif //end comment



	#if 0 //begin comment
	/* Calculate empirical levels of a test */
    //Import a column of data
    ifstream data;
    data.open("e:/Dropbox/Codes/Supermicro-Office-1/Exogeneity/Results/HimdiRoy/size_out_T=200_lag_smooth=25_threshold=1.54_bivariate.txt", ios::in);
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
    Dep_tests dep_obj;
    Matrix cv(2, 1);
    int bandW = 25; //set a lag-smoothing parameter
    cv = dep_obj.asymp_CV_ChiSq2 (bandW);
    for (auto i = 1; i <= N; ++i) {
    	if (X(i) >= cv(2)) ++count10;
    	if (X(i) >= cv(1)) ++count5;
	}
	cout << "N = " << N << endl;
	cout << "10% rejection rate = " << ((double) count10/N) << endl;
    cout << "5% rejection rate = " << ((double) count5/N) << endl;
    /*//Import a csv file
    ifstream data;
    data.open("./Results/Others/Correlation/QS/size_out_T=200_lag_smooth=25_rho=0.80000000000000004.txt", ios::in);
    if (!data)
        return(EXIT_FAILURE);
    vector<vector<string>* > data1;
    loadCSV(data, data1);
    int nR = data1.size(); //Rows
    int nC = (*data1[1]).size(); //Columns
    Matrix X(nR,nC);
    for (auto i = 1; i <= nR; i++)
    {
        for (int j = 1; j <= nC; j++)
        {
            X(i,j) = hex2d((*data1[i-1])[j-1]);
        }
    }
    for (vector<vector<string>*>::iterator p = data1.begin( ); p != data1.end( ); ++p)
    {
        delete *p;
    }
    data.close();
    Matrix count10(nC, 1), count5(nC, 1), asymp_cv_Haugh(2, 1);
    int bandW = 25; //set a lag-smoothing parameter
    Dep_tests dep_obj;
    asymp_cv_Haugh = dep_obj.asymp_CV_ChiSq (bandW);
    for (auto i = 1; i <= nR; ++i) {
    	if (X(i,1) >= 1.28) ++count10(1);
    	if (X(i,1) >= 1.645) ++count5(1);
    	if (X(i,2) >= asymp_cv_Haugh(2)) ++count10(2);
    	if (X(i,2) >= asymp_cv_Haugh(1)) ++count5(2);
	}
	cout << "N = " << nR << endl;
	for (auto j = 1; j <= nC; ++j) {
	    cout << "Test = " << j << ": 10% rejection rate = " << ((double) count10(j)/nR) << endl;
        cout << "Test = " << j << ": 5% rejection rate = " << ((double) count5(j)/nR) << endl;
    }*/
    #endif //end comment

    #if 0 //begin comment
    /************************************** Start the Simulations for the Proposed Test: The Univariate Bilinear Case *******************************************/
    /************************************************************************************************************************************************************/
	int TL = 2;
     int T = 100;
     int N = 500;
	int number_sampl = 500;
	double expn = 1.5, delta = 0.1, rho = 0.8; //delta = 0.7
	int choose_alt = 0;
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    int lag_smooth = 5; //set M_T = 5, 10, 20, and 25
    Matrix asymp_CV(2,1), empir_REJF(2,1), empir_CV(2,1), asymp_REJF(2,1), alpha_X(6,1), alpha_Y(6,1), epsilon_t(N,1), epsilon_s(N,1), eta_t(N,1), eta_s(N,1);
    asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
    asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
    alpha_X(1) = 0.01; //define the AR coefficients
    alpha_X(2) = 0.2;
    alpha_X(3) = 0.03;
    alpha_X(4) = -0.6;
    alpha_X(5) = -0.1;
    alpha_X(6) = 0.1;
    alpha_Y(1) = 0.04;
    alpha_Y(2) = 0.2;
    alpha_Y(3) = 0.04;
    alpha_Y(4) = -0.7;
    alpha_Y(5) = 0.1;
    alpha_Y(6) = 0.1;
    NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_t, eta_t, delta, 0., 0, 13435); //generate skewed-normal random errors
    NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_s, eta_s, delta, 0., 0, 53435);

    /************************************************************ Using the Bartlett kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_BL_T=" + std::to_string(T) +  "_N= "  + std::to_string(N) + "_lag_smooth=" + std::to_string(lag_smooth) + "_delta="
                                    + std::to_string(delta) +  "_expn=" + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_BL_T=" + std::to_string(T)  +  "_N= "  + std::to_string(N)  +  "_lag_smooth=" + std::to_string(lag_smooth) + "_delta="
                                + std::to_string(delta)  + "_expn=" + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_BL_T=" + std::to_string(T)  +  "_N= "  + std::to_string(N) + "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                    + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
    NGPower::cValue<bartlett_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                             alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
                                                                                                             delta, expn, asymp_CV, seed, size_out);
    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 1; //nonzero correlation
    NGPower::power_f<bartlett_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 2; //zero correlation
	NGPower::power_f<bartlett_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	choose_alt = 3; //dependence
	NGPower::power_f<bartlett_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();

    /************************************************************ Using the Daniell kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_BL_T=" + std::to_string(T)  +  "_N= "  + std::to_string(N) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_delta="
                                    + std::to_string(delta)  + "_expn=" + std::to_string(expn) + "_daniell_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_BL_T=" + std::to_string(T)  +  "_N= "  + std::to_string(N) + "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                + std::to_string(delta) +  "_expn=" + std::to_string(expn) + "_daniell_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_BL_T=" + std::to_string(T)  +  "_N= "  + std::to_string(N) +  "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                    + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_daniell_kernel_univariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
    NGPower::cValue<daniell_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                             alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
                                                                                                             delta, expn, asymp_CV, seed, size_out);
    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 1; //nonzero correlation
    NGPower::power_f<daniell_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 2; //zero correlation
	NGPower::power_f<daniell_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	choose_alt = 3; //dependence
	NGPower::power_f<daniell_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();

    /************************************************************ Using the QS kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_BL_T=" + std::to_string(T)  +  "_N= "  + std::to_string(N) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_delta="
                                    + std::to_string(delta)  + "_expn=" + std::to_string(expn) + "_QS_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_BL_T=" + std::to_string(T)  +  "_N= "  + std::to_string(N) +  "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_QS_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_BL_T=" + std::to_string(T)  +  "_N= "  + std::to_string(N) +  "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                    + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_QS_kernel_univariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
    NGPower::cValue<QS_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                             alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
                                                                                                             delta, expn, asymp_CV, seed, size_out);
    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 1; //nonzero correlation
    NGPower::power_f<QS_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 2; //zero correlation
	NGPower::power_f<QS_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	choose_alt = 3; //dependence
	NGPower::power_f<QS_kernel, NL_Dgp::gen_Bilinear<NL_Dgp::gen_SN>, NGReg::cmean_BL, NL_Dgp::est_BL> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    /***************************************** Finish the Simulations for the Proposed Test: The Univariate Bilinear Case ***************************************/
    /************************************************************************************************************************************************************/
	#endif //end comment

    //#if 0 //begin comment
    /************************************** Start the Simulations for the Proposed Test: The Univariate TAR Case ************************************************/
    /************************************************************************************************************************************************************/
	int TL = 4;
    int T = 100;
    int N = 500;
	int number_sampl = 500;
	double expn = 1.5, delta = 0.1, rho = 0.8;
	int choose_alt = 0;
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    int lag_smooth = 5; //set M_T = 5, 10, 20, and 25
    Matrix asymp_CV(2,1), empir_REJF(2,1), empir_CV(2,1), asymp_REJF(2,1), alpha_X(5,1), alpha_Y(5,1), epsilon_t(N,1), epsilon_s(N,1), eta_t(N,1), eta_s(N,1);
    asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
    asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
    alpha_X(1) = 0.01; //define the AR coefficients
    alpha_X(2) = 0.7;
    alpha_X(3) = 0.1;
    alpha_X(4) = 0.1;
    alpha_X(5) = -0.3;
    alpha_Y(1) = 0.04;
    alpha_Y(2) = 0.4;
    alpha_Y(3) = 0.5;
    alpha_Y(4) = 0.1;
    alpha_Y(5) = -0.6;
    NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_t, eta_t, delta, 0., 0, 13435); //generate skewed-normal random errors
    NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_s, eta_s, delta, 0., 0, 53435);

    /************************************************************ Using the Bartlett kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_TAR_T=" + std::to_string(T)  +  "_N="  + std::to_string(N) +  "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                    + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_TAR_T=" + std::to_string(T)  +  "_N="  + std::to_string(N) + "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                    + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_TAR_T=" + std::to_string(T)  +  "_N="  + std::to_string(N) + "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                    + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
    NGPower::cValue<bartlett_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                             alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
                                                                                                             delta, expn, asymp_CV, seed, size_out);
    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 1; //nonzero correlation
    NGPower::power_f<bartlett_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 2; //zero correlation
	NGPower::power_f<bartlett_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	choose_alt = 3; //dependence
	NGPower::power_f<bartlett_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();

    /************************************************************ Using the Daniell kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_TAR_T=" + std::to_string(T)  +  "_N="  + std::to_string(N) + "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                    + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_daniell_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_TAR_T=" + std::to_string(T)  +  "_N="  + std::to_string(N) +  "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                    + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_daniell_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_TAR_T=" + std::to_string(T)  +  "_N="  + std::to_string(N) +  "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                    + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_daniell_kernel_univariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
    NGPower::cValue<daniell_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                             alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
                                                                                                             delta, expn, asymp_CV, seed, size_out);
    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 1; //nonzero correlation
    NGPower::power_f<daniell_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 2; //zero correlation
	NGPower::power_f<daniell_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	choose_alt = 3; //dependence
	NGPower::power_f<daniell_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();

    /************************************************************ Using the QS kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_TAR_T=" + std::to_string(T)  +  "_N="  + std::to_string(N) + "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                   + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_QS_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_TAR_T=" + std::to_string(T)  +  "_N="  + std::to_string(N) + "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                   + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_QS_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_TAR_T=" + std::to_string(T)  +  "_N="  + std::to_string(N) + "_lag_smooth=" + std::to_string(lag_smooth)  + "_delta="
                                   + std::to_string(delta) + "_expn=" + std::to_string(expn) + "_QS_kernel_univariate.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    //calculate 5%- and 10%- asymptotic sizes and empirical critical values
    NGPower::cValue<QS_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (asymp_REJF, empir_CV, number_sampl, T, TL, lag_smooth,
                                                                                                             alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s,
                                                                                                             delta, expn, asymp_CV, seed, size_out);
    output << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical values for lag_smooth = " << lag_smooth << ":  " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << endl;
    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 1; //nonzero correlation
    NGPower::power_f<QS_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    choose_alt = 2; //zero correlation
	NGPower::power_f<QS_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	choose_alt = 3; //dependence
	NGPower::power_f<QS_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_SN>, NGReg::cmean_TAR, NL_Dgp::est_TAR> (empir_REJF, asymp_REJF, number_sampl, T, TL, lag_smooth,
                                                                                                              alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, delta,
                                                                                                              rho, expn, empir_CV, asymp_CV, choose_alt, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
	output.close();
    size_out.close();
    pwr_out.close();
    /***************************************** Finish the Simulations for the Proposed Test: The Univariate TAR Case ********************************************/
    /************************************************************************************************************************************************************/
	//#endif //end comment

	#if 0 //begin comment
	/**************************************************** Start the Simulations for El Himdi & Roy's test: The Bivariate Case ***********************************/
    /************************************************************************************************************************************************************/
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
	sigma(1) = 1.5; // cf. Dgp::gen_CAR1 in dgp.h
	sigma(2) = 1.5; // cf. Dgp::gen_CAR1 in dgp.h
	//double rho = 0.2, threshold = 1.54; //set threshold equal to 1.54 for zero correlation
	double rho = 0.2, corr = 0.8; //set the correlation betwen X and Y1 to 0.8
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    Dep_tests dep_obj;
    Matrix cv(2, 1), asymp_cv(2, 1), empir_REJF(2, 1), asymp_REJF(2, 1);
    int lag_smooth = 5;
    output_filename = "./Results/HimdiRoy/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/HimdiRoy/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/HimdiRoy/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    dep_obj.cValue (cv, T, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    asymp_cv = dep_obj.asymp_CV_ChiSq2 (lag_smooth);//obtain 5%- and 10%- asymptotic critical values for the test
	dep_obj.power_f <Dgp, &Dgp::gen_CAR1> (empir_REJF, asymp_REJF, number_sampl, T, lag_smooth, alpha, beta, lambda, sigma, rho, corr, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

	//============================================================================================================================================================
	lag_smooth = 10;
    output_filename = "./Results/HimdiRoy/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/HimdiRoy/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/HimdiRoy/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    dep_obj.cValue (cv, T, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    asymp_cv = dep_obj.asymp_CV_ChiSq2 (lag_smooth);//obtain 5%- and 10%- asymptotic critical values for the test
	dep_obj.power_f <Dgp, &Dgp::gen_CAR1> (empir_REJF, asymp_REJF, number_sampl, T, lag_smooth, alpha, beta, lambda, sigma, rho, corr, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //============================================================================================================================================================
    lag_smooth = 20;
    output_filename = "./Results/HimdiRoy/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/HimdiRoy/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/HimdiRoy/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    dep_obj.cValue (cv, T, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    asymp_cv = dep_obj.asymp_CV_ChiSq2 (lag_smooth);//obtain 5%- and 10%- asymptotic critical values for the test
	dep_obj.power_f <Dgp, &Dgp::gen_CAR1> (empir_REJF, asymp_REJF, number_sampl, T, lag_smooth, alpha, beta, lambda, sigma, rho, corr, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    //============================================================================================================================================================
    lag_smooth = 25;
    output_filename = "./Results/HimdiRoy/file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
	size_filename = "./Results/HimdiRoy/size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    power_filename = "./Results/HimdiRoy/power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_corr="
	                  + boost::lexical_cast<std::string>(corr) + "_bivariate_corr.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;
    dep_obj.cValue (cv, T, lag_smooth, alpha, beta, lambda, sigma, rho, seed, size_out);
    output << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    cout << "5%- and 10%- empirical critical value for lag_smooth = " << lag_smooth << ":  " << "(" << cv(1) << " , " << cv(2) << ")" << endl;
    asymp_cv = dep_obj.asymp_CV_ChiSq2 (lag_smooth);//obtain 5%- and 10%- asymptotic critical values for the test
	dep_obj.power_f <Dgp, &Dgp::gen_CAR1> (empir_REJF, asymp_REJF, number_sampl, T, lag_smooth, alpha, beta, lambda, sigma, rho, corr, cv, asymp_cv, seed, pwr_out);
	output << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << empir_REJF(1) << " , " << empir_REJF(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for lag_smooth = " << lag_smooth << ": " << "(" << asymp_REJF(1) << " , " << asymp_REJF(2) << ")" << endl;
    output.close();
    size_out.close();
    pwr_out.close();

    /***************************************** Finish the Simulations for El Himdi & Roy's test: The Bivariate Case *********************************************/
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
    cout << "This program took " << std::chrono::duration_cast <std::chrono::milliseconds> (time-timelast).count() << " milliseconds to run.\n";
    //output.close ();
    //pwr_out.close ();
    //gsl_rng_free (r);
    #if defined(_WIN64) || defined(_WIN32)
    system("PAUSE");
    #endif
    return 0;
}






