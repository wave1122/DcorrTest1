/* Implement  El Himdi and Roy's (1997) test statistic using real data. */

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
#include <boost/numeric/conversion/cast.hpp>
#include <boost/lexical_cast.hpp>
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
#include <tests.h>
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
	int bandw = 5; //set a minimum kernel bandwidth for El Himdi & Roy's (1997) statistic
	int bandw_max = 25; //set a maximum kernel bandwidth for El Himdi & Roy's (1997) statistic
	int lag_len = 20; //set an AR lag length

    /***********************************************************************************************************************************************************/
    /***************************************** Start Calculating El Himdi and Roy's (1997) test statistic (Q^{M}_{HR}) Using Real Data *****************************************/
	//Import a csv file of oil prices
	ifstream oildata;
	oildata.open ("./Application/oilprices_2017.csv", ios::in);
	if (!oildata)
		return(EXIT_FAILURE);
	vector<string> oildata1;
	loadCSV(oildata, oildata1);
	int T = oildata1.size() - 1; //number of rows to calculate returns
	int nC_x = 1;
	Matrix X(T,nC_x);
	for (auto t = 1; t <= T; t++)
		//X(t) = 100*(hex2d(oildata1[t]) - hex2d(oildata1[t-1])) / hex2d(oildata1[t-1]); //calculate simple oil returns
		X(t) = 100*(log(hex2d(oildata1[t])) - log(hex2d(oildata1[t-1]))); //calculate log oil returns
	oildata.close(); //close the input stream

	//Import a csv file of stock prices: use 'stockprices_2017.csv' for the U.S. stock market indices and 'data_MSCI_GCC_2017.csv' for the two major GCC countries stock indices
	ifstream stockdata;
	stockdata.open ("./Application/data_MSCI_GCC_2017.csv", ios::in);
	if (!stockdata)
		return(EXIT_FAILURE);
	vector<vector<string>* > stockdata1; //an array pointer
	loadCSV(stockdata, stockdata1);
	int nC_y = (*stockdata1[1]).size(); //number of columns
	int T_y = stockdata1.size() - 1; //number of rows
	ASSERT (T == T_y); //to confirm that both X and Y have the same number of rows
	Matrix Y(T,nC_y);
	for (auto t = 1; t <= T; ++t)
		for (auto j = 1; j <= nC_y; ++j)
			//Y(t,j) = 100*(hex2d((*stockdata1[t])[j-1]) - hex2d((*stockdata1[t-1])[j-1])) / hex2d((*stockdata1[t-1])[j-1]); //calculate simple stock returns
			Y(t,j) = 100*(log(hex2d((*stockdata1[t])[j-1])) - log(hex2d((*stockdata1[t-1])[j-1]))); //calculate log stock returns
	for (vector<vector<string>*>::iterator p = stockdata1.begin( ); p != stockdata1.end( ); ++p)
		delete *p; //free memory
	stockdata.close(); //close the input stream

	Matrix resid_X(T-lag_len,1), slope_X(lag_len+1,1), Yj(T,1), resid_Yj(T-lag_len,1), slope_Yj(lag_len+1,1), resid_Y(T-lag_len,nC_y);

	Dgp dgp_obj; //define an object for the class `Dgp'
	double SSR = 0.;
	dgp_obj.est_AR (resid_X, slope_X, SSR, X, lag_len); //fit X to an AR(lag_len) process, and obtain a sequence of residuals
	for (auto j = 1; j <= nC_y; ++j) {
		for (auto t = 1; t <= T; ++t) {
			Yj(t) = Y(t, j);
		}
		dgp_obj.est_AR (resid_Yj, slope_Yj, SSR, Yj, lag_len); //fit Yj to an AR(lag_len) process and obtain a sequence of residuals
		for (auto t = 1; t <= T-lag_len; ++t) {
			resid_Y(t, j) = resid_Yj(t);
		}
	}

	ofstream output;
	string output_filename = "./Application/file_out_HR_T=" + std::to_string(T) + "_dx=" + std::to_string(nC_x) + "_dy=" + std::to_string(nC_y) + "_lag_len="
										+ std::to_string(lag_len) +  ".txt";
	output.open (output_filename.c_str(), ios::out);
	output << "El Himdi & Roy's (1997) (HR) test: " << endl;
	cout << "El Himdi & Roy's (1997) (HR) test: " << endl;

	Dep_tests dep_obj; //define an object for the class `Dep_tests'
	do {
		output << "bandwidth = " << bandw << ": HR = " << dep_obj.do_ElHimdiRoyTest (bandw, resid_X, resid_Y) << endl;
		bandw += 5;
	} while (bandw <= bandw_max); //loop through a set of bandwidths at step 5 till the value set by bandw_max

	output.close(); //close the output stream
    /**************************************** Finish Calculating El Himdi and Roy's (1997) test statistic (Q^{M}_{HR}) Using Real Data ******************************************/
    /************************************************************************************************************************************************************/

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






