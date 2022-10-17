#include <iostream>
#include <fstream>
#include <iomanip>   // format manipulation
#include <string>
#include <sstream>
#include <cstdlib>
#include <math.h>
#include <cmath>
#include <numeric>
#include <stdio.h>
#include <map>
#include <unistd.h>
#include <gsl/gsl_math.h>
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
#include <nl_dgp.h>
#include <dep_tests.h>
#include <plot.h>
#include <utils.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/config.h>
//#include <shogun/base/init.h>
//#include <shogun/base/some.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/multiclass/tree/CARTree.h>
#include <shogun/util/iterators.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/linop/MatrixOperator.h>

#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/File.h>


//#include <ShogunML/ml_reg_6_1_4.h>
#include <ShogunML/data/data.h>
//#include <ShogunML/tscv.h>
//#include <ML_reg_dcorr.h>
#include <ML_dcorr.h>
#include <mgarch.h>
#include <nmsimplex.h>
#include <hsic_test.h>
#include <ML_dcorr_power.h>



#define CHUNK 1



using namespace std;
using namespace shogun;
using namespace shogun::linalg;


int main(void) {

	//start the timer%%%%%
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    auto time = std::chrono::high_resolution_clock::now();
    auto timelast = time;

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
	double rho = 0.2;
	Dgp dgp_obj;
	Matrix X(T, 1), Y1(T, 1), Y2(T, 1), resid_X(T-2, 1), slope_X(3, 1), resid_Y1(T-2, 1), slope_Y1(3, 1), resid_Y2(T-2, 1), slope_Y2(3, 1), resid_Y(T-2, 2);
	dgp_obj.gen_AR1 (X, Y1, Y2, alpha, beta, lambda, sigma, rho, 0., 525235);//draw two independent Gaussian random samples for X and Y := (Y1, Y2)
	dgp_obj.gen_Resid (resid_X, slope_X, X);//estimate the d.g.p of X
	dgp_obj.gen_Resid (resid_Y1, slope_Y1, Y1);//estimate the d.g.p of Y1
	dgp_obj.gen_Resid (resid_Y2, slope_Y2, Y2);//estimate the d.g.p of Y2
	for (auto t = 1; t <= T-2; ++t) {
		resid_Y(t, 1) = resid_Y1(t);
		resid_Y(t, 2) = resid_Y2(t);
	}
	Matrix cov(1, 2);
	cov = cross_Cov (resid_X, resid_Y, 5);
	cout << cov(1, 1) << " , " << cov(1, 2) << endl;*/
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
	//#if 0 //begin comment
	/******************************************* Start the Simulations for the Proposed Test ********************************************************************/
    /************************************************************************************************************************************************************/
    int seed = 1535;
    int T = 100;
	int L = 2;
	int lag_smooth = 10; //set M_T = 3, 5, 7, 10, 20, and 25
	double expn = 1.5;
	int number_sampl = 100;
	int num_B = 500;

	/* Generate two independent sequences of i.i.d. standard normal random variables */
	gsl_rng *r = nullptr;
	const gsl_rng_type *gen; //random number generator
	gsl_rng_env_setup();
	gen = gsl_rng_taus;
	r = gsl_rng_alloc(gen);
	gsl_rng_set(r, seed);

	SGMatrix<double> xi_x(num_B, T), xi_y(num_B, T);
	for (auto i = 0; i < num_B; i++) {
		for (auto t = 0; t < T; t++) {
			xi_x(i,t) = gsl_ran_ugaussian (r);
			xi_y(i,t) = gsl_ran_ugaussian (r);
		}
	}

	gsl_rng_free (r); //free memory

	/* List of hyperparameters used to train Random Forest */
	int num_subsets = 1, min_subset_size = 10, tree_max_depths_list_size = 2, num_iters_list_size = 1, learning_rates_list_size = 10, \
		subset_fractions_list_size = 4, num_rand_feats_list_size = L, num_bags_list_size = 2;
	SGVector<int> tree_max_depths_list(tree_max_depths_list_size), num_iters_list(num_iters_list_size), \
					num_rand_feats_list(num_rand_feats_list_size), num_bags_list(num_bags_list_size);
	SGVector<double> learning_rates_list(learning_rates_list_size), subset_fractions_list(subset_fractions_list_size);

	for (int i = 0; i < tree_max_depths_list_size; ++i)
		tree_max_depths_list[i] = 4*(i+1);
	tree_max_depths_list.display_vector("tree_max_depths_list");

	for (int i = 0; i < num_iters_list_size; ++i)
		num_iters_list[i] = i*100 + 200;
	num_iters_list.display_vector("num_iters_list");

	for (int i = 0; i < learning_rates_list_size; ++i)
		learning_rates_list[i] =  0.01*(i+1);
	learning_rates_list.display_vector("learning_rates_list");

	for (int i = 0; i < subset_fractions_list_size; ++i)
		subset_fractions_list[i] = 0.1*(i+1);
	subset_fractions_list.display_vector("subset_fractions_list");

	for (int i = 0; i < num_rand_feats_list_size; ++i)
		num_rand_feats_list[i] = 2*(i+1);
	num_rand_feats_list.display_vector("number of features list");

	for (int i = 0; i < num_bags_list_size; ++i)
		num_bags_list[i] = 20*i + 30;
	num_bags_list.display_vector("number of bags list");

	/* define VAR(1) parameters */
    SGVector<double> theta1(4), theta2(4);
    theta1[0] = 0.4;
    theta1[1] = 0.1;
    theta1[2] = -1.;
    theta1[3] = 0.5;

    theta2[0] = -1.5;
    theta2[1] = 1.2;
    theta2[2] = -0.9;
    theta2[3] = 0.5;

	const char *new_dir;
	string dir_name, output_filename, size_filename, power_filename;
	ofstream output, size_out, pwr_out;

	Matrix bootstrap_REJF(2, 1);

	output << std::fixed << std::setprecision(5);

    /************************************************************ Using the Bartlett kernel ***********************************************************************/
    // Creating a directory
    dir_name = "./Results/ML/VAR/Bootstrap/BL/";
    new_dir = dir_name.c_str();
    if (mkdir(new_dir, 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;

    else
        cout << "Directory created" << endl;

    output_filename = dir_name + "file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_bartlett_kernel_bootstrap_var.txt";
	size_filename = dir_name + "size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_bartlett_kernel_bootstrap_var.txt";
    power_filename = dir_name + "power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_bartlett_kernel_bootstrap_var.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;

    //calculate 5%- and 10%- empirical sizes
    int choose_alt = 0;
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, bartlett_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

    choose_alt = 1;  //correlation (between eta1 and eta2) = 0.3
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, bartlett_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;


	choose_alt = 2;  //the errors (eta1 and eta2) are not correlated, but dependent
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, bartlett_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	choose_alt = 3;  //the errors (eta1 and eta2) are not correlated, but dependent
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, bartlett_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	choose_alt = 4;  //the errors (eta1 and eta2) are not correlated, but dependent
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, bartlett_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	choose_alt = 5;  //the errors (eta1 and eta2) are highly correlated
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, bartlett_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	output.close();
    size_out.close();
    pwr_out.close();

    /************************************************************ Using the Daniell kernel ***********************************************************************/
	// Creating a directory
    dir_name = "./Results/ML/VAR/Bootstrap/DN/";
    new_dir = dir_name.c_str();
    if (mkdir(new_dir, 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;

    else
        cout << "Directory created" << endl;

	output_filename = dir_name + "file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_daniell_kernel_bootstrap_var.txt";
	size_filename = dir_name + "size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_daniell_kernel_bootstrap_var.txt";
    power_filename = dir_name + "power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_daniell_kernel_bootstrap_var.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;

     //calculate 5%- and 10%- empirical sizes
    choose_alt = 0;
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, daniell_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

    choose_alt = 1;  //correlation (between eta1 and eta2) = 0.3
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, daniell_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;


	choose_alt = 2;  //the errors (eta1 and eta2) are not correlated, but dependent
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, daniell_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	choose_alt = 3;  //the errors (eta1 and eta2) are not correlated, but dependent
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, daniell_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	choose_alt = 4;  //the errors (eta1 and eta2) are not correlated, but dependent
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, daniell_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	choose_alt = 5;  //the errors (eta1 and eta2) are highly correlated
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, daniell_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	output.close();
    size_out.close();
    pwr_out.close();

    /************************************************************ Using the QS kernel ***********************************************************************/
    // Creating a directory
    dir_name = "./Results/ML/VAR/Bootstrap/QS/";
    new_dir = dir_name.c_str();
    if (mkdir(new_dir, 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;

    else
        cout << "Directory created" << endl;

    output_filename = dir_name + "file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_QS_kernel_bootstrap_var.txt";
	size_filename = dir_name + "size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_QS_kernel_bootstrap_var.txt";
    power_filename = dir_name + "power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn=" + std::to_string(expn)
                                   + "_QS_kernel_bootstrap_var.txt";
    output.open (output_filename.c_str(), ios::out);
    size_out.open (size_filename.c_str(), ios::out);
    pwr_out.open (power_filename.c_str(), ios::out);
    output << "Lag smooth length = " << lag_smooth << endl;

     //calculate 5%- and 10%- empirical sizes
    choose_alt = 0;
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, QS_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

    output << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5% - and 10% - sizes for the lag_smooth = " << lag_smooth << ": (" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

    choose_alt = 1;  //correlation (between eta1 and eta2) = 0.3
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, QS_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;


	choose_alt = 2;  //the errors (eta1 and eta2) are not correlated, but dependent
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, QS_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	choose_alt = 3;  //the errors (eta1 and eta2) are not correlated, but dependent
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, QS_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	choose_alt = 4;  //the errors (eta1 and eta2) are not correlated, but dependent
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, QS_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	choose_alt = 5;  //the errors (eta1 and eta2) are highly correlated
	ML_dcorr_power::power_f<NL_Dgp::gen_VAR, ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, QS_kernel>> \
							( 	bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
								theta1, theta2, /*true parameters of two DGPs*/
								number_sampl, /*number of random samples drawn*/
								L, /*maximum truncation lag*/
								lag_smooth, /*kernel bandwidth*/
								expn, /*exponent of the Euclidean distance*/
								xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								seed,
								pwr_out);

	output << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;
    cout << "5%- and 10% - empirical rejection rates for choose_alt = " <<  choose_alt << ": " << "(" << bootstrap_REJF(1) << " , " << bootstrap_REJF(2) << ")" << endl;

	output.close();
    size_out.close();
    pwr_out.close();

	/***************************************** Finish the Simulations for the Proposed Test *********************************************************************/
    /************************************************************************************************************************************************************/
	//#endif //end comment


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
    /************************************** Start the Simulations for the Proposed Test: The Univariate Skewed-Normal Case **************************************/
    /************************************************************************************************************************************************************/
	int TL = 2;
    int T = 100;
    int N = 1000;
	int number_sampl = 500;
	double expn = 1.5, delta = 0.7, rho = 0.8;
	int choose_alt = 0;
    unsigned long seed = 856764325;
    string output_filename, size_filename, power_filename;
    ofstream output, size_out, pwr_out;
    int lag_smooth = 5; //set M_T = 5, 10, 20, and 25
    //============================================================= Bilinear Models ===============================================================================/
    Matrix asymp_CV(2,1), empir_REJF(2,1), empir_CV(2,1), asymp_REJF(2,1), alpha_X(6,1), alpha_Y(6,1), epsilon_t(N,1), epsilon_s(N,1), eta_t(N,1), eta_s(N,1);
    asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
    asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
    alpha_X(1) = 0.01; //define the AR coefficients
    alpha_X(2) = 0.5;
    alpha_X(3) = -0.2;
    alpha_X(4) = -0.1;
    alpha_X(5) = 0.2;
    alpha_X(6) = 0.1;
    alpha_Y(1) = 0.04;
    alpha_Y(2) = 0.7;
    alpha_Y(3) = -0.1;
    alpha_Y(4) = -0.3;
    alpha_Y(5) = 0.1;
    alpha_Y(6) = 0.1;
    NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_t, eta_t, delta, 0., 0, 13435); //generate skewed-normal random errors
    NL_Dgp::gen_RANV<NL_Dgp::gen_SN> (epsilon_s, eta_s, delta, 0., 0, 53435);

    /************************************************************ Using the Bartlett kernel ***********************************************************************/
    output_filename = "./Results/NonGaussian/SN/file_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_bartlett_kernel_univariate.txt";
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
    output_filename = "./Results/NonGaussian/SN/file_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_daniell_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_daniell_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_daniell_kernel_univariate.txt";
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
    output_filename = "./Results/NonGaussian/SN/file_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_QS_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/SN/size_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_QS_kernel_univariate.txt";
    power_filename = "./Results/NonGaussian/SN/power_out_BL_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_expn="
	                  + std::to_string(expn) + "_QS_kernel_univariate.txt";
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
    /***************************************** Finish the Simulations for the Proposed Test: The Univariate Case ************************************************/
    /************************************************************************************************************************************************************/
	#endif //end comment

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






