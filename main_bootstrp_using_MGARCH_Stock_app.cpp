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


#include <plot.h>
#include "utils.h"

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


#include <ShogunML/ml_reg_6_1_4.h>
#include <ShogunML/data/data.h>
#include <ShogunML/tscv.h>
#include <ML_dcorr.h>



#define CHUNK 1


using namespace std;
using namespace shogun;
using namespace shogun::linalg;

//void (*aktfgv)(double *,double *,int *,int *,void *,Matrix&);

int main(void) {

	//start the timer%%%%%
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    auto time = std::chrono::high_resolution_clock::now();
    auto timelast = time;

	/*========================================== Perform the Distance-based Test using Random Forest to Fit the Data ====================================*/

	/****************************************************************** Import data into Shogun matrices ***********************************************/

	std::shared_ptr<CSVFile> stocks_file( new CSVFile("./Application/data/Bond/stocks_2000_2022_monthly_returns1.csv", 'r') );
    std::shared_ptr<CSVFile> bonds_file( new CSVFile("./Application/data/Bond/bonds_2000_2022_monthly_returns1.csv", 'r') );

    SGMatrix<double> X, Y;
    double scaling = 1e+2; //scale up observations to facilitate RF

    X.load(stocks_file);
    X = scale(transpose_matrix(X), scaling);
	X.display_matrix("stock returns");
	cout << "(num_rows, num_cols) = " << X.num_rows << " , " << X.num_cols << endl;

	Y.load(bonds_file);
    Y = scale(transpose_matrix(Y), scaling);
	Y.display_matrix("bond returns");
	cout << "(num_rows, num_cols) = " << Y.num_rows << " , " << Y.num_cols << endl;

	ASSERT_(X.num_rows == Y.num_rows); // the number of rows in the two matrices must be equal

	/************************************************************** Perform the bootstrap test **********************************************************/
	int num_B = 1000; //number of bootstrap samples to be generated
	int L = 4; //set a maximum truncation lag
    std::vector<int> lag_smooth = {5, 10, 15, 20, 25}; //set a kernel bandwidth: 5, 10, 15, 20, 25, 30, 35, 40, 45
    double expn = 1.5; //set an exponent for the distance correlation
    unsigned long int seed = 134323;
    int T = X.num_rows;

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

	/* Compute the integrals of quadratic and quartic functions of the kernel weight */
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	ML_DCORR::integrate_Kernel <daniell_kernel> (kernel_QDSum, kernel_QRSum);
	cout << "(kernel_QDSum, kernel_QRSum) = (" << kernel_QDSum << ", " << kernel_QRSum << ")" << endl;

	/* List of hyperparameters used to train Random Forest */
	int num_subsets = 2, min_subset_size = 10, tree_max_depths_list_size = 2, num_iters_list_size = 1, learning_rates_list_size = 10, \
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

	SGVector<double> tstat_bootstrap(num_B);
	double pvalue = 0., tstat = 0.;
	ofstream  output; //open an output stream


	for (auto ii : lag_smooth) {
		string output_filename = "./Application/output/Bond/results_T=" + std::to_string(T) + "_lag_smooth=" + std::to_string(ii) + "_L="
											+ std::to_string(L) +  "_RF.txt";
		output.open (output_filename.c_str(), ios::out);

		std::tie(pvalue, tstat, tstat_bootstrap) = ML_DCORR::do_Test_bt<ML_REG::RF_cv1, ML_REG::RF_Plot, daniell_kernel> \
																		(X, Y, /*Shogun matrices of observations*/
																		L, ii, expn,
																		xi_x, xi_y, /*num_B by T matrices of auxiliary random variables used for bootstrapping*/
																		num_subsets, /*number of subsets for TSCV*/
																		min_subset_size, /*minimum subset size for TSCV*/
																		tree_max_depths_list, /*list of tree max depths (for GBM)*/
																		num_iters_list, /*list of numbers of iterations (for GBM)*/
																		learning_rates_list, /*list of learning rates (for GBM)*/
																		subset_fractions_list, /*list of subset fractions (for GBM)*/
																		num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
																		num_bags_list, /*list of number of bags (for RF)*/
																		kernel_QDSum, kernel_QRSum, /*quadratically and quartically integrate the kernel function*/
																		seed /*seed for random number generator*/);
		output << "p-value = " << pvalue << endl;
		output << "the value of the distance-based statistic = " << tstat << endl;
		output << "The bootstrap statistics: " << endl;
		for (int i = 0; i < num_B; i++)
			output << tstat_bootstrap[i] << endl;
		output.close();
	}


    //Please do not comment out the lines below.
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    time = std::chrono::high_resolution_clock::now();
    //output << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //cout << "Elapsed time = " << (time-timelast)/60 << " minutes" << endl;
    //output << "This program took " << std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count() << " seconds to run.\n";
    auto duration =  std::chrono::duration_cast <std::chrono::seconds> (time-timelast).count();
    cout << "This program took " << duration << " seconds (" << duration/60.0 << " minutes) to run!" << endl;
    //output.close ();
    //pwr_out.close ();
    //gsl_rng_free (r);
    system("PAUSE");
    return 0;
}







