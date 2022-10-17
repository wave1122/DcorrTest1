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
#include <sys/stat.h>
#include <sys/types.h>

#include <matrix_ops2.h>
#include <dist_corr.h>
#include <kernel.h>
//#include <power.h>
#include <tests.h>
#include <nl_dgp.h>
//#include <nongaussian_reg.h>
#include <dep_tests.h>
//#include <nongaussian_dist_corr.h>
//#include <VAR_gLasso.h>
//#include <student_reg.h>

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


//#include <ShogunML/ml_reg_6_1_4.h>
#include <ShogunML/data/data.h>
//#include <ShogunML/tscv.h>
//#include <ML_reg_dcorr.h>
#include <ML_dcorr.h>
#include <mgarch.h>
#include <nmsimplex.h>
#include <hsic_test.h>
#include <ML_dcorr_power.h>
#include <others_power.h>



#define CHUNK 1



using namespace std;
using namespace shogun;
using namespace shogun::linalg;//void (*aktfgv)(double *,double *,int *,int *,void *,Matrix&);

int main(void) {

	//start the timer%%%%%
    //time = ((double) clock())/((double) CLOCKS_PER_SEC);
    auto time = std::chrono::high_resolution_clock::now();
    auto timelast = time;


     //# if 0 //begin comment
	/************************************************************************************************************************************************************/
	/************ Start the Simulations for Wang et al.'s (2021) HSIC-based test, El Himdi and Roy's (1997), Bouhaddioui and Roy's (2006),
												Robbins and Fisher's (2015), and Tchahou and Duchesne's (2013) tests ********************************************/
	/******************************************************* True DGP: CC-MGARCH and Fitted GDP: VAR(1) *******************************************************/
	int number_sampl = 500; /*number of random samples to be drawn*/
    int num_bts = 500; /*number of bootstrap samples as in Wang et al. (2021)*/
    int T = 100;
    int lag_smooth = 3; /*3, 5, 7, 10, 20, 25*/
    int choose_alt = 0;
    double h = 0.; /*a differential factor to compute numerical derivatives*/
    unsigned long int seed = 12425;


    /* define VAR(1) parameters */
    SGVector<double> theta_var1(4), theta_var2(4);
    theta_var1[0] = 0.4;
    theta_var1[1] = 0.1;
    theta_var1[2] = -1.;
    theta_var1[3] = 0.5;

    theta_var2[0] = -1.5;
    theta_var2[1] = 1.2;
    theta_var2[2] = -0.9;
    theta_var2[3] = 0.5;

    /* define CC-MGARCH parameters */
    int dim = 7;
	SGVector<double> theta_mgarch1(dim), theta_mgarch2(dim);
	double theta_mgarch1a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	for (int i = 0; i < dim; ++i)
		theta_mgarch1[i] = theta_mgarch1a[i];
	theta_mgarch1.display_vector("theta_mgarch1");

	double theta_mgarch2a[dim] = {0.2, 0.8, 0.1, 0.2, 0.8, 0.1, 0.5};
	for (int i = 0; i < dim; ++i)
		theta_mgarch2[i] = theta_mgarch2a[i];
	theta_mgarch2.display_vector("theta_mgarch2");

	Matrix HSIC_J1_REJF(2,1), HSIC_J2_REJF(2,1), ER_asymp_REJF(2,1), ER_empir_CV(2,1), BR_asymp_REJF(2,1), BR_empir_CV(2,1), RbF_asymp_REJF(2,1), \
			RbF_empir_CV(2,1), TD_L1_asymp_REJF(2,1), TD_L1_empir_CV(2,1), TD_T1_asymp_REJF(2,1), TD_T1_empir_CV(2,1);

	Matrix HSIC_J1_empir_REJF(2,1), HSIC_J2_empir_REJF(2,1), ER_empir_REJF(2,1), BR_empir_REJF(2,1), RbF_empir_REJF(2,1), TD_L1_empir_REJF(2,1), \
																									TD_T1_empir_REJF(2,1);

	Matrix ER_asymp_CV(2,1), BR_asymp_CV(2,1), RbF_asymp_CV(2,1), TD_L1_asymp_CV(2,1), TD_T1_asymp_CV(2,1);

	BR_asymp_CV(1) = 1.645; //5%-critical value from the standard normal distribution
	BR_asymp_CV(2) = 1.28; //10%-critical value from the standard normal distribution
	ER_asymp_CV = Dep_tests::asymp_CV_ChiSq3(lag_smooth);
	RbF_asymp_CV = Dep_tests::asymp_CV_Gamma(lag_smooth, 2, 2);
	TD_L1_asymp_CV = Dep_tests::asymp_CV_ChiSq(lag_smooth);
	TD_T1_asymp_CV = Dep_tests::asymp_CV_ChiSq4(lag_smooth);

	const char *new_dir;
	string dir_name, output_filename, size_filename, power_filename;
	ofstream output, size_out, pwr_out;

	bool use_bartlett_ker = false;


    //*************************************************************using the Bartlett kernel****************************************************************************/
	// Creating a directory
    dir_name = "./Results/Others/CC-MGARCH_misspec/BL/";
    new_dir = dir_name.c_str();
    if (mkdir(new_dir, 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;

    else
        cout << "Directory created" << endl;

	output_filename = dir_name + "file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_bartlett_kernel_cc_mgarch_misspec.txt";
	size_filename = dir_name + "size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_bartlett_kernel_cc_mgarch_misspec.txt";
	power_filename = dir_name + "power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_bartlett_kernel_cc_mgarch_misspec.txt";
	output.open (output_filename.c_str(), ios::out);
	size_out.open (size_filename.c_str(), ios::out);
	pwr_out.open (power_filename.c_str(), ios::out);

	output << std::fixed << std::setprecision(5);
	output << "T = " << T << " and Lag smooth length = " << lag_smooth << endl;

	//calculate empirical critical values and sizes
     Others_power::cValue<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,bartlett_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_asymp_REJF, /*asymptotic rejection rates of El Himdi and Roy's (1997) test*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							BR_asymp_REJF, /*asymptotic rejection rates of Bouhaddioui and Roy's (2006) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_asymp_REJF, /*asymptotic rejection rates of Robbins and Fisher's (2015) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							TD_L1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of and Robbins and Fisher's (2015) test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							h,
							seed,
							size_out);
	output << "The HSIC-based J1 test: Empirical 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << HSIC_J1_REJF(1) \
			<< " , " << HSIC_J1_REJF(2) << ")" << endl;
	output << "The HSIC-based J2 test: Empirical 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << HSIC_J2_REJF(1) \
			<< " , " << HSIC_J2_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << ER_asymp_CV(1) \
																								<< " , " << ER_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << ER_empir_CV(1) \
																								<< " , " << ER_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << ER_asymp_REJF(1) \
																								<< " , " << ER_asymp_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << BR_asymp_CV(1) \
																								<< " , " << BR_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << BR_empir_CV(1) \
																								<< " , " << BR_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << BR_asymp_REJF(1) \
																								<< " , " << BR_asymp_REJF(2) << ")" << endl;
	output << "==================================================================================================================" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << RbF_asymp_CV(1) \
																								<< " , " << RbF_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << RbF_empir_CV(1) \
																								<< " , " << RbF_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << RbF_asymp_REJF(1) \
																								<< " , " << RbF_asymp_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_L1_asymp_CV(1) \
																								<< " , " << TD_L1_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_L1_empir_CV(1) \
																								<< " , " << TD_L1_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << TD_L1_asymp_REJF(1) \
																								<< " , " << TD_L1_asymp_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_T1_asymp_CV(1) \
																								<< " , " << TD_T1_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_T1_empir_CV(1) \
																								<< " , " << TD_T1_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << TD_T1_asymp_REJF(1) \
																								<< " , " << TD_T1_asymp_REJF(2) << ")" << endl;

	//calculate rejection rates
	choose_alt = 1; //correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,bartlett_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 2; //non-correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,bartlett_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 3; //non-correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,bartlett_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 4; //non-correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,bartlett_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 5; //strongly correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,bartlett_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	output.close(); //close file streams
	size_out.close();
	pwr_out.close();

    //***********************************************************using the Daniell kernel*************************************************************************
 	// Creating a directory
    dir_name = "./Results/Others/CC-MGARCH_misspec/DN/";
    new_dir = dir_name.c_str();
    if (mkdir(new_dir, 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;

    else
        cout << "Directory created" << endl;

 	output_filename = dir_name + "file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_daniell_kernel_cc_mgarch_misspec.txt";
	size_filename = dir_name + "size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_daniell_kernel_cc_mgarch_misspec.txt";
	power_filename = dir_name + "power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_daniell_kernel_cc_mgarch_misspec.txt";
	output.open (output_filename.c_str(), ios::out);
	size_out.open (size_filename.c_str(), ios::out);
	pwr_out.open (power_filename.c_str(), ios::out);

	output << std::fixed << std::setprecision(5);
	output << "T = " << T << " and Lag smooth length = " << lag_smooth << endl;

	//calculate empirical critical values and sizes
     Others_power::cValue<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,daniell_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_asymp_REJF, /*asymptotic rejection rates of El Himdi and Roy's (1997) test*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							BR_asymp_REJF, /*asymptotic rejection rates of Bouhaddioui and Roy's (2006) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_asymp_REJF, /*asymptotic rejection rates of Robbins and Fisher's (2015) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							TD_L1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of and Robbins and Fisher's (2015) test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							h,
							seed,
							size_out,
							use_bartlett_ker);

	output << "The HSIC-based J1 test: Empirical 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << HSIC_J1_REJF(1) \
			<< " , " << HSIC_J1_REJF(2) << ")" << endl;
	output << "The HSIC-based J2 test: Empirical 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << HSIC_J2_REJF(1) \
			<< " , " << HSIC_J2_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << ER_asymp_CV(1) \
																								<< " , " << ER_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << ER_empir_CV(1) \
																								<< " , " << ER_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << ER_asymp_REJF(1) \
																								<< " , " << ER_asymp_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << BR_asymp_CV(1) \
																								<< " , " << BR_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << BR_empir_CV(1) \
																								<< " , " << BR_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << BR_asymp_REJF(1) \
																								<< " , " << BR_asymp_REJF(2) << ")" << endl;
	output << "==================================================================================================================" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << RbF_asymp_CV(1) \
																								<< " , " << RbF_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << RbF_empir_CV(1) \
																								<< " , " << RbF_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << RbF_asymp_REJF(1) \
																								<< " , " << RbF_asymp_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_L1_asymp_CV(1) \
																								<< " , " << TD_L1_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_L1_empir_CV(1) \
																								<< " , " << TD_L1_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << TD_L1_asymp_REJF(1) \
																								<< " , " << TD_L1_asymp_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_T1_asymp_CV(1) \
																								<< " , " << TD_T1_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_T1_empir_CV(1) \
																								<< " , " << TD_T1_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << TD_T1_asymp_REJF(1) \
																								<< " , " << TD_T1_asymp_REJF(2) << ")" << endl;

	//calculate rejection rates
	choose_alt = 1; //correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,daniell_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 2; //non-correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,daniell_kernel> \
						( 	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 3; //non-correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,daniell_kernel> \
						( 	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 4; //non-correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,daniell_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 5; //strongly correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,daniell_kernel> \
						( 	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;


	output.close(); //close file streams
	size_out.close();
	pwr_out.close();

    //******************************************************************using the QS kernel***********************************************************************
	// Creating a directory
    dir_name = "./Results/Others/CC-MGARCH_misspec/QS/";
    new_dir = dir_name.c_str();
    if (mkdir(new_dir, 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;

    else
        cout << "Directory created" << endl;

	output_filename = dir_name + "file_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_QS_kernel_cc_mgarch_misspec.txt";
	size_filename = dir_name + "size_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_QS_kernel_cc_mgarch_misspec.txt";
	power_filename = dir_name + "power_out_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(lag_smooth) + "_QS_kernel_cc_mgarch_misspec.txt";
	output.open (output_filename.c_str(), ios::out);
	size_out.open (size_filename.c_str(), ios::out);
	pwr_out.open (power_filename.c_str(), ios::out);

	output << std::fixed << std::setprecision(5);
	output << "T = " << T << " and Lag smooth length = " << lag_smooth << endl;

	//calculate empirical critical values and sizes
     Others_power::cValue<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,QS_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_asymp_REJF, /*asymptotic rejection rates of El Himdi and Roy's (1997) test*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							BR_asymp_REJF, /*asymptotic rejection rates of Bouhaddioui and Roy's (2006) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_asymp_REJF, /*asymptotic rejection rates of Robbins and Fisher's (2015) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							TD_L1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of and Robbins and Fisher's (2015) test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							h,
							seed,
							size_out,
							use_bartlett_ker);

	output << "The HSIC-based J1 test: Empirical 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << HSIC_J1_REJF(1) \
			<< " , " << HSIC_J1_REJF(2) << ")" << endl;
	output << "The HSIC-based J2 test: Empirical 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << HSIC_J2_REJF(1) \
			<< " , " << HSIC_J2_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << ER_asymp_CV(1) \
																								<< " , " << ER_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << ER_empir_CV(1) \
																								<< " , " << ER_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << ER_asymp_REJF(1) \
																								<< " , " << ER_asymp_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << BR_asymp_CV(1) \
																								<< " , " << BR_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << BR_empir_CV(1) \
																								<< " , " << BR_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << BR_asymp_REJF(1) \
																								<< " , " << BR_asymp_REJF(2) << ")" << endl;
	output << "==================================================================================================================" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << RbF_asymp_CV(1) \
																								<< " , " << RbF_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << RbF_empir_CV(1) \
																								<< " , " << RbF_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << RbF_asymp_REJF(1) \
																								<< " , " << RbF_asymp_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_L1_asymp_CV(1) \
																								<< " , " << TD_L1_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_L1_empir_CV(1) \
																								<< " , " << TD_L1_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << TD_L1_asymp_REJF(1) \
																								<< " , " << TD_L1_asymp_REJF(2) << ")" << endl;

	output << "==================================================================================================================" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_T1_asymp_CV(1) \
																								<< " , " << TD_T1_asymp_CV(2) << ")" << endl;
	output << "Empirical 5%- and 10%- critical values for lag_smooth = " << lag_smooth << " are "  << "(" << TD_T1_empir_CV(1) \
																								<< " , " << TD_T1_empir_CV(2) << ")" << endl;
	output << "Asymptotic 5%- and 10%- sizes for lag_smooth = " << lag_smooth << " are "  << "(" << TD_T1_asymp_REJF(1) \
																								<< " , " << TD_T1_asymp_REJF(2) << ")" << endl;

	//calculate rejection rates
	choose_alt = 1; //correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,QS_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 2; //non-correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,QS_kernel> \
						( 	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 3; //non-correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,QS_kernel> \
						( 	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 4; //non-correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,QS_kernel> \
						(	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	choose_alt = 5; //strongly correlated errors
    Others_power::power_f<NL_Dgp::gen_CC_MGARCH,HSIC::kernel,HSIC::kernel,NL_Dgp::estimate_VAR1,NL_Dgp::gen_VAR,NL_Dgp::resid_VAR1,\
																NL_Dgp::sse_VAR1_gradient,NL_Dgp::sse_VAR1_hessian,QS_kernel> \
						( 	HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							ER_empir_REJF, ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							BR_empir_REJF, BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_REJF, RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							TD_L1_empir_REJF, TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_REJF, TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							theta_var1, theta_var2, /*true parameters of two DGPs*/
							theta_var1, theta_var2, /*initial values used to estimate DGPs*/
							number_sampl, /*number of random samples drawn*/
							T, /*sample size*/
							lag_smooth, /*kernel bandwidth*/
							num_bts, /*number of bootstrap samples*/
							ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							choose_alt, /*set a degree of dependence between two DGPs*/
							h, /*a finite differential factor*/
							seed, /*seed for random number generator*/
							pwr_out,
							use_bartlett_ker);
	output << "========================================================================================================================================" << endl;
	output << "choose_alt = " << choose_alt << endl;
	output << "Wang et al.'s (2021) HSIC-based J1 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J1_REJF(1) << " , " << HSIC_J1_REJF(2) \
																																				<< ")" << endl;
	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Wang et al.'s (2021) HSIC-based J2 test:" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << HSIC_J2_REJF(1) << " , " << HSIC_J2_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "El Himdi and Roy's (1997) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_asymp_REJF(1) << " , " << ER_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << ER_empir_REJF(1) << " , " << ER_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Bouhaddioui and Roy's (2006) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_asymp_REJF(1) << " , " << BR_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << BR_empir_REJF(1) << " , " << BR_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Robbins and Fisher's (2015) test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_asymp_REJF(1) << " , " << RbF_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << RbF_empir_REJF(1) << " , " << RbF_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) L1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_asymp_REJF(1) << " , " << TD_L1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_L1_empir_REJF(1) << " , " << TD_L1_empir_REJF(2) \
																																				<< ")" << endl;

	output << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output << "Tchahou and Duchesne's (2013) T1 test:" << endl;
	output << "Asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_asymp_REJF(1) << " , " << TD_T1_asymp_REJF(2) \
																																				<< ")" << endl;
	output << "Empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << TD_T1_empir_REJF(1) << " , " << TD_T1_empir_REJF(2) \
																																				<< ")" << endl;
	output << "=======================================================================================================================================" << endl;

	output.close(); //close file streams
	size_out.close();
	pwr_out.close();

	/************ Finish the Simulations for Wang et al.'s (2021) HSIC-based test, El Himdi and Roy's (1997), Bouhaddioui and Roy's (2006),
												Robbins and Fisher's (2015), and Tchahou and Duchesne's (2013) tests ********************************************/
    /**************************************************************************************************************************************************************/
	//#endif //end comment

	# if 0 //begin comment
     /***********************************************************************************************************************************************/
     /******************************************** Start the Simulations for Other Tests: The Univariate TAR Case ********************************************/
     auto T = 100,  number_sampl = 500;
     double rho = 0.8;
	Matrix X(T,1), Y(T,1), alpha_X(5,1), alpha_Y(5,1);
     alpha_X(1) = 0.01;
     alpha_X(2) = 0.7;
     alpha_X(3) = 0.1;
     alpha_X(4) = 0.1;
     alpha_X(5) = -0.3;
     alpha_Y(1) = 0.04;
     alpha_Y(2) = 0.4;
     alpha_Y(3) = 0.5;
     alpha_Y(4) = 0.1;
     alpha_Y(5) = -0.6;

	unsigned long seed = 856764325;
     string output_filename, size_filename, power_filename;
     ofstream output, size_out, pwr_out;
     Matrix REJF_Hong(2,1), REJF_ECFTest(2,1), REJF_Haugh(2,1), empir_CV_Hong(2,1), empir_CV_ECFTest(2,1), empir_CV_Haugh(2,1), asymp_CV_Hong(2,1), asymp_CV_Haugh(2,1);
     Matrix empir_REJF_Hong(2,1), asymp_REJF_Hong(2,1), empir_REJF_ECFTest(2,1), asymp_REJF_ECFTest(2,1), empir_REJF_Haugh(2,1), asymp_REJF_Haugh(2,1);
     asymp_CV_Hong(1) = 1.645; //5%-critical value from the standard normal distribution
     asymp_CV_Hong(2) = 1.28; //10%-critical value from the standard normal distribution
     int bandW = 5 /*10, 20, 25*/, choose_alt = 1;
     asymp_CV_Haugh = Dep_tests::asymp_CV_ChiSq (bandW);

    //*************************************************************using the Bartlett kernel****************************************************************************/
     output_filename = "./Results/NonGaussian/MN/Others/file_out_TAR_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                     + "_bartlett_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_TAR_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                     + "_bartlett_kernel_univariate.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_TAR_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                     + "_bartlett_kernel_univariate.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values and sizes
     Dep_tests::cValue <bartlett_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (REJF_Hong, REJF_ECFTest, REJF_Haugh, empir_CV_Hong, empir_CV_ECFTest,
                                                                                                                empir_CV_Haugh, alpha_X, alpha_Y, asymp_CV_Hong, asymp_CV_Haugh, number_sampl, T, bandW, 0., seed, size_out);
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << empir_CV_Hong(1) << " , "
                 << empir_CV_Hong(2) << ")" << endl;
     output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << asymp_CV_Hong(1) << " , "
                 << asymp_CV_Hong(2) << ")" << endl;
     output << "Hong's (1996) spectral test: 5%- and 10%- sizes for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << REJF_Hong(1) << " , " << REJF_Hong(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << empir_CV_ECFTest(1) << " , "
                 << empir_CV_ECFTest(2) << ")" << endl;
     output << "Hong's (2001) ecf-based test: 5%- and 10%- sizes for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << REJF_ECFTest(1) << " , " << REJF_ECFTest(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << empir_CV_Haugh(1) << " , "
                 << empir_CV_Haugh(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << asymp_CV_Haugh(1) << " , "
                 << asymp_CV_Haugh(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: 5%- and 10%- sizes for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << REJF_Haugh(1) << " , " << REJF_Haugh(2)
                 << ")" << endl;
	//calculate rejection rates
	choose_alt = 1; //correlated errors
     Dep_tests::power_f<bartlett_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_ECFTest, asymp_REJF_ECFTest,
                                                                                                                                                                                empir_REJF_Haugh, asymp_REJF_Haugh, alpha_X, alpha_Y, empir_CV_Hong,
                                                                                                                                                                                asymp_CV_Hong, empir_CV_ECFTest, asymp_CV_Hong, empir_CV_Haugh,
                                                                                                                                                                                asymp_CV_Haugh, number_sampl, T, bandW, 0., rho, choose_alt, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     output << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
     cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     cout << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	cout << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;

     choose_alt = 2; //uncorrelated but dependent errors
     Dep_tests::power_f<bartlett_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_ECFTest, asymp_REJF_ECFTest,
                                                                                                                                                                                empir_REJF_Haugh, asymp_REJF_Haugh, alpha_X, alpha_Y, empir_CV_Hong,
                                                                                                                                                                                asymp_CV_Hong, empir_CV_ECFTest, asymp_CV_Hong, empir_CV_Haugh,
                                                                                                                                                                                asymp_CV_Haugh, number_sampl, T, bandW, 0., rho, choose_alt, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     output << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
     cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     cout << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	cout << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3; //strongly dependent errors
     Dep_tests::power_f<bartlett_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_ECFTest, asymp_REJF_ECFTest,
                                                                                                                                                                                empir_REJF_Haugh, asymp_REJF_Haugh, alpha_X, alpha_Y, empir_CV_Hong,
                                                                                                                                                                                asymp_CV_Hong, empir_CV_ECFTest, asymp_CV_Hong, empir_CV_Haugh,
                                                                                                                                                                                asymp_CV_Haugh, number_sampl, T, bandW, 0., rho, choose_alt, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     output << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
     cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     cout << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	cout << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output.close();
     size_out.close();
     pwr_out.close();

    //***********************************************************using the Daniell kernel*************************************************************************
     output_filename = "./Results/NonGaussian/MN/Others/file_out_TAR_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                     + "_daniell_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_TAR_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                     + "_daniell_kernel_univariate.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_TAR_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                     + "_daniell_kernel_univariate.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values and sizes
     Dep_tests::cValue <daniell_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (REJF_Hong, REJF_ECFTest, REJF_Haugh, empir_CV_Hong, empir_CV_ECFTest,
                                                                                                                empir_CV_Haugh, alpha_X, alpha_Y, asymp_CV_Hong, asymp_CV_Haugh, number_sampl, T, bandW, 0., seed, size_out);
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << empir_CV_Hong(1) << " , "
                 << empir_CV_Hong(2) << ")" << endl;
     output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << asymp_CV_Hong(1) << " , "
                 << asymp_CV_Hong(2) << ")" << endl;
     output << "Hong's (1996) spectral test: 5%- and 10%- sizes for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << REJF_Hong(1) << " , " << REJF_Hong(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << empir_CV_ECFTest(1) << " , "
                 << empir_CV_ECFTest(2) << ")" << endl;
     output << "Hong's (2001) ecf-based test: 5%- and 10%- sizes for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << REJF_ECFTest(1) << " , " << REJF_ECFTest(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << empir_CV_Haugh(1) << " , "
                 << empir_CV_Haugh(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << asymp_CV_Haugh(1) << " , "
                 << asymp_CV_Haugh(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: 5%- and 10%- sizes for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << REJF_Haugh(1) << " , " << REJF_Haugh(2)
                 << ")" << endl;
	//calculate rejection rates
	choose_alt = 1;
     Dep_tests::power_f<daniell_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_ECFTest, asymp_REJF_ECFTest,
                                                                                                                                                                                empir_REJF_Haugh, asymp_REJF_Haugh, alpha_X, alpha_Y, empir_CV_Hong,
                                                                                                                                                                                asymp_CV_Hong, empir_CV_ECFTest, asymp_CV_Hong, empir_CV_Haugh,
                                                                                                                                                                                asymp_CV_Haugh, number_sampl, T, bandW, 0., rho, choose_alt, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     output << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
     cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     cout << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	cout << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2;
     Dep_tests::power_f<daniell_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_ECFTest, asymp_REJF_ECFTest,
                                                                                                                                                                                empir_REJF_Haugh, asymp_REJF_Haugh, alpha_X, alpha_Y, empir_CV_Hong,
                                                                                                                                                                                asymp_CV_Hong, empir_CV_ECFTest, asymp_CV_Hong, empir_CV_Haugh,
                                                                                                                                                                                asymp_CV_Haugh, number_sampl, T, bandW, 0., rho, choose_alt, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     output << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
     cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     cout << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	cout << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3;
     Dep_tests::power_f<daniell_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_ECFTest, asymp_REJF_ECFTest,
                                                                                                                                                                                empir_REJF_Haugh, asymp_REJF_Haugh, alpha_X, alpha_Y, empir_CV_Hong,
                                                                                                                                                                                asymp_CV_Hong, empir_CV_ECFTest, asymp_CV_Hong, empir_CV_Haugh,
                                                                                                                                                                                asymp_CV_Haugh, number_sampl, T, bandW, 0., rho, choose_alt, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     output << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
     cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     cout << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	cout << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output.close();
     size_out.close();
     pwr_out.close();

    //******************************************************************using the QS kernel***********************************************************************
     output_filename = "./Results/NonGaussian/MN/Others/file_out_TAR_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                     + "_QS_kernel_univariate.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_TAR_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                     + "_QS_kernel_univariate.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_TAR_T=" + std::to_string(T) +  "_lag_smooth=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                     + "_QS_kernel_univariate.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     output << "Lag smooth length = " << bandW << endl;
	//calculate empirical critical values and sizes
     Dep_tests::cValue <QS_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (REJF_Hong, REJF_ECFTest, REJF_Haugh, empir_CV_Hong, empir_CV_ECFTest,
                                                                                                                empir_CV_Haugh, alpha_X, alpha_Y, asymp_CV_Hong, asymp_CV_Haugh, number_sampl, T, bandW, 0., seed, size_out);
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << empir_CV_Hong(1) << " , "
                 << empir_CV_Hong(2) << ")" << endl;
     output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << asymp_CV_Hong(1) << " , "
                 << asymp_CV_Hong(2) << ")" << endl;
     output << "Hong's (1996) spectral test: 5%- and 10%- sizes for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << REJF_Hong(1) << " , " << REJF_Hong(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << empir_CV_ECFTest(1) << " , "
                 << empir_CV_ECFTest(2) << ")" << endl;
     output << "Hong's (2001) ecf-based test: 5%- and 10%- sizes for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << REJF_ECFTest(1) << " , " << REJF_ECFTest(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are "  << "(" << empir_CV_Haugh(1) << " , "
                 << empir_CV_Haugh(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- critical value for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << asymp_CV_Haugh(1) << " , "
                 << asymp_CV_Haugh(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: 5%- and 10%- sizes for lag_smooth = " << bandW << " and rho= " << rho <<  " are " << "(" << REJF_Haugh(1) << " , " << REJF_Haugh(2)
                 << ")" << endl;
	//calculate rejection rates
	choose_alt = 1;
     Dep_tests::power_f<QS_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_ECFTest, asymp_REJF_ECFTest,
                                                                                                                                                                                empir_REJF_Haugh, asymp_REJF_Haugh, alpha_X, alpha_Y, empir_CV_Hong,
                                                                                                                                                                                asymp_CV_Hong, empir_CV_ECFTest, asymp_CV_Hong, empir_CV_Haugh,
                                                                                                                                                                                asymp_CV_Haugh, number_sampl, T, bandW, 0., rho, choose_alt, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     output << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
     cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     cout << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	cout << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2;
     Dep_tests::power_f<QS_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_ECFTest, asymp_REJF_ECFTest,
                                                                                                                                                                                empir_REJF_Haugh, asymp_REJF_Haugh, alpha_X, alpha_Y, empir_CV_Hong,
                                                                                                                                                                                asymp_CV_Hong, empir_CV_ECFTest, asymp_CV_Hong, empir_CV_Haugh,
                                                                                                                                                                                asymp_CV_Haugh, number_sampl, T, bandW, 0., rho, choose_alt, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     output << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
     cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     cout << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	cout << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3;
     Dep_tests::power_f<QS_kernel, NL_Dgp::gen_TAR<NL_Dgp::gen_MN>, NL_Dgp::est_TAR> (empir_REJF_Hong, asymp_REJF_Hong, empir_REJF_ECFTest, asymp_REJF_ECFTest,
                                                                                                                                                                                empir_REJF_Haugh, asymp_REJF_Haugh, alpha_X, alpha_Y, empir_CV_Hong,
                                                                                                                                                                                asymp_CV_Hong, empir_CV_ECFTest, asymp_CV_Hong, empir_CV_Haugh,
                                                                                                                                                                                asymp_CV_Haugh, number_sampl, T, bandW, 0., rho, choose_alt, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
     output << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	output << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     output << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	output << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     output << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	output << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     output << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
     cout << "Hong's (1996) spectral test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Hong(1) << " , " << empir_REJF_Hong(2)
                                                                                                                                                                         << ")" << endl;
	cout << "Hong's (1996) spectral test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Hong(1) << " , " << asymp_REJF_Hong(2)
                                                                                                                                                                           << ")" << endl;
     cout << "Hong's (2001) ecf-based test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_ECFTest(1) << " , "
                                                                                                                                                                           << empir_REJF_ECFTest(2) << ")" << endl;
	cout << "Hong's (2001) ecf-based test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_ECFTest(1) << " , "
	                                                                                                                                                                         << asymp_REJF_ECFTest(2) << ")" << endl;
     cout << "Haugh's (1976) Portmanteau test: empirical 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << empir_REJF_Haugh(1) << " , "
                                                                                                                                                                                   << empir_REJF_Haugh(2) << ")" << endl;
	cout << "Haugh's (1976) Portmanteau test: asymptotic 5%- and 10%- rejection rates for choose_alt = " << choose_alt << ": " << "(" << asymp_REJF_Haugh(1) << " , "
                                                                                                                                                                                     << asymp_REJF_Haugh(2) << ")" << endl;
     cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
	output.close();
     size_out.close();
     pwr_out.close();
	/*************************************** Finish the Simulations for Other Tests: The Univariate TAR Case ***********************************************/
    /**********************************************************************************************************************************************/
	#endif //end comment

	#if 0 //begin comment
	/********************************************** Start the Simulations for El Himdi & Roy's test: The Bivariate Bilinear Case ***********************************/
    /************************************************************************************************************************************************/
     auto T = 100, number_sampl = 500, bandW = 1, choose_alt = 0;
     Matrix X(T,1), Y1(T,1), Y2(T,1), alpha_X(6,1), alpha_Y(6,2), delta(3,1);
     Matrix REJF_ElHimdiRoy(2,1), empir_CV_ElHimdiRoy(2,1), empir_REJF_ElHimdiRoy(2,1), asymp_REJF_ElHimdiRoy(2,1);
     alpha_X(1) = 0.01;
     alpha_X(2) = 0.2;
     alpha_X(3) = 0.03;
     alpha_X(4) = -0.6;
     alpha_X(5) = -0.1;
     alpha_X(6) = 0.1;
     alpha_Y(1,1) = 0.04;
     alpha_Y(2,1) = 0.2;
     alpha_Y(3,1) = 0.04;
     alpha_Y(4,1) = -0.7;
     alpha_Y(5,1) = 0.1;
     alpha_Y(6,1) = 0.1;
     alpha_Y(1,2) = 0.05;
     alpha_Y(2,2) = 0.8;
     alpha_Y(3,2) = -0.05;
     alpha_Y(4,2) = -0.5;
     alpha_Y(5,2) = -0.1;
     alpha_Y(6,2) = 0.1;
     double rho = 0.5;
     unsigned long seed = 856764325;
     string output_filename, size_filename, power_filename;
     ofstream output, size_out, pwr_out;
     bandW = 5;
     output << "Lag smooth length = " << bandW << endl;
     output_filename = "./Results/NonGaussian/MN/Others/file_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     //calculate empirical critical values and sizes
	Matrix asymp_CV_ElHimdiRoy = Dep_tests::asymp_CV_ChiSq2 (bandW); //obtain 5%- and 10%- asymptotic critical values for the test
     Dep_tests::cValue <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (REJF_ElHimdiRoy, empir_CV_ElHimdiRoy, alpha_X, alpha_Y, delta, asymp_CV_ElHimdiRoy,
                                                                                                                                                          number_sampl, T, bandW, seed, size_out);
     output << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     output << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     //calculate rejection rates
     choose_alt = 1; //errors are correlated
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2; //errors are uncorrelated but still dependent
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3; //errors are (strongly) dependent
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output.close();
     size_out.close();
     pwr_out.close();

	//================================================================================================================================
     bandW = 10;
     output << "Lag smooth length = " << bandW << endl;
     output_filename = "./Results/NonGaussian/MN/Others/file_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     //calculate empirical critical values and sizes
	asymp_CV_ElHimdiRoy = Dep_tests::asymp_CV_ChiSq2 (bandW);//obtain 5%- and 10%- asymptotic critical values for the test
     Dep_tests::cValue <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (REJF_ElHimdiRoy, empir_CV_ElHimdiRoy, alpha_X, alpha_Y, delta, asymp_CV_ElHimdiRoy,
                                                                                                                                                          number_sampl, T, bandW, seed, size_out);
     output << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     output << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     //calculate rejection rates
     choose_alt = 1; //errors are correlated
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2; //errors are uncorrelated but still dependent
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3; //errors are (strongly) dependent
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;

    output.close();
     size_out.close();
     pwr_out.close();

    //============================================================================================================================================================
     bandW = 20;
     output << "Lag smooth length = " << bandW << endl;
     output_filename = "./Results/NonGaussian/MN/Others/file_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     //calculate empirical critical values and sizes
	asymp_CV_ElHimdiRoy = Dep_tests::asymp_CV_ChiSq2 (bandW);//obtain 5%- and 10%- asymptotic critical values for the test
     Dep_tests::cValue <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (REJF_ElHimdiRoy, empir_CV_ElHimdiRoy, alpha_X, alpha_Y, delta, asymp_CV_ElHimdiRoy,
                                                                                                                                                          number_sampl, T, bandW, seed, size_out);
     output << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     output << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     //calculate rejection rates
     choose_alt = 1; //errors are correlated
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2; //errors are uncorrelated but still dependent
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3; //errors are (strongly) dependent
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output.close();
     size_out.close();
     pwr_out.close();

    //============================================================================================================================================================
     bandW = 25;
     output << "Lag smooth length = " << bandW << endl;
     output_filename = "./Results/NonGaussian/MN/Others/file_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_BL_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho)
                                    + "_HRTest.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     //calculate empirical critical values and sizes
	asymp_CV_ElHimdiRoy = Dep_tests::asymp_CV_ChiSq2 (bandW);//obtain 5%- and 10%- asymptotic critical values for the test
     Dep_tests::cValue <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (REJF_ElHimdiRoy, empir_CV_ElHimdiRoy, alpha_X, alpha_Y, delta, asymp_CV_ElHimdiRoy,
                                                                                                                                                          number_sampl, T, bandW, seed, size_out);
     output << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     output << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     //calculate rejection rates
     choose_alt = 1; //errors are correlated
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2; //errors are uncorrelated but still dependent
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3; //errors are (strongly) dependent
     Dep_tests::power_f <NL_Dgp::gen_Bilinear<NL_Dgp::gen_TriMN>, NL_Dgp::est_BL> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output.close();
     size_out.close();
     pwr_out.close();

    /***************************************** Finish the Simulations for El Himdi & Roy's test: The Bivariate Bilinear Case ****************************************/
    /************************************************************************************************************************************************/
	#endif //end comment

	#if 0 //begin comment
	/********************************************** Start the Simulations for El Himdi & Roy's test: The Bivariate TAR Case ***********************************/
    /************************************************************************************************************************************************/
     auto T = 100, number_sampl = 500, bandW = 1, choose_alt = 0;
     Matrix X(T,1), Y1(T,1), Y2(T,1), alpha_X(5,1), alpha_Y(5,2), delta(3,1);
     Matrix REJF_ElHimdiRoy(2,1), empir_CV_ElHimdiRoy(2,1), empir_REJF_ElHimdiRoy(2,1), asymp_REJF_ElHimdiRoy(2,1);
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

     double rho = 0.5;
     unsigned long seed = 856764325;
     string output_filename, size_filename, power_filename;
     ofstream output, size_out, pwr_out;
     bandW = 5;
     output << "Lag smooth length = " << bandW << endl;
     output_filename = "./Results/NonGaussian/MN/Others/file_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     //calculate empirical critical values and sizes
	Matrix asymp_CV_ElHimdiRoy = Dep_tests::asymp_CV_ChiSq2 (bandW); //obtain 5%- and 10%- asymptotic critical values for the test
     Dep_tests::cValue <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (REJF_ElHimdiRoy, empir_CV_ElHimdiRoy, alpha_X, alpha_Y, delta, asymp_CV_ElHimdiRoy,
                                                                                                                                                          number_sampl, T, bandW, seed, size_out);
     output << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     output << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     //calculate rejection rates
     choose_alt = 1; //errors are correlated
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2; //errors are uncorrelated but still dependent
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3; //errors are (strongly) dependent
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output.close();
     size_out.close();
     pwr_out.close();

	//================================================================================================================================
     bandW = 10;
     output << "Lag smooth length = " << bandW << endl;
     output_filename = "./Results/NonGaussian/MN/Others/file_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     //calculate empirical critical values and sizes
	asymp_CV_ElHimdiRoy = Dep_tests::asymp_CV_ChiSq2 (bandW);//obtain 5%- and 10%- asymptotic critical values for the test
     Dep_tests::cValue <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (REJF_ElHimdiRoy, empir_CV_ElHimdiRoy, alpha_X, alpha_Y, delta, asymp_CV_ElHimdiRoy,
                                                                                                                                                          number_sampl, T, bandW, seed, size_out);
     output << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     output << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     //calculate rejection rates
     choose_alt = 1; //errors are correlated
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2; //errors are uncorrelated but still dependent
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3; //errors are (strongly) dependent
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;

    output.close();
     size_out.close();
     pwr_out.close();

    //============================================================================================================================================================
     bandW = 20;
     output << "Lag smooth length = " << bandW << endl;
     output_filename = "./Results/NonGaussian/MN/Others/file_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     //calculate empirical critical values and sizes
	asymp_CV_ElHimdiRoy = Dep_tests::asymp_CV_ChiSq2 (bandW);//obtain 5%- and 10%- asymptotic critical values for the test
     Dep_tests::cValue <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (REJF_ElHimdiRoy, empir_CV_ElHimdiRoy, alpha_X, alpha_Y, delta, asymp_CV_ElHimdiRoy,
                                                                                                                                                          number_sampl, T, bandW, seed, size_out);
     output << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     output << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     //calculate rejection rates
     choose_alt = 1; //errors are correlated
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2; //errors are uncorrelated but still dependent
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3; //errors are (strongly) dependent
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output.close();
     size_out.close();
     pwr_out.close();

    //============================================================================================================================================================
     bandW = 25;
     output << "Lag smooth length = " << bandW << endl;
     output_filename = "./Results/NonGaussian/MN/Others/file_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
	size_filename = "./Results/NonGaussian/MN/Others/size_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
     power_filename = "./Results/NonGaussian/MN/Others/power_out_TAR_T=" + std::to_string(T) +  "_bandW=" + std::to_string(bandW) + "_rho=" + std::to_string(rho) + "_"
                                    + "HRTest.txt";
     output.open (output_filename.c_str(), ios::out);
     size_out.open (size_filename.c_str(), ios::out);
     pwr_out.open (power_filename.c_str(), ios::out);
     //calculate empirical critical values and sizes
	asymp_CV_ElHimdiRoy = Dep_tests::asymp_CV_ChiSq2 (bandW);//obtain 5%- and 10%- asymptotic critical values for the test
     Dep_tests::cValue <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (REJF_ElHimdiRoy, empir_CV_ElHimdiRoy, alpha_X, alpha_Y, delta, asymp_CV_ElHimdiRoy,
                                                                                                                                                          number_sampl, T, bandW, seed, size_out);
     output << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     output << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- empirical critical values for bandW = " <<  bandW << ":  " << "(" << empir_CV_ElHimdiRoy(1) << " , " << empir_CV_ElHimdiRoy(2) << ")" << endl;
     cout << "5%- and 10%- sizes for bandW = " <<  bandW << ":  " << "(" << REJF_ElHimdiRoy(1) << " , " << REJF_ElHimdiRoy(2) << ")" << endl;
     //calculate rejection rates
     choose_alt = 1; //errors are correlated
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 2; //errors are uncorrelated but still dependent
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     choose_alt = 3; //errors are (strongly) dependent
     Dep_tests::power_f <NL_Dgp::gen_TAR<NL_Dgp::gen_TriMN>, NL_Dgp::est_TAR> (empir_REJF_ElHimdiRoy, asymp_REJF_ElHimdiRoy, alpha_X, alpha_Y, delta,
                                                                                                                                                           empir_CV_ElHimdiRoy, asymp_CV_ElHimdiRoy, rho, 0., choose_alt, number_sampl,
                                                                                                                                                           T, bandW, seed, pwr_out);
     output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output << "choose_alt = " << choose_alt << endl;
	output << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	output << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     cout << "choose_alt = " << choose_alt << endl;
	cout << "5%- and 10% - empirical rejection rates for bandW = " << bandW << ": " << "(" << empir_REJF_ElHimdiRoy(1) << " , " << empir_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "5%- and 10% - asymptotic rejection rates for bandW = " << bandW << ": " << "(" << asymp_REJF_ElHimdiRoy(1) << " , " << asymp_REJF_ElHimdiRoy(2) << ")" << endl;
	cout << "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
     output.close();
     size_out.close();
     pwr_out.close();

     /***************************************** Finish the Simulations for El Himdi & Roy's test: The Bivariate TAR Case ****************************************/
     /************************************************************************************************************************************************/
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
    auto duration =  std::chrono::duration_cast <std::chrono::milliseconds> (time-timelast).count();
    cout << "This program took " << duration << " seconds (" << duration << " milliseconds) to run." << endl;
    //output.close ();
    //pwr_out.close ();
    //gsl_rng_free (r);
    #if defined(_WIN64) || defined(_WIN32)
    system("PAUSE");
    #endif
    return 0;
}






