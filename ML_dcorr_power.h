#ifndef ML_DCORR_POWER_H
#define ML_DCORR_POWER_H

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <nongaussian_dist_corr.h>
#include <kernel.h>
#include <nl_dgp.h>
#include <ML_dcorr.h>

using namespace std;

class ML_dcorr_power {
	public:
		ML_dcorr_power (){  }; //default constructor
		~ML_dcorr_power () { }; //default destructor

	/* Calculate 5%- and 10%- asymptotic sizes and empirical critical values. */
	template <	/*simulate data*/
				void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
								const SGVector<double>, /*a vector of parameters for the first process*/
								const SGVector<double>, /*a vector of parameters for the second process*/
								const int, /*select how the innovations are generated*/
								unsigned long /*a seed used to generate random numbers*/),
				/*calculate the distance-based test statistic*/
				double do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
								int, /*maximum truncation lag (L)*/
								int, /*lag_smooth*/
								double /*expn*/,
								int, /*number of subsets for TSCV*/
								int, /*minimum subset size for TSCV*/
								SGVector<int>, /*list of tree max depths (for GBM)*/
								SGVector<int>, /*list of numbers of iterations (for GBM)*/
								SGVector<double>, /*list of learning rates (for GBM)*/
								SGVector<double>, /*list of subset fractions (for GBM)*/
								SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int>, /*list of number of bags (for RF)*/
								double, double, /*quadratically and quartically integrate the kernel function*/
								int/*seed for random number generator*/),
				double kernel(double) /* a kernel function*/ >
	static void cValue (Matrix &asymp_REJF, Matrix &empir_CV,
						const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
						const int number_sampl, /*number of random samples drawn*/
						const int T, /*sample size*/
						const int L, /*maximum truncation lag*/
						const int lag_smooth, /*kernel bandwidth*/
						const double expn, /*exponent of the Euclidean distance*/
						const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
						int num_subsets, /*number of subsets for TSCV*/
						int min_subset_size, /*minimum subset size for TSCV*/
						SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
						SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
						SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
						SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
						SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
						SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
						int seed,
						ofstream &size_out);

	/* Calculate 5%- and 10%- asymptotic sizes and empirical critical values of transformed observations */
	template <	/*simulate data*/
				void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
								const SGVector<double>, /*a vector of parameters for the first process*/
								const SGVector<double>, /*a vector of parameters for the second process*/
								const int, /*select how the innovations are generated*/
								unsigned long /*a seed used to generate random numbers*/),
				/*calculate the distance-based test statistic*/
				double do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
								int, /*maximum truncation lag (L)*/
								int, /*lag_smooth*/
								double, /*expn*/
								double, /*expn_x*/
								int, /*number of subsets for TSCV*/
								int, /*minimum subset size for TSCV*/
								SGVector<int>, /*list of tree max depths (for GBM)*/
								SGVector<int>, /*list of numbers of iterations (for GBM)*/
								SGVector<double>, /*list of learning rates (for GBM)*/
								SGVector<double>, /*list of subset fractions (for GBM)*/
								SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int>, /*list of number of bags (for RF)*/
								double, double, /*quadratically and quartically integrate the kernel function*/
								int/*seed for random number generator*/),
				double kernel(double) /* a kernel function*/>
	static void cValue (Matrix &asymp_REJF, Matrix &empir_CV,
						const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
						const int number_sampl, /*number of random samples drawn*/
						const int T, /*sample size*/
						const int L, /*maximum truncation lag*/
						const int lag_smooth, /*kernel bandwidth*/
						const double expn, /*exponent of the Euclidean distance*/
						const double expn_x, /*exponent of data*/
						const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
						int num_subsets, /*number of subsets for TSCV*/
						int min_subset_size, /*minimum subset size for TSCV*/
						SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
						SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
						SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
						SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
						SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
						SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
						int seed,
						ofstream &size_out);

	/* Calculate 5%- and 10%- asymptotic and bootstrap sizes, and empirical critical values */
	template <	/*simulate data*/
				void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
								const SGVector<double>, /*a vector of parameters for the first process*/
								const SGVector<double>, /*a vector of parameters for the second process*/
								const int, /*select how the innovations are generated*/
								unsigned long /*a seed used to generate random numbers*/),
				/*calculate the distance-based test statistic*/
				std::tuple<	double /*p-value*/,
							double /*statistic*/,
							SGVector<double> /*bootstrap statistics*/> do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
																				int, /*L*/
																				int, /*lag_smooth*/
																				double, /*expn*/
																				const SGMatrix<double> &, const SGMatrix<double> &, /*num_B by T matrices
																																	of auxiliary random variables used for bootstrapping*/
																				int, /*number of subsets for TSCV*/
																				int, /*minimum subset size for TSCV*/
																				SGVector<int>, /*list of tree max depths (for GBM)*/
																				SGVector<int>, /*list of numbers of iterations (for GBM)*/
																				SGVector<double>, /*list of learning rates (for GBM)*/
																				SGVector<double>, /*list of subset fractions (for GBM)*/
																				SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																				SGVector<int>, /*list of number of bags (for RF)*/
																				double, double, /*quadratically and quartically integrate the kernel function*/
																				int /*seed for random number generator*/),
				double kernel(double) /* a kernel function*/>
	static void cValue (Matrix &asymp_REJF, Matrix &empir_CV,
						Matrix &bootstrap_REJF, /*5%- and 10% bootstrap sizes*/
						const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
						const int number_sampl, /*number of random samples drawn*/
						const int T, /*sample size*/
						const int L, /*maximum truncation lag*/
						const int lag_smooth, /*kernel bandwidth*/
						const double expn, /*exponent of the Euclidean distance*/
						const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices of auxiliary random variables
																											  used for bootstrapping*/
						const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
						int num_subsets, /*number of subsets for TSCV*/
						int min_subset_size, /*minimum subset size for TSCV*/
						SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
						SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
						SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
						SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
						SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
						SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
						int seed,
						ofstream &size_out);

	/* Calculate 5%- and 10%- empirical and asymptotic rejection frequencies */
	template <	/*simulate data*/
				void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
								const SGVector<double>, /*a vector of parameters for the first process*/
								const SGVector<double>, /*a vector of parameters for the second process*/
								const int, /*select how the innovations are generated*/
								unsigned long /*a seed used to generate random numbers*/),
				/*calculate the distance-based test statistic*/
				double do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
								int, /*maximum truncation lag (L)*/
								int, /*lag_smooth*/
								double /*expn*/,
								int, /*number of subsets for TSCV*/
								int, /*minimum subset size for TSCV*/
								SGVector<int>, /*list of tree max depths (for GBM)*/
								SGVector<int>, /*list of numbers of iterations (for GBM)*/
								SGVector<double>, /*list of learning rates (for GBM)*/
								SGVector<double>, /*list of subset fractions (for GBM)*/
								SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int>, /*list of number of bags (for RF)*/
								double, double, /*quadratically and quartically integrate the kernel function*/
								int/*seed for random number generator*/),
				double kernel(double) /* a kernel function*/ >
	static void power_f(Matrix &empir_REJF, Matrix &asymp_REJF,
						const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
						const int number_sampl, /*number of random samples drawn*/
						const int T, /*sample size*/
						const int L, /*maximum truncation lag*/
						const int lag_smooth, /*kernel bandwidth*/
						const double expn, /*exponent of the Euclidean distance*/
						const Matrix &empir_CV, /*5% and 10% empirical critical values*/
						const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
						int num_subsets, /*number of subsets for TSCV*/
						int min_subset_size, /*minimum subset size for TSCV*/
						SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
						SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
						SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
						SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
						SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
						SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
						const int choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
						int seed,
						ofstream &pwr_out);

	/* Calculate 5%- and 10%- empirical and asymptotic rejection frequencies of transformed observations */
	template <	/*simulate data*/
				void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
								const SGVector<double>, /*a vector of parameters for the first process*/
								const SGVector<double>, /*a vector of parameters for the second process*/
								const int, /*select how the innovations are generated*/
								unsigned long /*a seed used to generate random numbers*/),
				/*calculate the distance-based test statistic*/
				double do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
								int, /*maximum truncation lag (L)*/
								int, /*lag_smooth*/
								double, /*expn*/
								double, /*expn_x*/
								int, /*number of subsets for TSCV*/
								int, /*minimum subset size for TSCV*/
								SGVector<int>, /*list of tree max depths (for GBM)*/
								SGVector<int>, /*list of numbers of iterations (for GBM)*/
								SGVector<double>, /*list of learning rates (for GBM)*/
								SGVector<double>, /*list of subset fractions (for GBM)*/
								SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int>, /*list of number of bags (for RF)*/
								double, double, /*quadratically and quartically integrate the kernel function*/
								int/*seed for random number generator*/),
				double kernel(double) /* a kernel function*/ >
	static void power_f(Matrix &empir_REJF, Matrix &asymp_REJF,
						const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
						const int number_sampl, /*number of random samples drawn*/
						const int T, /*sample size*/
						const int L, /*maximum truncation lag*/
						const int lag_smooth, /*kernel bandwidth*/
						const double expn, /*exponent of the Euclidean distance*/
						const double expn_x, /*exponent of data*/
						const Matrix &empir_CV, /*5% and 10% empirical critical values*/
						const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
						int num_subsets, /*number of subsets for TSCV*/
						int min_subset_size, /*minimum subset size for TSCV*/
						SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
						SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
						SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
						SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
						SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
						SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
						const int choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
						int seed,
						ofstream &pwr_out);

	/* Calculate 5%- and 10%- empirical, asymptotic, and bootstrap rejection frequencies */
	template <	/*simulate data*/
				void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
								const SGVector<double>, /*a vector of parameters for the first process*/
								const SGVector<double>, /*a vector of parameters for the second process*/
								const int, /*select how the innovations are generated*/
								unsigned long /*a seed used to generate random numbers*/),
				/*calculate the distance-based test statistic*/
				std::tuple<	double /*p-value*/,
							double /*statistic*/,
							SGVector<double> /*bootstrap statistics*/> do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
																				int, /*L*/
																				int, /*lag_smooth*/
																				double, /*expn*/
																				const SGMatrix<double> &, const SGMatrix<double> &, /*num_B by T matrices
																																	of auxiliary random variables used for bootstrapping*/
																				int, /*number of subsets for TSCV*/
																				int, /*minimum subset size for TSCV*/
																				SGVector<int>, /*list of tree max depths (for GBM)*/
																				SGVector<int>, /*list of numbers of iterations (for GBM)*/
																				SGVector<double>, /*list of learning rates (for GBM)*/
																				SGVector<double>, /*list of subset fractions (for GBM)*/
																				SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																				SGVector<int>, /*list of number of bags (for RF)*/
																				double, double, /*quadratically and quartically integrate the kernel function*/
																				int /*seed for random number generator*/),
				double kernel(double) /* a kernel function*/ >
	static void power_f(Matrix &empir_REJF, Matrix &asymp_REJF,
						Matrix &bootstrap_REJF, /*5%- and 10%- bootstrap rejection frequencies*/
						const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
						const int number_sampl, /*number of random samples drawn*/
						const int T, /*sample size*/
						const int L, /*maximum truncation lag*/
						const int lag_smooth, /*kernel bandwidth*/
						const double expn, /*exponent of the Euclidean distance*/
						const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices of auxiliary random variables
																											  used for bootstrapping*/
						const Matrix &empir_CV, /*5% and 10% empirical critical values*/
						const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
						int num_subsets, /*number of subsets for TSCV*/
						int min_subset_size, /*minimum subset size for TSCV*/
						SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
						SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
						SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
						SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
						SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
						SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
						const int choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
						int seed,
						ofstream &pwr_out);



		//calculate 5%- and 10%- asymptotic sizes and empirical critical values when X and Y are generated by a univariate non-Gaussian AR(2) process
        //and a bivariate non-Gaussian AR(2) process respectively. INPUT: number of random samples generated (number_sampl), a sample size (T), a lag truncation (TL),
        //a lag-smoothing parameter (lag_smooth), A 6x1 vector of coefficients for X (alpha_X), a 6x2 matrix of coefficients for Y1 and Y2 (alpha_Y),
        //Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed)
        //with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, an exponent for the Euclidean distance (expn in (1,2)), and a 5%- and 10%- asymptotic critical
        //values (asymp_CV), a kernel weight (kernel), a data-generating process (gen_DGP), a conditional mean function (cmean), an OLS estimator (est_DGP).
        //OUTPUT: 5%- and 10%- sizes (asymp_REJF) and empirical critical values (empir_CV).
        template <double kernel (double), void gen_DGP (Matrix &, Matrix &, Matrix &, const Matrix, const Matrix, const Matrix &, const double, const double, const int, unsigned long),
		          void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
        static void cValue (Matrix &asymp_REJF, Matrix &empir_CV, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                            const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t,
					        const Matrix &eta2_s, const Matrix &delta, const double expn, const Matrix &asymp_CV, unsigned long seed, ofstream &size_out);
		//calculate 5% and 10% empirical and asymptotic rejection frequencies at a cut-off point when each X or Y is generated by a non-Gaussian AR(2) process.
        //INPUT: number of random samples generated (number_sampl), a sample size (T), a lag truncation (TL), a lag-smoothing parameter (lag_smooth),
        //6x1 vectors of AR coefficients (alpha_X and alpha_Y), vectors of random errors (epsilon_t, epsilon_s, eta_t and eta_s) generated by
        //gen_RAN (epsilon, eta, delta, 0., 0, rseed) with random seeds (rseed), a delta in (-1,1), an exponent for the Euclidean distance (expn in (1,2)),
        //5% and 10% empirical critical values (empir_CV), 5% and 10% asymptotic critical values (asymp_CV),
        //a DGP definition: choose_alt = 0 (independent), = 1 (correlated), = 2 (uncorrelated but dependent), = 3 (correlated and dependent),
        //a kernel weight (kernel), a data-generating process (gen_DGP), a conditional mean function (cmean),
        //an OLS estimator (est_DGP). OUTPUT: 5% and 10% empirical rejection frequencies (empir_REJF) and asymptotic rejection frequencies (asymp_REJF).
        template <double kernel (double), void gen_DGP (Matrix &, Matrix &, const Matrix, const Matrix, const double, const double, const int, unsigned long),
		          void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
        static void power_f (Matrix &empir_REJF, Matrix &asymp_REJF, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                             const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta_t, const Matrix &eta_s, const double delta,
					         const double rho, const double expn, const Matrix &empir_CV, const Matrix &asymp_CV, const int choose_alt, unsigned long seed,
					         ofstream &pwr_out);
		//calculate 5% and 10% empirical and asymptotic rejection frequencies at a cut-off point when X and Y are generated by a univariate non-Gaussian AR(2) process
        //and a bivariate non-Gaussian AR(2) process respectively. INPUT: number of random samples generated (number_sampl), a sample size (T), a lag truncation (TL),
        //a lag-smoothing parameter (lag_smooth), A 6x1 vector of coefficients for X (alpha_X), a 6x2 matrix of coefficients for Y1 and Y2 (alpha_Y),
        //Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed)
        //with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, coefficients of correlation (rho12 and rho13), an exponent for the Euclidean distance (expn in (1,2)),
        //5% and 10% empirical critical values (empir_CV), 5% and 10% asymptotic critical values (asymp_CV),
        //a DGP definition: choose_alt = 0 (independent), = 1 (correlated), = 2 (uncorrelated but dependent), = 3 (correlated and dependent),
        //a kernel weight (kernel), an error-generating process (gen_RAN), a data-generating process (gen_DGP), a conditional mean function (cmean),
        //and an OLS estimator (est_DGP). OUTPUT: 5% and 10% empirical rejection frequencies (empir_REJF) and asymptotic rejection frequencies (asymp_REJF).
        template <double kernel (double), void gen_DGP (Matrix &, Matrix &, Matrix &, const Matrix, const Matrix, const Matrix &, const double, const double, const int, unsigned long),
		          void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
        static void power_f (Matrix &empir_REJF, Matrix &asymp_REJF, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                             const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t,
					         const Matrix &eta2_s, const Matrix &delta, const double rho12, const double rho13, const double expn, const Matrix &empir_CV,
					         const Matrix &asymp_CV, const int choose_alt, unsigned long seed, ofstream &pwr_out);
		//calculate 5% and 10% bootstrap rejection frequencies when X and Y are generated by a univariate non-Gaussian AR(2) process and a bivariate non-Gaussian AR(2) process respectively.
		// INPUT: number of random samples generated (number_sampl), a lag truncation (TL), a lag-smoothing parameter (lag_smooth), A 6x1 vector of coefficients for X (alpha_X),
		//a 6x2 matrix of coefficients for Y1 and Y2 (alpha_Y), Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by
		//gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed) with random seeds (rseed), two BxT matrices of i.i.d. auxiliary random variables for wild bootstrap (xi_x and xi_y),
		//a 3x1 vector (delta) in (-1,1)^3, coefficients of correlation (rho12 and rho13), an exponent for the Euclidean distance (expn in (1,2)), a DGP definition: choose_alt = 0 (independent),
		//= 1 (correlated), = 2 (uncorrelated but dependent), = 3 (correlated and dependent), a kernel weight (kernel), an error-generating process (gen_RAN), a data-generating process (gen_DGP),
		//a conditional mean function (cmean), and an OLS estimator (est_DGP).
		//OUTPUT: 5% and 10% bootstrap rejection frequencies (bootstrap_REJF)
		template <double kernel (double), void gen_DGP (Matrix &, Matrix &, Matrix &, const Matrix, const Matrix, const Matrix &, const double, const double, const int, unsigned long),
						void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
		static void power_f (Matrix &bootstrap_REJF, const int number_sampl, const int TL, const int lag_smooth, const Matrix &alpha_X, const Matrix &alpha_Y,
									const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t, const Matrix &eta2_s,
									const Matrix &xi_x, const Matrix &xi_y, const Matrix &delta, const double rho12, const double rho13, const double expn, const int choose_alt,
									unsigned long seed, ofstream &pwr_out);

};

/* Calculate 5%- and 10%- asymptotic sizes and empirical critical values. */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			/*calculate the distance-based test statistic*/
			double do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
							int, /*maximum truncation lag (L)*/
							int, /*lag_smooth*/
							double, /*expn*/
							int, /*number of subsets for TSCV*/
							int, /*minimum subset size for TSCV*/
							SGVector<int>, /*list of tree max depths (for GBM)*/
							SGVector<int>, /*list of numbers of iterations (for GBM)*/
							SGVector<double>, /*list of learning rates (for GBM)*/
							SGVector<double>, /*list of subset fractions (for GBM)*/
							SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
							SGVector<int>, /*list of number of bags (for RF)*/
							double, double, /*quadratically and quartically integrate the kernel function*/
							int/*seed for random number generator*/),
			double kernel(double) /* a kernel function*/>
void ML_dcorr_power::cValue (	Matrix &asymp_REJF, Matrix &empir_CV,
								const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
								const int number_sampl, /*number of random samples drawn*/
								const int T, /*sample size*/
								const int L, /*maximum truncation lag*/
								const int lag_smooth, /*kernel bandwidth*/
								const double expn, /*exponent of the Euclidean distance*/
								const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
								int num_subsets, /*number of subsets for TSCV*/
								int min_subset_size, /*minimum subset size for TSCV*/
								SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
								SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
								SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
								SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
								SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
								int seed,
								ofstream &size_out) {
	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

    //calculate integrals of quadratic and quartic functions of the kernel weight
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	ML_DCORR::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);

	size_out << std::fixed << std::setprecision(5);
	size_out << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;

    cout << "Calculating empirical critical values ..." << endl;

    int i = 1, choose_alt = 0;
    //SGMatrix<double> X(T,2), Y(T,2);
    Matrix tvalue(number_sampl,1);
    int asymp_REJ_5 = 0, asymp_REJ_10 = 0;

	#pragma omp parallel for default(shared) reduction (+:asymp_REJ_5,asymp_REJ_10) schedule(dynamic,CHUNK) private(i) firstprivate(rseed)
	for (i = 1; i <= number_sampl; i++) {
		SGMatrix<double> X(T,2), Y(T,2);
		//#pragma omp critical
		{
			rseed = gsl_rng_get(r); //generate a random seed

			//draw two independent random samples from the dgp of X and Y each using a random seed
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);
		}

			//calculate the test statistic
			tvalue(i) = do_Test(X, Y, L, lag_smooth, expn,
								num_subsets, /*number of subsets for TSCV*/
								min_subset_size, /*minimum subset size for TSCV*/
								tree_max_depths_list, /*list of tree max depths (for GBM)*/
								num_iters_list, /*list of numbers of iterations (for GBM)*/
								learning_rates_list, /*list of learning rates (for GBM)*/
								subset_fractions_list, /*list of subset fractions (for GBM)*/
								num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								num_bags_list, /*list of number of bags (for RF)*/
								kernel_QDSum, kernel_QRSum, seed);


		#pragma omp critical
		{
			size_out << tvalue(i) << endl;
		}

		if (tvalue(i) >= asymp_CV(1)) ++asymp_REJ_5; //using 5%-critical value
		if (tvalue(i) >= asymp_CV(2)) ++asymp_REJ_10; //using 10%-critical value
	}

	asymp_REJF(1) = ((double) asymp_REJ_5/number_sampl); //calculate sizes
	asymp_REJF(2) = ((double) asymp_REJ_10/number_sampl);
	empir_CV(1) = quantile (tvalue, 0.95); //calculate quantiles
	empir_CV(2) = quantile (tvalue, 0.90);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << empir_CV(2) << endl;
	gsl_rng_free (r); //free up memory
}



/* Calculate 5%- and 10%- asymptotic sizes and empirical critical values with transformed observations. */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			/*calculate the distance-based test statistic*/
			double do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
							int, /*maximum truncation lag (L)*/
							int, /*lag_smooth*/
							double, /*expn*/
							double, /*expn_x*/
							int, /*number of subsets for TSCV*/
							int, /*minimum subset size for TSCV*/
							SGVector<int>, /*list of tree max depths (for GBM)*/
							SGVector<int>, /*list of numbers of iterations (for GBM)*/
							SGVector<double>, /*list of learning rates (for GBM)*/
							SGVector<double>, /*list of subset fractions (for GBM)*/
							SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
							SGVector<int>, /*list of number of bags (for RF)*/
							double, double, /*quadratically and quartically integrate the kernel function*/
							int/*seed for random number generator*/),
			double kernel(double) /* a kernel function*/>
void ML_dcorr_power::cValue (	Matrix &asymp_REJF, Matrix &empir_CV,
								const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
								const int number_sampl, /*number of random samples drawn*/
								const int T, /*sample size*/
								const int L, /*maximum truncation lag*/
								const int lag_smooth, /*kernel bandwidth*/
								const double expn, /*exponent of the Euclidean distance*/
								const double expn_x, /*exponent of data*/
								const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
								int num_subsets, /*number of subsets for TSCV*/
								int min_subset_size, /*minimum subset size for TSCV*/
								SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
								SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
								SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
								SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
								SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
								int seed,
								ofstream &size_out) {
	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    int rseed = 1;

    //calculate integrals of quadratic and quartic functions of the kernel weight
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	ML_DCORR::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);

	size_out << std::fixed << std::setprecision(5);
	size_out << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;

    cout << "Calculating empirical critical values ..." << endl;

    int i = 1, choose_alt = 0;
    SGMatrix<double> X(T,2), Y(T,2);
    Matrix tvalue(number_sampl,1);
    int asymp_REJ_5 = 0, asymp_REJ_10 = 0;

	//#pragma omp parallel for default(shared) reduction (+:asymp_REJ_5,asymp_REJ_10) schedule(dynamic,CHUNK) private(i) firstprivate(rseed, X, Y)
	for (i = 1; i <= number_sampl; i++) {
		rseed = gsl_rng_get(r); //generate a random seed

		//draw two independent random samples from the dgp of X and Y each using a random seed
		gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);

		//calculate the test statistic
		tvalue(i) = do_Test(X, Y, L, lag_smooth, expn, expn_x,
							num_subsets, /*number of subsets for TSCV*/
							min_subset_size, /*minimum subset size for TSCV*/
							tree_max_depths_list, /*list of tree max depths (for GBM)*/
							num_iters_list, /*list of numbers of iterations (for GBM)*/
							learning_rates_list, /*list of learning rates (for GBM)*/
							subset_fractions_list, /*list of subset fractions (for GBM)*/
							num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
							num_bags_list, /*list of number of bags (for RF)*/
							kernel_QDSum, kernel_QRSum, seed);

		size_out << tvalue(i) << endl;

		if (tvalue(i) >= asymp_CV(1)) ++asymp_REJ_5; //using 5%-critical value
		if (tvalue(i) >= asymp_CV(2)) ++asymp_REJ_10; //using 10%-critical value
	}

	asymp_REJF(1) = ((double) asymp_REJ_5/number_sampl); //calculate sizes
	asymp_REJF(2) = ((double) asymp_REJ_10/number_sampl);
	empir_CV(1) = quantile (tvalue, 0.95); //calculate quantiles
	empir_CV(2) = quantile (tvalue, 0.90);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << empir_CV(2) << endl;
	gsl_rng_free (r); //free up memory
}

/* Calculate 5%- and 10%- asymptotic and bootstrap sizes, and empirical critical values */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			/*calculate the distance-based test statistic*/
			std::tuple<	double /*p-value*/,
						double /*statistic*/,
						SGVector<double> /*bootstrap statistics*/> do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
																			int, /*L*/
																			int, /*lag_smooth*/
																			double, /*expn*/
																			const SGMatrix<double> &, const SGMatrix<double> &, /*num_B by T matrices
																																of auxiliary random variables used for bootstrapping*/
																			int, /*number of subsets for TSCV*/
																			int, /*minimum subset size for TSCV*/
																			SGVector<int>, /*list of tree max depths (for GBM)*/
																			SGVector<int>, /*list of numbers of iterations (for GBM)*/
																			SGVector<double>, /*list of learning rates (for GBM)*/
																			SGVector<double>, /*list of subset fractions (for GBM)*/
																			SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int>, /*list of number of bags (for RF)*/
																			double, double, /*quadratically and quartically integrate the kernel function*/
																			int /*seed for random number generator*/ ),
			double kernel(double) /* a kernel function*/>
void ML_dcorr_power::cValue (	Matrix &asymp_REJF, Matrix &empir_CV,
								Matrix &bootstrap_REJF, /*5%- and 10% bootstrap sizes*/
								const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
								const int number_sampl, /*number of random samples drawn*/
								const int T, /*sample size*/
								const int L, /*maximum truncation lag*/
								const int lag_smooth, /*kernel bandwidth*/
								const double expn, /*exponent of the Euclidean distance*/
								const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices of auxiliary random variables
																													  used for bootstrapping*/
								const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
								int num_subsets, /*number of subsets for TSCV*/
								int min_subset_size, /*minimum subset size for TSCV*/
								SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
								SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
								SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
								SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
								SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
								int seed,
								ofstream &size_out) {
	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

    //calculate integrals of quadratic and quartic functions of the kernel weight
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	ML_DCORR::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);

	size_out << std::fixed << std::setprecision(5);
	size_out << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;

    cout << "Calculating empirical critical values ..." << endl;

    int i = 1, choose_alt = 0;
    int num_B = xi_x.num_rows;
    SGMatrix<double> X(T,2), Y(T,2);
    Matrix pvalue(number_sampl,1), tvalue(number_sampl,1);
    int asymp_REJ_5 = 0, asymp_REJ_10 = 0, bootstrap_REJ_5 = 0, bootstrap_REJ_10 = 0;

    size_out << std::fixed << std::setprecision(5);
	size_out << " pvalue " << "   ,   " << "t-stat" << endl;

	#pragma omp parallel for default(shared) reduction (+:asymp_REJ_5,asymp_REJ_10,bootstrap_REJ_5,bootstrap_REJ_10) schedule(dynamic,CHUNK) \
																														private(i) firstprivate(rseed)
	for (i = 1; i <= number_sampl; i++) {
		SGMatrix<double> X(T,2), Y(T,2);
        SGVector<double> cstat_bootstrap(num_B);

		rseed = gsl_rng_get(r); //generate a random seed

		//draw two independent random samples from the dgp of X and Y each using a random seed
		gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);

		//calculate the test statistic
		std::tie(pvalue(i), tvalue(i), cstat_bootstrap) = do_Test(X, Y, L, lag_smooth, expn,
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
		#pragma omp critical
        {
            size_out <<  pvalue(i) << "   ,   " << tvalue(i) << endl;
        }

		if (tvalue(i) >= asymp_CV(1)) ++asymp_REJ_5; //using 5%-critical value
		if (tvalue(i) >= asymp_CV(2)) ++asymp_REJ_10; //using 10%-critical value
		if (pvalue(i) <= 0.05)  ++bootstrap_REJ_5;//using 5%-critical value
		if (pvalue(i) <= 0.10)  ++bootstrap_REJ_10;//using 10%-critical value
	}

	asymp_REJF(1) = ((double) asymp_REJ_5/number_sampl); //calculate sizes
	asymp_REJF(2) = ((double) asymp_REJ_10/number_sampl);
	empir_CV(1) = quantile (tvalue, 0.95); //calculate quantiles
	empir_CV(2) = quantile (tvalue, 0.90);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << empir_CV(2) << endl;

	bootstrap_REJF(1) = ((double) bootstrap_REJ_5/number_sampl);
	bootstrap_REJF(2) = ((double) bootstrap_REJ_10/number_sampl);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-bootstrap size for T =" << T << " is " << bootstrap_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-bootstrap size for T = " << T << " is " << bootstrap_REJF(2) << endl;

	gsl_rng_free (r); //free up memory
}



/* Calculate 5%- and 10%- empirical and asymptotic rejection frequencies */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			/*calculate the distance-based test statistic*/
			double do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
							int, /*maximum truncation lag (L)*/
							int, /*lag_smooth*/
							double /*expn*/,
							int, /*number of subsets for TSCV*/
							int, /*minimum subset size for TSCV*/
							SGVector<int>, /*list of tree max depths (for GBM)*/
							SGVector<int>, /*list of numbers of iterations (for GBM)*/
							SGVector<double>, /*list of learning rates (for GBM)*/
							SGVector<double>, /*list of subset fractions (for GBM)*/
							SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
							SGVector<int>, /*list of number of bags (for RF)*/
							double, double, /*quadratically and quartically integrate the kernel function*/
							int/*seed for random number generator*/),
			double kernel(double) /* a kernel function*/ >
void ML_dcorr_power::power_f(	Matrix &empir_REJF, Matrix &asymp_REJF,
								const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
								const int number_sampl, /*number of random samples drawn*/
								const int T, /*sample size*/
								const int L, /*maximum truncation lag*/
								const int lag_smooth, /*kernel bandwidth*/
								const double expn, /*exponent of the Euclidean distance*/
								const Matrix &empir_CV, /*5% and 10% empirical critical values*/
								const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
								int num_subsets, /*number of subsets for TSCV*/
								int min_subset_size, /*minimum subset size for TSCV*/
								SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
								SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
								SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
								SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
								SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
								const int choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								int seed,
								ofstream &pwr_out) {
	gsl_rng * r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

	cout << "Calculating rejection frequencies for T = " << T << endl;
	cout << "5%- and 10%- empirical critical values = " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << " ..." << endl;
	cout << "5%- and 10%- asymptotic critical values = " << "(" << asymp_CV(1) << " , " << asymp_CV(2) << ")" << " ..." << endl;

	 //calculate integrals of quadratic and quartic functions of the kernel weight
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	ML_DCORR::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);
	cout << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;

	int i = 1, empir_REJ_5 = 0, asymp_REJ_5 = 0, empir_REJ_10 = 0, asymp_REJ_10 = 0;

	Matrix tvalue(number_sampl,1);
	//SGMatrix<double> X(T,2), Y(T,2);

	pwr_out << std::fixed << std::setprecision(5);


	#pragma omp parallel for default(shared) reduction(+:empir_REJ_5,empir_REJ_10,asymp_REJ_5,asymp_REJ_10) schedule(dynamic,CHUNK) private(i) firstprivate(rseed)
	for (i = 1; i <= number_sampl; ++i) {
		SGMatrix<double> X(T,2), Y(T,2);
		//#pragma omp critical
		{
			rseed = gsl_rng_get(r); //generate a random seed

			//draw two independent random samples from the dgp of X and Y each using a random seed
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);


		//calculate the test statistic
		tvalue(i) = do_Test(X, Y, L, lag_smooth, expn,
							num_subsets, /*number of subsets for TSCV*/
							min_subset_size, /*minimum subset size for TSCV*/
							tree_max_depths_list, /*list of tree max depths (for GBM)*/
							num_iters_list, /*list of numbers of iterations (for GBM)*/
							learning_rates_list, /*list of learning rates (for GBM)*/
							subset_fractions_list, /*list of subset fractions (for GBM)*/
							num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
							num_bags_list, /*list of number of bags (for RF)*/
							kernel_QDSum, kernel_QRSum, seed);
		}

		#pragma omp critical
		{
			pwr_out << tvalue(i) << endl;
		}

        if (tvalue(i) >= empir_CV(1)) ++empir_REJ_5;//using 5%-critical value
        if (tvalue(i) >= asymp_CV(1)) ++asymp_REJ_5;//using 5%-critical value
        if (tvalue(i) >= empir_CV(2)) ++empir_REJ_10;//using 10%-critical value
        if (tvalue(i) >= asymp_CV(2)) ++asymp_REJ_10;//using 10%-critical value
	}

	empir_REJF(1) = ((double) empir_REJ_5/number_sampl);
	empir_REJF(2) = ((double) empir_REJ_10/number_sampl);
	asymp_REJF(1) = ((double) asymp_REJ_5/number_sampl);
	asymp_REJF(2) = ((double) asymp_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << empir_REJF(1) << " and " << empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << asymp_REJF(1) << " and " << asymp_REJF(2) << endl;

	gsl_rng_free (r); // free up memory
}


/* Calculate 5%- and 10%- empirical, asymptotic, and bootstrap rejection frequencies */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			/*calculate the distance-based test statistic*/
			std::tuple<	double /*p-value*/,
						double /*statistic*/,
						SGVector<double> /*bootstrap statistics*/> do_Test(	const SGMatrix<double> &, const SGMatrix<double> &,
																			int, /*L*/
																			int, /*lag_smooth*/
																			double, /*expn*/
																			const SGMatrix<double> &, const SGMatrix<double> &, /*num_B by T matrices
																																of auxiliary random variables used for bootstrapping*/
																			int, /*number of subsets for TSCV*/
																			int, /*minimum subset size for TSCV*/
																			SGVector<int>, /*list of tree max depths (for GBM)*/
																			SGVector<int>, /*list of numbers of iterations (for GBM)*/
																			SGVector<double>, /*list of learning rates (for GBM)*/
																			SGVector<double>, /*list of subset fractions (for GBM)*/
																			SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int>, /*list of number of bags (for RF)*/
																			double, double, /*quadratically and quartically integrate the kernel function*/
																			int /*seed for random number generator*/),
			double kernel(double) /* a kernel function*/ >
void ML_dcorr_power::power_f(	Matrix &empir_REJF, Matrix &asymp_REJF,
								Matrix &bootstrap_REJF, /*5%- and 10%- bootstrap rejection frequencies*/
								const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
								const int number_sampl, /*number of random samples drawn*/
								const int T, /*sample size*/
								const int L, /*maximum truncation lag*/
								const int lag_smooth, /*kernel bandwidth*/
								const double expn, /*exponent of the Euclidean distance*/
								const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices of auxiliary random variables
																													  used for bootstrapping*/
								const Matrix &empir_CV, /*5% and 10% empirical critical values*/
								const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
								int num_subsets, /*number of subsets for TSCV*/
								int min_subset_size, /*minimum subset size for TSCV*/
								SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
								SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
								SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
								SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
								SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
								const int choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
								int seed,
								ofstream &pwr_out) {
	gsl_rng * r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

	cout << "Calculating rejection frequencies for T = " << T << endl;
	cout << "5%- and 10%- empirical critical values = " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << " ..." << endl;
	cout << "5%- and 10%- asymptotic critical values = " << "(" << asymp_CV(1) << " , " << asymp_CV(2) << ")" << " ..." << endl;

	 //calculate integrals of quadratic and quartic functions of the kernel weight
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	ML_DCORR::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);
	cout << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;

	int i = 1, empir_REJ_5 = 0, asymp_REJ_5 = 0, empir_REJ_10 = 0, asymp_REJ_10 = 0, bootstrap_REJ_5 = 0,bootstrap_REJ_10 = 0;

	Matrix tvalue(number_sampl,1), pvalue(number_sampl, 1);
	int num_B = xi_x.num_rows;

	pwr_out << std::fixed << std::setprecision(5);
	pwr_out << "pvalue" << " , " << "t-stat" << endl;

	#pragma omp parallel for default(shared) reduction(+:empir_REJ_5,empir_REJ_10,asymp_REJ_5,asymp_REJ_10,bootstrap_REJ_5,bootstrap_REJ_10) \
																							schedule(dynamic,CHUNK) private(i) firstprivate(rseed)
	for (i = 1; i <= number_sampl; ++i) {
		SGMatrix<double> X(T,2), Y(T,2);
		SGVector<double> cstat_bootstrap(num_B);
		//#pragma omp critical
		{
			rseed = gsl_rng_get(r); //generate a random seed

			//draw two independent random samples from the dgp of X and Y each using a random seed
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);


		//calculate the test statistic
		std::tie(pvalue(i), tvalue(i), cstat_bootstrap) = do_Test(X, Y, L, lag_smooth, expn,
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
		}

		#pragma omp critical
		{
			pwr_out << pvalue(i) << " , " << tvalue(i) << endl;
		}

		if (tvalue(i) >= empir_CV(1)) ++empir_REJ_5;//using 5%-critical value
		if (tvalue(i) >= asymp_CV(1)) ++asymp_REJ_5;//using 5%-critical value
		if (tvalue(i) >= empir_CV(2)) ++empir_REJ_10;//using 10%-critical value
		if (tvalue(i) >= asymp_CV(2)) ++asymp_REJ_10;//using 10%-critical value
		if (pvalue(i) <= 0.05)  ++bootstrap_REJ_5;//using 5%-critical value
		if (pvalue(i) <= 0.10)  ++bootstrap_REJ_10;//using 10%-critical value
	}

	empir_REJF(1) = ((double) empir_REJ_5/number_sampl);
	empir_REJF(2) = ((double) empir_REJ_10/number_sampl);
	asymp_REJF(1) = ((double) asymp_REJ_5/number_sampl);
	asymp_REJF(2) = ((double) asymp_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << empir_REJF(1) << " and " << empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << asymp_REJF(1) << " and " << asymp_REJF(2) << endl;

	bootstrap_REJF(1) = ((double) bootstrap_REJ_5/number_sampl);
	bootstrap_REJF(2) = ((double) bootstrap_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% bootstrap reject frequencies for choose_alt = " << choose_alt << " are "
	        << bootstrap_REJF(1) << " and " << bootstrap_REJF(2) << endl;


	gsl_rng_free (r); // free up memory
}



//calculate 5%- and 10%- asymptotic sizes and empirical critical values when X and Y are generated by a univariate non-Gaussian AR(2) process
//and a bivariate non-Gaussian AR(2) process respectively. INPUT: number of random samples generated (number_sampl), a sample size (T), a lag truncation (TL),
//a lag-smoothing parameter (lag_smooth), A 6x1 vector of coefficients for X (alpha_X), a 6x2 matrix of coefficients for Y1 and Y2 (alpha_Y),
//Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed)
//with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, an exponent for the Euclidean distance (expn in (1,2)), and a 5%- and 10%- asymptotic critical
//values (asymp_CV), a kernel weight (kernel), a data-generating process (gen_DGP), a conditional mean function (cmean), an OLS estimator (est_DGP).
//OUTPUT: 5%- and 10%- sizes (asymp_REJF) and empirical critical values (empir_CV).
template <double kernel (double), void gen_DGP (Matrix &, Matrix &, Matrix &, const Matrix, const Matrix, const Matrix &, const double, const double, const int, unsigned long),
		  void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
void ML_dcorr_power::cValue (Matrix &asymp_REJF, Matrix &empir_CV, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                      const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t,
					  const Matrix &eta2_s, const Matrix &delta, const double expn, const Matrix &asymp_CV, unsigned long seed, ofstream &size_out) {
	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long rseed = 1;
    //calculate integrals of quadratic and quartic functions of the kernel weight
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	NGDist_corr::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);
	cout << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;
    cout << "Calculating empirical critical values ..." << endl;
    int i = 1, j = 1, size_alpha_X = alpha_X.nRow(), size_alpha_Y = alpha_Y.nRow();
    Matrix X(T,1), Y1(T,1), Y2(T,1), alpha_X_hat(size_alpha_X,1), alpha_Y1_hat(size_alpha_Y,1), alpha_Y2_hat(size_alpha_Y,1), alpha_Y_hat(size_alpha_Y,2),
	       resid_X(T-2,1), resid_Y1(T-2,1), resid_Y2(T-2,1), tvalue(number_sampl,1);
    int asymp_REJ_5 = 0, asymp_REJ_10 = 0;
	#pragma omp parallel for default(shared) reduction (+:asymp_REJ_5,asymp_REJ_10) schedule(dynamic,CHUNK) private(i,j) \
                                                                          firstprivate(rseed,X,Y1,Y2,resid_X,resid_Y1,resid_Y2,alpha_X_hat,alpha_Y1_hat,alpha_Y2_hat,alpha_Y_hat)
	for (i = 1; i <= number_sampl; i++) {
        rseed = gsl_rng_get (r);
		gen_DGP (X, Y1, Y2, alpha_X, alpha_Y, delta, 0., 0., 0, rseed); //draw two independent random samples from the dgp of X and Y each using a random seed
		//then use these samples to estimate the AR coefficients
		est_DGP (resid_X, alpha_X_hat, X);
		est_DGP (resid_Y1, alpha_Y1_hat, Y1);
		est_DGP (resid_Y2, alpha_Y2_hat, Y2);
		for (j = 1; j <= size_alpha_Y; j++) {
			alpha_Y_hat(j,1) = alpha_Y1_hat(j); //collect all estimates in a matrix
			alpha_Y_hat(j,2) = alpha_Y2_hat(j);
		}
		//cout << alpha_X_hat(1) << " , " << alpha_Y_hat(1) << endl;
		//calculate the test statistic
		tvalue(i) = NGDist_corr::do_Test<kernel, cmean> (X, Y1, Y2, TL, lag_smooth, kernel_QRSum, alpha_X_hat, alpha_Y_hat, epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t, eta2_s, expn);
		size_out << tvalue(i) << endl;
        if (tvalue(i) >= asymp_CV(1)) ++asymp_REJ_5; //using 5%-critical value
        if (tvalue(i) >= asymp_CV(2)) ++asymp_REJ_10; //using 10%-critical value
	}
	asymp_REJF(1) = ((double) asymp_REJ_5/number_sampl); //calculate sizes
	asymp_REJF(2) = ((double) asymp_REJ_10/number_sampl);
	empir_CV(1) = quantile (tvalue, 0.95); //calculate quantiles
	empir_CV(2) = quantile (tvalue, 0.90);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << empir_CV(2) << endl;
	gsl_rng_free (r);
}

//calculate 5% and 10% empirical and asymptotic rejection frequencies at a cut-off point when each X or Y is generated by a non-Gaussian AR(2) process.
//INPUT: number of random samples generated (number_sampl), a sample size (T), a lag truncation (TL), a lag-smoothing parameter (lag_smooth),
//6x1 vectors of AR coefficients (alpha_X and alpha_Y), vectors of random errors (epsilon_t, epsilon_s, eta_t and eta_s) generated by
//gen_RAN (epsilon, eta, delta, 0., 0, rseed) with random seeds (rseed), a delta in (-1,1), an exponent for the Euclidean distance (expn in (1,2)),
//5% and 10% empirical critical values (empir_CV), 5% and 10% asymptotic critical values (asymp_CV),
//a DGP definition: choose_alt = 0 (independent), = 1 (correlated), = 2 (uncorrelated but dependent), = 3 (correlated and dependent),
//a kernel weight (kernel), a data-generating process (gen_DGP), a conditional mean function (cmean),
//an OLS estimator (est_DGP). OUTPUT: 5% and 10% empirical rejection frequencies (empir_REJF) and asymptotic rejection frequencies (asymp_REJF).
template <double kernel (double), void gen_DGP (Matrix &, Matrix &, const Matrix, const Matrix, const double, const double, const int, unsigned long),
		  void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
void ML_dcorr_power::power_f (Matrix &empir_REJF, Matrix &asymp_REJF, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                       const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta_t, const Matrix &eta_s, const double delta,
					   const double rho, const double expn, const Matrix &empir_CV, const Matrix &asymp_CV, const int choose_alt, unsigned long seed,
					   ofstream &pwr_out)  {
	gsl_rng * r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long rseed = 1;
	cout << "Calculating rejection frequencies for T = " << T << endl;
	cout << "5%- and 10%- empirical critical values = " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << " ..." << endl;
	cout << "5%- and 10%- asymptotic critical values = " << "(" << asymp_CV(1) << " , " << asymp_CV(2) << ")" << " ..." << endl;
	//calculate integrals of quadratic and quartic functions of the kernel weight
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	NGDist_corr::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);
	cout << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;
	int i = 1, size_alpha_X = alpha_X.nRow(), size_alpha_Y = alpha_Y.nRow(), empir_REJ_5 = 0, asymp_REJ_5 = 0, empir_REJ_10 = 0, asymp_REJ_10 = 0;
	auto tvalue = 0.;
	Matrix X(T,1), Y(T,1), alpha_X_hat(size_alpha_X,1), alpha_Y_hat(size_alpha_Y,1), resid_X(T-2,1), resid_Y(T-2,1);
	#pragma omp parallel for default(shared) reduction (+:empir_REJ_5,empir_REJ_10,asymp_REJ_5,asymp_REJ_10) schedule(dynamic,CHUNK) private(i) \
                                                                                                                                                                               firstprivate(rseed,X,Y,resid_X,resid_Y,alpha_X_hat,alpha_Y_hat,tvalue)
	for (i = 1; i <= number_sampl; ++i) {
		rseed = gsl_rng_get (r); //set a random seed for the random number generator
		//draw two samples from the dgp of X and Y each using a random seed
		gen_DGP (X, Y, alpha_X, alpha_Y, delta, rho, choose_alt, rseed);
		//then use these samples to estimate the AR coefficients
		est_DGP (resid_X, alpha_X_hat, X);
		est_DGP (resid_Y, alpha_Y_hat, Y);
		//calculate the test statistic
		tvalue = NGDist_corr::do_Test<kernel, cmean> (X, Y, TL, lag_smooth, kernel_QRSum, alpha_X_hat, alpha_Y_hat, epsilon_t, epsilon_s, eta_t,
		                                              eta_s, expn);
        pwr_out << tvalue << endl;
        if (tvalue >= empir_CV(1)) ++empir_REJ_5;//using 5%-critical value
        if (tvalue >= asymp_CV(1)) ++asymp_REJ_5;//using 5%-critical value
        if (tvalue >= empir_CV(2)) ++empir_REJ_10;//using 10%-critical value
        if (tvalue >= asymp_CV(2)) ++asymp_REJ_10;//using 10%-critical value
	}
	empir_REJF(1) = ((double) empir_REJ_5/number_sampl);
	empir_REJF(2) = ((double) empir_REJ_10/number_sampl);
	asymp_REJF(1) = ((double) asymp_REJ_5/number_sampl);
	asymp_REJF(2) = ((double) asymp_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << empir_REJF(1) << " and " << empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << asymp_REJF(1) << " and " << asymp_REJF(2) << endl;
	gsl_rng_free (r);
}

//calculate 5% and 10% empirical and asymptotic rejection frequencies at a cut-off point when X and Y are generated by a univariate non-Gaussian AR(2) process
//and a bivariate non-Gaussian AR(2) process respectively. INPUT: number of random samples generated (number_sampl), a sample size (T), a lag truncation (TL),
//a lag-smoothing parameter (lag_smooth), A 6x1 vector of coefficients for X (alpha_X), a 6x2 matrix of coefficients for Y1 and Y2 (alpha_Y),
//Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed)
//with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, coefficients of correlation (rho12 and rho13), an exponent for the Euclidean distance (expn in (1,2)),
//5% and 10% empirical critical values (empir_CV), 5% and 10% asymptotic critical values (asymp_CV),
//a DGP definition: choose_alt = 0 (independent), = 1 (correlated), = 2 (uncorrelated but dependent), = 3 (correlated and dependent),
// a kernel weight (kernel), an error-generating process (gen_RAN), a data-generating process (gen_DGP), a conditional mean function (cmean),
//and an OLS estimator (est_DGP). OUTPUT: 5% and 10% empirical rejection frequencies (empir_REJF) and asymptotic rejection frequencies (asymp_REJF).
template <double kernel (double), void gen_DGP (Matrix &, Matrix &, Matrix &, const Matrix, const Matrix, const Matrix &, const double, const double, const int, unsigned long),
		  void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
void ML_dcorr_power::power_f (Matrix &empir_REJF, Matrix &asymp_REJF, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                                            const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t,
                                            const Matrix &eta2_s, const Matrix &delta, const double rho12, const double rho13, const double expn, const Matrix &empir_CV, const Matrix &asymp_CV,
                                            const int choose_alt, unsigned long seed, ofstream &pwr_out)  {
	gsl_rng * r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long rseed = 1;
	cout << "Calculating rejection frequencies for T = " << T << endl;
	cout << "5%- and 10%- empirical critical values = " << "(" << empir_CV(1) << " , " << empir_CV(2) << ")" << " ..." << endl;
	cout << "5%- and 10%- asymptotic critical values = " << "(" << asymp_CV(1) << " , " << asymp_CV(2) << ")" << " ..." << endl;
	//calculate integrals of quadratic and quartic functions of the kernel weight
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	NGDist_corr::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);
	cout << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;
	int i = 1, j = 1, size_alpha_X = alpha_X.nRow(), size_alpha_Y = alpha_Y.nRow(), empir_REJ_5 = 0, asymp_REJ_5 = 0, empir_REJ_10 = 0, asymp_REJ_10 = 0;
	auto tvalue = 0.;
	Matrix X(T,1), Y1(T,1), Y2(T,1), alpha_X_hat(size_alpha_X,1), alpha_Y1_hat(size_alpha_Y,1), alpha_Y2_hat(size_alpha_Y,1), alpha_Y_hat(size_alpha_Y,2),
	       resid_X(T-2,1), resid_Y1(T-2,1), resid_Y2(T-2,1);
	#pragma omp parallel for default(shared) reduction (+:empir_REJ_5,empir_REJ_10,asymp_REJ_5,asymp_REJ_10) schedule(dynamic,CHUNK) private(i,j) \
	                                                                                                                      firstprivate(rseed,X,Y1,Y2,resid_X,resid_Y1,resid_Y2,alpha_X_hat,alpha_Y1_hat,alpha_Y2_hat,alpha_Y_hat,tvalue)
	for (i = 1; i <= number_sampl; ++i) {
		rseed = gsl_rng_get (r); //set a random seed for the random number generator
		//draw two samples from the dgp of X and Y each using a random seed
		gen_DGP (X, Y1, Y2, alpha_X, alpha_Y, delta, rho12, rho13, choose_alt, rseed);
		//then use these samples to estimate the AR coefficients
		est_DGP (resid_X, alpha_X_hat, X);
		est_DGP (resid_Y1, alpha_Y1_hat, Y1);
		est_DGP (resid_Y2, alpha_Y2_hat, Y2);
		for (j = 1; j <= size_alpha_Y; j++) {
			alpha_Y_hat(j,1) = alpha_Y1_hat(j); //collect all the OLS estimates in a matrix
			alpha_Y_hat(j,2) = alpha_Y2_hat(j);
		}
		//calculate the test statistic
		tvalue = NGDist_corr::do_Test<kernel, cmean> (X, Y1, Y2, TL, lag_smooth, kernel_QRSum, alpha_X_hat, alpha_Y_hat, epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t, eta2_s, expn);
        pwr_out << tvalue << endl;
        if (tvalue >= empir_CV(1)) ++empir_REJ_5;//using 5%-critical value
        if (tvalue >= asymp_CV(1)) ++asymp_REJ_5;//using 5%-critical value
        if (tvalue >= empir_CV(2)) ++empir_REJ_10;//using 10%-critical value
        if (tvalue >= asymp_CV(2)) ++asymp_REJ_10;//using 10%-critical value
	}
	empir_REJF(1) = ((double) empir_REJ_5/number_sampl);
	empir_REJF(2) = ((double) empir_REJ_10/number_sampl);
	asymp_REJF(1) = ((double) asymp_REJ_5/number_sampl);
	asymp_REJF(2) = ((double) asymp_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << empir_REJF(1) << " and " << empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << asymp_REJF(1) << " and " << asymp_REJF(2) << endl;
	gsl_rng_free (r);
}


//calculate 5% and 10% bootstrap rejection frequencies when X and Y are generated by a univariate non-Gaussian AR(2) process and a bivariate non-Gaussian AR(2) process respectively.
// INPUT: number of random samples generated (number_sampl), a lag truncation (TL), a lag-smoothing parameter (lag_smooth), A 6x1 vector of coefficients for X (alpha_X),
//a 6x2 matrix of coefficients for Y1 and Y2 (alpha_Y), Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by
//gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed) with random seeds (rseed), two BxT matrices of i.i.d. auxiliary random variables for wild bootstrap (xi_x and xi_y),
//a 3x1 vector (delta) in (-1,1)^3, coefficients of correlation (rho12 and rho13), an exponent for the Euclidean distance (expn in (1,2)), a DGP definition: choose_alt = 0 (independent), = 1
// (correlated), = 2 (uncorrelated but dependent), = 3 (correlated and dependent), a kernel weight (kernel), an error-generating process (gen_RAN), a data-generating process (gen_DGP),
//a conditional mean function (cmean), and an OLS estimator (est_DGP).
//OUTPUT: 5% and 10% bootstrap rejection frequencies (bootstrap_REJF)
template <double kernel (double), void gen_DGP (Matrix &, Matrix &, Matrix &, const Matrix, const Matrix, const Matrix &, const double, const double, const int, unsigned long),
		  void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
void ML_dcorr_power::power_f (Matrix &bootstrap_REJF, const int number_sampl, const int TL, const int lag_smooth, const Matrix &alpha_X, const Matrix &alpha_Y,
								    const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t, const Matrix &eta2_s,
								    const Matrix &xi_x, const Matrix &xi_y, const Matrix &delta, const double rho12, const double rho13, const double expn, const int choose_alt,
								    unsigned long seed, ofstream &pwr_out)  {
	gsl_rng * r = nullptr;
	const gsl_rng_type * gen;//random number generator
	gsl_rng_env_setup();
	gen = gsl_rng_taus;
	r = gsl_rng_alloc(gen);
	gsl_rng_set(r, seed);
	unsigned long rseed = 1;
	int num_B = xi_x.nRow(), T = xi_x.nCol();
	cout << "Calculating rejection frequencies for T = " << T << endl;

	int i = 1, j = 1, size_alpha_X = alpha_X.nRow(), size_alpha_Y = alpha_Y.nRow(), bootstrap_REJ_5 = 0, bootstrap_REJ_10 = 0;
	auto pvalue = 0., cstat = 0.;

	Matrix X(T,1), Y1(T,1), Y2(T,1), alpha_X_hat(size_alpha_X,1), alpha_Y1_hat(size_alpha_Y,1), alpha_Y2_hat(size_alpha_Y,1), alpha_Y_hat(size_alpha_Y,2),
	       resid_X(T-2,1), resid_Y1(T-2,1), resid_Y2(T-2,1), cstat_bootstrap(num_B,1);
	pwr_out << " pvalue " << "   ,   " << "statistics" << endl;
	#pragma omp parallel for default(shared) reduction (+:bootstrap_REJ_5,bootstrap_REJ_10) schedule(dynamic,CHUNK) private(i,j) \
	                                                                                                                      firstprivate(rseed,X,Y1,Y2,resid_X,resid_Y1,resid_Y2,alpha_X_hat,alpha_Y1_hat,alpha_Y2_hat,alpha_Y_hat, \
																									        pvalue,cstat,cstat_bootstrap)
	for (i = 1; i <= number_sampl; ++i) {
		rseed = gsl_rng_get (r); //set a random seed for the random number generator
		//draw two samples from the dgp of X and Y each using a random seed
		gen_DGP (X, Y1, Y2, alpha_X, alpha_Y, delta, rho12, rho13, choose_alt, rseed);
		//then use these samples to estimate the AR coefficients
		est_DGP (resid_X, alpha_X_hat, X);
		est_DGP (resid_Y1, alpha_Y1_hat, Y1);
		est_DGP (resid_Y2, alpha_Y2_hat, Y2);
		for (j = 1; j <= size_alpha_Y; j++) {
			alpha_Y_hat(j,1) = alpha_Y1_hat(j); //collect all the OLS estimates in a matrix
			alpha_Y_hat(j,2) = alpha_Y2_hat(j);
		}
		//calculate bootstrap p-values
		pvalue = NGDist_corr::calcul_Pvalue <kernel, cmean> (cstat, cstat_bootstrap, X, Y1, Y2, TL, lag_smooth, alpha_X_hat, alpha_Y_hat, epsilon_t, epsilon_s, eta1_t, eta1_s,
																				    eta2_t, eta2_s, xi_x, xi_y, expn);
		pwr_out <<  pvalue << "   ,   " << cstat << endl;
		if (pvalue <= 0.05)  ++bootstrap_REJ_5;//using 5%-critical value
		if (pvalue <= 0.10)  ++bootstrap_REJ_10;//using 10%-critical value
	}
	bootstrap_REJF(1) = ((double) bootstrap_REJ_5/number_sampl);
	bootstrap_REJF(2) = ((double) bootstrap_REJ_10/number_sampl);

	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% bootstrap reject frequencies for choose_alt = " << choose_alt << " are "
	        << bootstrap_REJF(1) << " and " << bootstrap_REJF(2) << endl;

	gsl_rng_free (r);
}
















#endif
