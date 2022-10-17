#ifndef ML_DCORR_H
#define ML_DCORR_H

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <asserts.h>
#include <kernel.h>
#include <ML_reg_dcorr_ver1.h>
#include <nl_dgp.h>


#define CHUNK 1

using namespace std;

class ML_DCORR {
	public:
		ML_DCORR(){  }; //default constructor
		~ML_DCORR () { };//default destructor

		/* Calculate the test statistics by ML methods
		INPUT: a T by N1 matrix (X) and a T by N2 matrix (Y), a maximum number of truncation lags (L), a lag-smoothing bandwidth (lag_smooth), \
		a distance correlation exponent (expn) */
		template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																		const SGVector<double> &, /*train labels*/
																		int, /*number of subsets for TSCV*/
																		int, /*minimum subset size for TSCV*/
																		SGVector<int>, /*list of tree max depths (for GBM)*/
																		SGVector<int>, /*list of numbers of iterations (for GBM)*/
																		SGVector<double>, /*list of learning rates (for GBM)*/
																		SGVector<double>, /*list of subset fractions (for GBM)*/
																		SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																		SGVector<int>, /*list of number of bags (for RF)*/
																		int	/*seed for random number generator*/),
				SGVector<double> ML_reg(const SGMatrix<double> &,
										const SGVector<double> &,
										int, /*tree max depth*/
										int, /*number of iterations*/
										double, /*learning rate*/
										double, /*subset fraction*/
										int, /*number of random features used for bagging (for RF)*/
										int, /*number of bags (for RF)*/
										int	/*seed for the random number generator*/),
				double kernel_k (double) /*kernel to smooth lags*/>
		static double do_Test(	const SGMatrix<double> &X, const SGMatrix<double> &Y, /*T by 2 matrices of observations*/
								int L, /*maximum truncation lag*/
								int lag_smooth, /*kernel bandwidth*/
								double expn, /*exponent of the Euclidean distance*/
								int num_subsets, /*number of subsets for TSCV*/
								int min_subset_size, /*minimum subset size for TSCV*/
								SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
								SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
								SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
								SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
								SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
								double kernel_QDSum, double kernel_QRSum, /*quadratically and quartically integrate the kernel function*/
								int seed /*seed for random number generator*/);

		/* Calculate the test statistics by ML methods with transformed data.
		INPUT: a T by N1 matrix (X) and a T by N2 matrix (Y), a maximum number of truncation lags (L), a lag-smoothing bandwidth (lag_smooth), \
		a distance correlation exponent (expn) */
		template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																		const SGVector<double> &, /*train labels*/
																		int, /*number of subsets for TSCV*/
																		int, /*minimum subset size for TSCV*/
																		SGVector<int>, /*list of tree max depths (for GBM)*/
																		SGVector<int>, /*list of numbers of iterations (for GBM)*/
																		SGVector<double>, /*list of learning rates (for GBM)*/
																		SGVector<double>, /*list of subset fractions (for GBM)*/
																		SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																		SGVector<int>, /*list of number of bags (for RF)*/
																		int /*seed for random number generator*/),
				SGVector<double> ML_reg(const SGMatrix<double> &,
										const SGVector<double> &,
										int, /*tree max depth*/
										int, /*number of iterations*/
										double, /*learning rate*/
										double, /*subset fraction*/
										int, /*number of random features used for bagging (for RF)*/
										int, /*number of bags (for RF)*/
										int /*seed for the random number generator*/),
				double kernel_k (double) /*kernel to smooth lags*/>
		static double do_Test(	const SGMatrix<double> &X, const SGMatrix<double> &Y,
								int L, /*maximum truncation lag*/
								int lag_smooth, /*kernel bandwidth*/
								double expn, /*exponent of distances*/
								double expn_x, /*exponent of data*/
								int num_subsets, /*number of subsets for TSCV*/
								int min_subset_size, /*minimum subset size for TSCV*/
								SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
								SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
								SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
								SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
								SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
								double kernel_QDSum, double kernel_QRSum, /*quadratically and quartically integrate the kernel function*/
								int seed /*seed for random number generator*/ );

		/* Calculate the test statistics by moving averages.
		INPUT: a T by N1 matrix (X) and a T by N2 matrix (Y), a maximum number of truncation lags (L), a lag-smoothing bandwidth (lag_smooth), \
		a distance correlation exponent (expn) */
		template< double kernel_k (double) /*kernel to smooth lags*/>
		static double do_Test(	const SGMatrix<double> &X, const SGMatrix<double> &Y, /*T by 2 matrices of observations*/
								int L, /*maximum truncation lag*/
								int lag_smooth, /*kernel bandwidth*/
								double expn, /*exponent of the Euclidean distance*/
								int num_subsets, /*number of subsets for TSCV*/
								int min_subset_size, /*minimum subset size for TSCV*/
								SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
								SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
								SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
								SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
								SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
								SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
								double kernel_QDSum, double kernel_QRSum, /*quadratically and quartically integrate the kernel function*/
								int seed /*seed for random number generator*/);

		/* Calculate the bootstrap test statistics by ML methods.
		OUTPUT: a bootstrap p-value (pvalue), a value of the statistic (cstat), and a vector of bootstrap statistics (cstat_bootstrap) */
		template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																		const SGVector<double> &, /*train labels*/
																		int, /*number of subsets for TSCV*/
																		int, /*minimum subset size for TSCV*/
																		SGVector<int>, /*list of tree max depths (for GBM)*/
																		SGVector<int>, /*list of numbers of iterations (for GBM)*/
																		SGVector<double>, /*list of learning rates (for GBM)*/
																		SGVector<double>, /*list of subset fractions (for GBM)*/
																		SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																		SGVector<int>, /*list of number of bags (for RF)*/
																		int /*seed for random number generator*/),
				SGVector<double> ML_reg(const SGMatrix<double> &,
										const SGVector<double> &,
										int, /*tree max depth*/
										int, /*number of iterations*/
										double, /*learning rate*/
										double, /*subset fraction*/
										int, /*number of random features used for bagging (for RF)*/
										int, /*number of bags (for RF)*/
										int /*seed for the random number generator*/),
				double kernel_k(double) /*kernel to smooth lags*/>
		static std::tuple<	double /*p-value*/,
							double /*statistic*/,
							SGVector<double> /*bootstrap statistics*/> do_Test_bt(	const SGMatrix<double> &X, const SGMatrix<double> &Y, /*T by 2 matrices of observations*/
																					int L, /*maximum truncation lag*/
																					int lag_smooth, /*kernel bandwidth*/
																					double expn, /*exponent of the Euclidean distance*/
																					const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices
																																				of auxiliary random variables
																																				used for bootstrapping*/
																					int num_subsets, /*number of subsets for TSCV*/
																					int min_subset_size, /*minimum subset size for TSCV*/
																					SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
																					SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
																					SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
																					SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
																					SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
																					SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
																					double kernel_QDSum, double kernel_QRSum, /*quadratically and quartically integrate the kernel function*/
																					int seed /*seed for random number generator*/);
	//private:
		//integrate quadratic and quartic functions of a kernel weight
        template <double kernel_k (double )>
        static void integrate_Kernel (double &kernel_QDSum, double &kernel_QRSum);
};

//integrate quadratic and quartic functions of a kernel weight
template <double kernel_k (double )>
void ML_DCORR::integrate_Kernel (double &kernel_QDSum, double &kernel_QRSum) {
	double x = 0.;
	kernel_QDSum = 0.;
	kernel_QRSum = 0.;
	int t = 1, N = 1000000;
	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, 3525364362);
	while (t <= N) {
		x =  10 * gsl_rng_uniform(r) - 5; //integral over the range [-5, 5]
		kernel_QDSum += 10 * ((double) 1/N) * pow(kernel_k (x), 2.);
		kernel_QRSum += 10 * ((double) 1/N) * pow(kernel_k (x), 4.);
		t++;
	}
	gsl_rng_free (r);
}



/* Calculate the test statistics by ML methods
INPUT: a T by N1 matrix (X) and a T by N2 matrix (Y), a maximum number of truncation lags (L), a lag-smoothing bandwidth (lag_smooth), \
a distance correlation exponent (expn) */
template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																const SGVector<double> &, /*train labels*/
																int, /*number of subsets for TSCV*/
																int, /*minimum subset size for TSCV*/
																SGVector<int>, /*list of tree max depths (for GBM)*/
																SGVector<int>, /*list of numbers of iterations (for GBM)*/
																SGVector<double>, /*list of learning rates (for GBM)*/
																SGVector<double>, /*list of subset fractions (for GBM)*/
																SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																SGVector<int>, /*list of number of bags (for RF)*/
																int /*seed for random number generator*/),
		SGVector<double> ML_reg(const SGMatrix<double> &,
								const SGVector<double> &,
								int, /*tree max depth*/
								int, /*number of iterations*/
								double, /*learning rate*/
								double, /*subset fraction*/
								int, /*number of random features used for bagging (for RF)*/
								int, /*number of bags (for RF)*/
								int  /*seed for the random number generator*/),
		double kernel_k (double) /*kernel to smooth lags*/>
double ML_DCORR::do_Test(	const SGMatrix<double> &X, const SGMatrix<double> &Y, /*T by 2 matrices of observations*/
							int L, /*maximum truncation lag*/
							int lag_smooth, /*kernel bandwidth*/
							double expn, /*exponent of the Euclidean distance*/
							int num_subsets, /*number of subsets for TSCV*/
							int min_subset_size, /*minimum subset size for TSCV*/
							SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
							SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
							SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
							SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
							SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
							SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
							double kernel_QDSum, double kernel_QRSum, /*quadratically and quartically integrate the kernel function*/
							int seed /*seed for random number generator*/ ) {
	int T = X.num_rows;
	int t = 1, s = 1, tau = 1, lag = 0;
	ASSERT_(T == Y.num_rows && X.num_cols == 2);
	int T0 = T-L;

	SGMatrix<double> var_U_x(T0, T0), var_U_y(T0, T0);
	//#pragma omp parallel sections num_threads(2)
	{
		//calculate \H{U}_{t,s}^{(x)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
//		//#pragma omp section
		var_U_x = ML_REG_DCORR::var_U<ML_cv, ML_reg>(	X, L, expn,
														num_subsets, /*number of subsets for TSCV*/
														min_subset_size, /*minimum subset size for TSCV*/
														tree_max_depths_list, /*list of tree max depths (for GBM)*/
														num_iters_list, /*list of numbers of iterations (for GBM)*/
														learning_rates_list, /*list of learning rates (for GBM)*/
														subset_fractions_list, /*list of subset fractions (for GBM)*/
														num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
														num_bags_list, /*list of number of bags (for RF)*/
														seed);

		//calculate \H{U}_{t,s}^{(y)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
		//#pragma omp section
		var_U_y = ML_REG_DCORR::var_U<ML_cv, ML_reg>(	Y, L, expn,
														num_subsets, /*number of subsets for TSCV*/
														min_subset_size, /*minimum subset size for TSCV*/
														tree_max_depths_list, /*list of tree max depths (for GBM)*/
														num_iters_list, /*list of numbers of iterations (for GBM)*/
														learning_rates_list, /*list of learning rates (for GBM)*/
														subset_fractions_list, /*list of subset fractions (for GBM)*/
														num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
														num_bags_list, /*list of number of bags (for RF)*/
														seed);
	}

	double aVar = 0.;
	for (s = 0; s < T0; ++s) {
		for (t = s+1; t < T0; ++t) {
			aVar += ( (double) 1/pow(T0, 2.) ) * pow(var_U_x(t, s)*var_U_y(t, s), 2.);
		}
	}


	double weight = 0., sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0., cov = 0., sum = 0.;
	//prod_Var = var(X, TL, alpha(1), beta(1), lambda(1), sigma(1)) * var(Y, TL, alpha(2), beta(2), lambda(2), sigma(2));//calculate product of distance variances
	//cout << "product of variances = " << prod_Var << endl;

	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) firstprivate(weight,sum1,sum2,sum3,sum4,cov) private(lag,t,s,tau)
	for (lag = 1-T0; lag <= T0-1; lag++) {
        #pragma omp critical
        {
            if (lag == 0) weight = 1.;
            else weight = kernel_k ((double) lag/lag_smooth);
        }

        if ( (weight > 1e-5) || (weight < -1e-5) ) {
			//#pragma omp critical
			{
				if (lag >= 0) {
					sum1 = 0.;
					sum2 = 0.;
					sum3 = 0.;
					sum4 = 0.;
					for (t = lag; t < T0; ++t) {
						for (s = lag; s < T0; ++s) {
							if (t != s) {
								sum1 += var_U_x(t, s) * var_U_y(t-lag, s-lag);
							}
							sum3 += var_U_x(t, s);
							sum4 += var_U_y(t-lag, s-lag) ;
							for (tau = lag; tau < T0; ++tau) {
								sum2 += 2 * var_U_x(t, s) * var_U_y(tau-lag, s-lag);
							}
						}
					}
					cov = ((double) 1/pow(T0-lag, 2.)) * sum1 - ((double) 1/pow(T0-lag, 3.)) * sum2 + ((double) 1/pow(T0-lag, 4.)) * sum3 * sum4;
				}
				else {
					sum1 = 0.;
					sum2 = 0.;
					sum3 = 0.;
					sum4 = 0.;
					for (t = -lag; t < T0; ++t) {
						for (s = -lag; s < T0; ++s) {
							if (t != s) {
								sum1 += var_U_x(t+lag, s+lag) * var_U_y(t, s);
							}
							sum3 += var_U_x(t+lag, s+lag);
							sum4 += var_U_y(t, s);
							for (tau = -lag; tau < T0; ++tau) {
								sum2 += 2 * var_U_x(t+lag, s+lag) * var_U_y(tau, s);
							}
						}
					}
					cov = ((double) 1/pow(T0+lag, 2.)) * sum1 - ((double) 1/pow(T0+lag, 3.)) * sum2 + ((double) 1/pow(T0+lag, 4.)) * sum3 * sum4;
				}
			}
		    sum += T0 * pow(weight, 2.) * cov;
            //cout << "counting lags: " << lag << endl;
        }
	}

	/* Comment out the following two lines */
	//auto kernel_QDSum = 0., kernel_QRSum = 0.;
	//ML_DCORR::integrate_Kernel <kernel_k> (kernel_QDSum, kernel_QRSum);

	return sum / ( 2 * sqrt(lag_smooth * kernel_QRSum * aVar) );
}

/* Calculate the test statistics by ML methods with transformed data.
INPUT: a T by N1 matrix (X) and a T by N2 matrix (Y), a maximum number of truncation lags (L), a lag-smoothing bandwidth (lag_smooth), \
a distance correlation exponent (expn) */
template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																const SGVector<double> &, /*train labels*/
																int, /*number of subsets for TSCV*/
																int, /*minimum subset size for TSCV*/
																SGVector<int>, /*list of tree max depths (for GBM)*/
																SGVector<int>, /*list of numbers of iterations (for GBM)*/
																SGVector<double>, /*list of learning rates (for GBM)*/
																SGVector<double>, /*list of subset fractions (for GBM)*/
																SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																SGVector<int>, /*list of number of bags (for RF)*/
																int /*seed for random number generator*/),
		SGVector<double> ML_reg(const SGMatrix<double> &,
								const SGVector<double> &,
								int, /*tree max depth*/
								int, /*number of iterations*/
								double, /*learning rate*/
								double, /*subset fraction*/
								int, /*number of random features used for bagging (for RF)*/
								int, /*number of bags (for RF)*/
								int /*seed for the random number generator*/),
		double kernel_k (double) /*kernel to smooth lags*/>
double ML_DCORR::do_Test(	const SGMatrix<double> &X, const SGMatrix<double> &Y,
							int L, /*maximum truncation lag*/
							int lag_smooth, /*kernel bandwidth*/
							double expn, /*exponent of distances*/
							double expn_x, /*exponent of data*/
							int num_subsets, /*number of subsets for TSCV*/
							int min_subset_size, /*minimum subset size for TSCV*/
							SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
							SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
							SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
							SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
							SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
							SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
							double kernel_QDSum, double kernel_QRSum, /*quadratically and quartically integrate the kernel function*/
							int seed /*seed for random number generator*/ ) {
	int T = X.num_rows;
	int t = 1, s = 1, tau = 1, lag = 0;
	ASSERT_(T == Y.num_rows);
	int T0 = T-L;


	SGMatrix<double> var_U_x(T0, T0), var_U_y(T0, T0);
	//#pragma omp parallel sections num_threads(2)
	{
		//calculate \H{U}_{t,s}^{(x)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
//		//#pragma omp section
		var_U_x = ML_REG_DCORR::var_U<ML_cv, ML_reg>(	X, L, expn, expn_x,
														num_subsets, /*number of subsets for TSCV*/
														min_subset_size, /*minimum subset size for TSCV*/
														tree_max_depths_list, /*list of tree max depths (for GBM)*/
														num_iters_list, /*list of numbers of iterations (for GBM)*/
														learning_rates_list, /*list of learning rates (for GBM)*/
														subset_fractions_list, /*list of subset fractions (for GBM)*/
														num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
														num_bags_list, /*list of number of bags (for RF)*/
														seed);

		//calculate \H{U}_{t,s}^{(y)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
		//#pragma omp section
		var_U_y = ML_REG_DCORR::var_U<ML_cv, ML_reg>(	Y, L, expn, expn_x,
														num_subsets, /*number of subsets for TSCV*/
														min_subset_size, /*minimum subset size for TSCV*/
														tree_max_depths_list, /*list of tree max depths (for GBM)*/
														num_iters_list, /*list of numbers of iterations (for GBM)*/
														learning_rates_list, /*list of learning rates (for GBM)*/
														subset_fractions_list, /*list of subset fractions (for GBM)*/
														num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
														num_bags_list, /*list of number of bags (for RF)*/
														seed);
	}

	double aVar = 0.;
	for (s = 0; s < T0; ++s) {
		for (t = s+1; t < T0; ++t) {
			aVar += ((double) 1/pow(T0, 2.)) * pow(var_U_x(t, s)*var_U_y(t, s), 2.);
		}
	}


	double weight = 0., sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0., cov = 0., sum = 0.;
	//prod_Var = var(X, TL, alpha(1), beta(1), lambda(1), sigma(1)) * var(Y, TL, alpha(2), beta(2), lambda(2), sigma(2));//calculate product of distance variances
	//cout << "product of variances = " << prod_Var << endl;
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) firstprivate(weight,sum1,sum2,sum3,sum4,cov) private(lag,t,s,tau)
	for (lag = 1-T0; lag <= T0-1; lag++) {
		#pragma omp critical
		{
			if (lag == 0) weight = 1.;
			else weight = kernel_k ((double) lag/lag_smooth);
		}
        if ( (weight > 1e-5) || (weight < -1e-5) ) {
        	if (lag >= 0) {
        	    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
        	    for (t = lag; t < T0; ++t) {
			        for (s = lag; s < T0; ++s) {
			        	if (t != s) {
				            sum1 += var_U_x(t, s) * var_U_y(t-lag, s-lag);
				        }
				        sum3 += var_U_x(t, s);
				        sum4 += var_U_y(t-lag, s-lag) ;
				        for (tau = lag; tau < T0; ++tau) {
					        sum2 += 2 * var_U_x(t, s) * var_U_y(tau-lag, s-lag);
				        }
				    }
			    }
			    cov = ((double) 1/pow(T0-lag, 2.)) * sum1 - ((double) 1/pow(T0-lag, 3.)) * sum2 + ((double) 1/pow(T0-lag, 4.)) * sum3 * sum4;
		    }
		    else {
				sum1 = 0.;
				sum2 = 0.;
				sum3 = 0.;
				sum4 = 0.;
			    for (t = -lag; t < T0; ++t) {
			        for (s = -lag; s < T0; ++s) {
			        	if (t != s) {
				            sum1 += var_U_x(t+lag, s+lag) * var_U_y(t, s);
				        }
				        sum3 += var_U_x(t+lag, s+lag);
				        sum4 += var_U_y(t, s);
				        for (tau = -lag; tau < T0; ++tau) {
							sum2 += 2 * var_U_x(t+lag, s+lag) * var_U_y(t, tau);
				        }
			        }
		        }
			    cov = ((double) 1/pow(T0+lag, 2.)) * sum1 - ((double) 1/pow(T0+lag, 3.)) * sum2 + ((double) 1/pow(T0+lag, 4.)) * sum3 * sum4;
		    }
		    sum += T0 * pow(weight, 2.) * cov;
            //cout << "counting lags: " << lag << endl;
        }
	}

	/* Comment out the following two lines */
	//auto kernel_QDSum = 0., kernel_QRSum = 0.;
	//ML_DCORR::integrate_Kernel <kernel_k> (kernel_QDSum, kernel_QRSum);

	return sum / ( 2 * sqrt(lag_smooth * kernel_QRSum * aVar) );
}



/* Calculate the test statistics by moving averages.
INPUT: a T by N1 matrix (X) and a T by N2 matrix (Y), a maximum number of truncation lags (L), a lag-smoothing bandwidth (lag_smooth), \
a distance correlation exponent (expn) */
template< double kernel_k (double) /*kernel to smooth lags*/>
double ML_DCORR::do_Test(	const SGMatrix<double> &X, const SGMatrix<double> &Y, int L, int lag_smooth, double expn,
							int num_subsets, /*number of subsets for TSCV*/
							int min_subset_size, /*minimum subset size for TSCV*/
							SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
							SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
							SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
							SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
							SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
							SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
							double kernel_QDSum, double kernel_QRSum, /*quadratically and quartically integrate the kernel function*/
							int seed /*seed for random number generator*/ ) {

	(void) num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed;

	int T = X.num_rows;
	int t = 1, s = 1, tau = 1, lag = 0;
	ASSERT_(T == Y.num_rows);
	int T0 = T-L;

	SGMatrix<double> var_U_x(T0, T0), var_U_y(T0, T0);
	#pragma omp parallel sections num_threads(2)
	{
		//calculate \H{U}_{t,s}^{(x)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
		#pragma omp section
		var_U_x = ML_REG_DCORR::var_U(X, L, expn);

		//calculate \H{U}_{t,s}^{(y)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
		#pragma omp section
		var_U_y = ML_REG_DCORR::var_U(Y, L, expn);
	}

	/* calculate the asymptotic variance */
	double aVar = 0.;
	for (s = 0; s < T0; ++s) {
		for (t = s+1; t < T0; ++t) {
			aVar += ((double) 1/pow(T0, 2.)) * pow(var_U_x(t, s)*var_U_y(t, s), 2.);
		}
	}


	double weight = 0., sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0., cov = 0., sum = 0.;

	//prod_Var = var(X, TL, alpha(1), beta(1), lambda(1), sigma(1)) * var(Y, TL, alpha(2), beta(2), lambda(2), sigma(2));//calculate product of distance variances
	//cout << "product of variances = " << prod_Var << endl;
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) firstprivate(weight,sum1,sum2,sum3,sum4,cov) private(lag,t,s,tau)
	for (lag = 1-T0; lag <= T0-1; lag++) {
		#pragma omp critical
		{
			if (lag == 0) weight = 1.;
			else weight = kernel_k ((double) lag/lag_smooth);
		}
        if ( (weight > 1e-5) || (weight < -1e-5) ) {
        	if (lag >= 0) {
        	    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
        	    for (t = lag; t < T0; ++t) {
			        for (s = lag; s < T0; ++s) {
			        	if (t != s) {
				            sum1 += var_U_x(t, s) * var_U_y(t-lag, s-lag);
				        }
				        sum3 += var_U_x(t, s);
				        sum4 += var_U_y(t-lag, s-lag) ;
				        for (tau = lag; tau < T0; ++tau) {
					        sum2 += 2 * var_U_x(t, s) * var_U_y(tau-lag, s-lag);
				        }
				    }
			    }
			    cov = ((double) 1/pow(T0-lag, 2.)) * sum1 - ((double) 1/pow(T0-lag, 3.)) * sum2 + ((double) 1/pow(T0-lag, 4.)) * sum3 * sum4;
		    }
		    else {
				sum1 = 0.;
				sum2 = 0.;
				sum3 = 0.;
				sum4 = 0.;
			    for (t = -lag; t < T0; ++t) {
			        for (s = -lag; s < T0; ++s) {
			        	if (t != s) {
				            sum1 += var_U_x(t+lag, s+lag) * var_U_y(t, s);
				        }
				        sum3 += var_U_x(t+lag, s+lag);
				        sum4 += var_U_y(t, s);
				        for (tau = -lag; tau < T0; ++tau) {
							sum2 += 2 * var_U_x(t+lag, s+lag) * var_U_y(t, tau);
				        }
			        }
		        }
			    cov = ((double) 1/pow(T0+lag, 2.)) * sum1 - ((double) 1/pow(T0+lag, 3.)) * sum2 + ((double) 1/pow(T0+lag, 4.)) * sum3 * sum4;
		    }
		    sum += T0 * pow(weight, 2.) * cov;
            //cout << "counting lags: " << lag << endl;
        }
	}

	/* Comment out the following two lines */
	//auto kernel_QDSum = 0., kernel_QRSum = 0.;
	//ML_DCORR::integrate_Kernel <kernel_k> (kernel_QDSum, kernel_QRSum);

	return sum / ( 2 * sqrt(lag_smooth * kernel_QRSum * aVar) );
}

//#if 0
/* Calculate the bootstrap test statistics by ML methods.
OUTPUT: a bootstrap p-value (pvalue), a value of the statistic (cstat), and a vector of bootstrap statistics (cstat_bootstrap) */
template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																const SGVector<double> &, /*train labels*/
																int, /*number of subsets for TSCV*/
																int, /*minimum subset size for TSCV*/
																SGVector<int>, /*list of tree max depths (for GBM)*/
																SGVector<int>, /*list of numbers of iterations (for GBM)*/
																SGVector<double>, /*list of learning rates (for GBM)*/
																SGVector<double>, /*list of subset fractions (for GBM)*/
																SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																SGVector<int>, /*list of number of bags (for RF)*/
																int /*seed for random number generator*/),
		SGVector<double> ML_reg(const SGMatrix<double> &,
								const SGVector<double> &,
								int, /*tree max depth*/
								int, /*number of iterations*/
								double, /*learning rate*/
								double, /*subset fraction*/
								int, /*number of random features used for bagging (for RF)*/
								int, /*number of bags (for RF)*/
								int /*seed for the random number generator*/),
		double kernel_k(double) /*kernel to smooth lags*/>
std::tuple<	double /*p-value*/,
			double /*statistic*/,
			SGVector<double> /*bootstrap statistics*/> ML_DCORR::do_Test_bt(const SGMatrix<double> &X, const SGMatrix<double> &Y, /*T by 2 matrices of observations*/
																			int L, /*maximum truncation lag*/
																			int lag_smooth, /*kernel bandwidth*/
																			double expn, /*exponent of the Euclidean distance*/
																			const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices
																																		of auxiliary random variables
																																		used for bootstrapping*/
																			int num_subsets, /*number of subsets for TSCV*/
																			int min_subset_size, /*minimum subset size for TSCV*/
																			SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
																			SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
																			SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
																			SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
																			SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
																			double kernel_QDSum, double kernel_QRSum, /*quadratically and quartically integrate the kernel function*/
																			int seed /*seed for random number generator*/ ) {
	int T = X.num_rows, num_B = xi_x.num_rows;
	int t = 1, s = 1, tau = 1, lag = 0, i = 0;

    ASSERT_(num_B == xi_y.num_rows && T == Y.num_rows && T == xi_x.num_cols && T == xi_y.num_cols); //check for the size consistency of matrices

	int T0 = T-L;

	SGMatrix<double> var_U_x(T0,T0), var_U_y(T0,T0);

	//#pragma omp parallel sections
	{
		//calculate \H{U}_{t,s}^{(x)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
		//#pragma omp section
		var_U_x = ML_REG_DCORR::var_U<ML_cv, ML_reg>(	X, L, expn,
														num_subsets, /*number of subsets for TSCV*/
														min_subset_size, /*minimum subset size for TSCV*/
														tree_max_depths_list, /*list of tree max depths (for GBM)*/
														num_iters_list, /*list of numbers of iterations (for GBM)*/
														learning_rates_list, /*list of learning rates (for GBM)*/
														subset_fractions_list, /*list of subset fractions (for GBM)*/
														num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
														num_bags_list, /*list of number of bags (for RF)*/
														seed);

		//calculate \H{U}_{t,s}^{(y)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
		//#pragma omp section
		var_U_y = ML_REG_DCORR::var_U<ML_cv, ML_reg>(	Y, L, expn,
														num_subsets, /*number of subsets for TSCV*/
														min_subset_size, /*minimum subset size for TSCV*/
														tree_max_depths_list, /*list of tree max depths (for GBM)*/
														num_iters_list, /*list of numbers of iterations (for GBM)*/
														learning_rates_list, /*list of learning rates (for GBM)*/
														subset_fractions_list, /*list of subset fractions (for GBM)*/
														num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
														num_bags_list, /*list of number of bags (for RF)*/
														seed);
	}

	/* calculate the asymptotic variance */
	double aVar = 0.;
	for (s = 0; s < T0; ++s) {
		for (t = s+1; t < T0; ++t) {
			aVar += ((double) 1/pow(T0, 2.)) * pow(var_U_x(t, s)*var_U_y(t, s), 2.);
		}
	}


	double weight = 0., sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0., cov = 0., sum = 0.;

	/* calculate the centered quantity */
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) firstprivate(weight,sum1,sum2,sum3,sum4,cov) private(lag,t,s,tau)
	for (lag = 1-T0; lag <= T0-1; lag++) {
		if (lag == 0) weight = 1.;
		else weight = kernel_k ((double) lag/lag_smooth);

        if ( (weight > 1e-5) || (weight < -1e-5) ) {
        	if (lag >= 0) {
        	    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
        	    for (t = lag; t < T0; ++t) {
			        for (s = lag; s < T0; ++s) {
			        	if (t != s) {
				            sum1 += var_U_x(t, s) * var_U_y(t-lag, s-lag);
				        }
				        sum3 += var_U_x(t, s);
				        sum4 += var_U_y(t-lag, s-lag) ;
				        for (tau = lag; tau < T0; ++tau) {
					        sum2 += 2 * var_U_x(t, s) * var_U_y(tau-lag, s-lag);
				        }
				    }
			    }
			    cov = ((double) 1/pow(T0-lag, 2.)) * sum1 - ((double) 1/pow(T0-lag, 3.)) * sum2 + ((double) 1/pow(T0-lag, 4.)) * sum3 * sum4;
		    }
		    else {
				sum1 = 0.;
				sum2 = 0.;
				sum3 = 0.;
				sum4 = 0.;
			    for (t = -lag; t < T0; ++t) {
			        for (s = -lag; s < T0; ++s) {
			        	if (t != s) {
							sum1 += var_U_x(t+lag, s+lag) * var_U_y(t, s);
				        }
				        sum3 += var_U_x(t+lag, s+lag);
				        sum4 += var_U_y(t, s);
				        for (tau = -lag; tau < T0; ++tau) {
					        sum2 += 2 * var_U_x(t+lag, s+lag) * var_U_y(t, tau);
				        }
			        }
		        }
				cov = ((double) 1/pow(T0+lag, 2.)) * sum1 - ((double) 1/pow(T0+lag, 3.)) * sum2 + ((double) 1/pow(T0+lag, 4.)) * sum3 * sum4;
		    }
		    sum += T0 * pow(weight, 2.) * cov;
            //cout << "counting lags: " << lag << endl;
        }
	}

	double cstat =  ( (double) sum / sqrt(lag_smooth) ); //compute the centered statistic
	double tstat = sum / ( 2 * sqrt(lag_smooth * kernel_QRSum * aVar) ); //compute the t-statistic
	//cout << "the value of the statistics = " << cstat << endl;

	SGVector<double> cstat_bootstrap(num_B);

	/* calculate bootstrap centered quantities */
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) firstprivate(weight,sum1,sum2,sum3,sum4,cov) private(i,lag,t,s,tau)
	for (i = 0; i < num_B; i++) {
		cov = 0.; //reset the value of 'cov' and 'sum' to zeros in each bootstrap iteration
		sum = 0.;
		for (lag = 1-T0; lag <= T0-1; lag++) {
			if (lag == 0) weight = 1.;
			else weight = kernel_k ((double) lag/lag_smooth);

			if ( (weight > 1e-5) || (weight < -1e-5) ) {
				if (lag >= 0) {
					sum1 = 0.;
					sum2 = 0.;
					sum3 = 0.;
					sum4 = 0.;
					for (t = lag; t < T0; ++t) {
						for (s = lag; s < T0; ++s) {
							if (t != s) {
								sum1 += xi_x(i, t) * xi_x(i, s) * var_U_x(t, s) * var_U_y(t-lag, s-lag) * xi_y(i, t-lag) * xi_y(i, s-lag);
							}
							sum3 += xi_x(i, t) * xi_x(i, s) * var_U_x(t, s);
							sum4 += xi_y(i, t-lag) * xi_y(i, s-lag) * var_U_y(t-lag, s-lag) ;
							for (tau = lag; tau < T0; ++tau) {
								sum2 += 2 * xi_x(i, t) * xi_x(i, s) * var_U_x(t, s) * var_U_y(tau-lag, s-lag) * xi_y(i, tau-lag) * xi_y(i, s-lag);
							}
						}
					}
					cov = ((double) 1/pow(T0-lag, 2.)) * sum1 - ((double) 1/pow(T0-lag, 3.)) * sum2 + ((double) 1/pow(T0-lag, 4.)) * sum3 * sum4;
				}
				else {
					sum1 = 0.;
					sum2 = 0.;
					sum3 = 0.;
					sum4 = 0.;
					for (t = -lag; t < T0; ++t) {
						for (s = -lag; s < T0; ++s) {
							if (t != s) {
								sum1 += xi_x(i, t+lag) * xi_x(i, s+lag) * var_U_x(t+lag, s+lag) * var_U_y(t, s) * xi_y(i, t) * xi_y(i, s);
							}
							sum3 += xi_x(i, t+lag) * xi_x(i, s+lag) * var_U_x(t+lag, s+lag);
							sum4 += xi_y(i, t) * xi_y(i, s) * var_U_y(t, s);
							for (tau = -lag; tau < T0; ++tau) {
								sum2 += 2 * xi_x(i, t+lag) * xi_x(i, s+lag) * var_U_x(t+lag, s+lag) * var_U_y(t, tau) * xi_y(i, t) * xi_y(i, tau);
							}
						}
					}
					cov = ((double) 1/pow(T0+lag, 2.)) * sum1 - ((double) 1/pow(T0+lag, 3.)) * sum2 + ((double) 1/pow(T0+lag, 4.)) * sum3 * sum4;
				}
				sum += T0 * pow(weight, 2.) * cov;
				//cout << "counting lags: " << lag << endl;
			}
		}
		cstat_bootstrap[i] =  ( (double) sum / sqrt(lag_smooth) );
	}

	int count1 = 0;
	for (i = 0; i < num_B; ++i) {
		if (cstat_bootstrap[i] >= cstat) count1++;
	}

	double pvalue = ( (double) count1 / num_B );

	return {pvalue, tstat, cstat_bootstrap};
}

//#endif

















#endif
