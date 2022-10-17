#ifndef HSIC_TEST_H
#define HSIC_TEST_H

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <asserts.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/util/iterators.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <nl_dgp.h>
#include <mgarch.h>


#define CHUNK 1

using namespace std;
using namespace shogun;
using namespace shogun::linalg;

class HSIC {
	public:
		HSIC(){  }; //default constructor
		~HSIC () { };//default destructor

		/* Calculate the HSIC-based test statistics for residuals.
		INPUT: T by dim matrices of residuals (eta1 and eta2). */
		template<double kernel1(SGVector<double>, SGVector<double>), double kernel2(SGVector<double>, SGVector<double>)>
		static std::pair<double, double> J(	const SGMatrix<double> &eta1, /*T by dim1 matrix*/
											const SGMatrix<double> &eta2, /*T by dim2 matrix*/
											int M /*maximum number of lags*/);

		/* Calculate the bootstrap p-values of the HSIC-based tests with residuals from multivariate processes.
		INPUT: two T by 2 matrices of data */
		template<	double kernel1(SGVector<double>, SGVector<double>), /*a kernel function for the first process*/
					double kernel2(SGVector<double>, SGVector<double>), /*a kernel function for the second process*/
					/*an estimator*/
					double estimator(	SGMatrix<double> &, /*T by 2 matrix of residuals*/
										SGVector<double> &, /*a vector of estimates*/
										const SGMatrix<double> &, /*T by 2 matrix of data*/
										SGVector<double>    /*a vector of initial parameters*/),
					/*generate data from residuals*/
					SGMatrix<double> gen_data(	const SGMatrix<double> &, /*T by 2 matrix of innovations*/
												const SGVector<double>    /*a vector of parameters*/),
					/*compute residuals*/
					SGMatrix<double> resid(	const SGMatrix<double> &, /*T by 2 matrix of observations*/
											SGVector<double> 		 /*a vector of parameters*/),
					/*compute the gradient vector*/
					SGVector<double> gradient(	const SGVector<double> &, /*a vector of parameters*/
												const SGMatrix<double> &, /*a T by 2 matrix of data*/
												double /*finite differential level*/),
					/*compute the Hessian matrix*/
					SGMatrix<double> hessian(	const SGVector<double> &, /*a vector of parameters*/
												const SGMatrix<double> &, /*a T by 2 matrix of data*/
												double  /*a finite differential level*/) >
		static std::pair<double, double> resid_bootstrap_pvalue(const SGMatrix<double> &Y1, /*T by 2 matrix of data*/
																const SGMatrix<double> &Y2,
																const SGVector<double> theta01, /*a vector of starting values used to estimate the DGP*/
																const SGVector<double> theta02,
																int M, /*maximum number of lags*/
																int num_bts, /*number of bootstrap samples*/
																double h, /*a finite differential factor*/
																unsigned long seed);

		/* Define the Gaussian kernel with unit variance */
		static double kernel(SGVector<double> x, SGVector<double> y);

	private:

		/* Standardize the columns of a matrix */
		static void standardize(SGMatrix<double> &eta);

};


/* Standardize the columns of a matrix */
void HSIC::standardize(SGMatrix<double> &eta) {
	int T = eta.num_rows, dim = eta.num_cols;

	SGVector<double> mean(dim), std_dev(dim);
	mean = Statistics::matrix_mean(eta);
	std_dev = Statistics::matrix_std_deviation(eta);

	for (int t = 0; t < T; ++t) {
		for (int i = 0; i < dim; ++i) {
			eta(t, i) = eta(t,i) - mean[i]; // scale an observation by the sample mean so that it has mean zero
			//eta(t, i) = (eta(t,i) - mean[i]) / std_dev[i]; // normalize an observation so that it has mean zero and variance one
		}
	}
}

/* Define the Gaussian kernel with unit variance */
double HSIC::kernel(SGVector<double> x, SGVector<double> y) {
	double sigma = 1.;
	double normv = shogun::linalg::norm( shogun::linalg::add(x, y, 1., -1.) );
	return exp( -0.5 * pow(normv/sigma, 2.) );
}

/* Calculate the HSIC-based test statistics for residuals.
INPUT: T by dim matrices of residuals (eta1 and eta2). */
template<double kernel1(SGVector<double>, SGVector<double>), double kernel2(SGVector<double>, SGVector<double>)>
std::pair<double, double> HSIC::J(	const SGMatrix<double> &eta1, /*T by dim1 matrix*/
									const SGMatrix<double> &eta2, /*T by dim2 matrix*/
									int M /*maximum number of lags*/) {
	int T = eta1.num_rows;
	ASSERT_(T == eta2.num_rows);

	/* Compute values of the kernels */
	SGMatrix<double> K1(T, T), K2(T, T);
	for (int t = 0; t < T; ++t) {
		for (int s = t; s < T; ++s) { // for s >= t
			K1(t, s) = kernel1( eta1.get_row_vector(t), eta1.get_row_vector(s) );
			K1(s, t) = K1(t, s);

			K2(t, s) = kernel2( eta2.get_row_vector(t), eta2.get_row_vector(s) );
			K2(s, t) = K2(t, s);
		}
	}

	/* Compute the joint HSIC-based statistics */
	double sum1 = 0., sum2a = 0., sum2b = 0., sum3 = 0.;
	double J1 = 0., J2 = 0.;
	for (int m = 0; m < M; ++m) {
		sum1 = 0.; // reset the summations
		sum2a = 0.;
		sum2b = 0.;
		sum3 = 0.;
		for (int t = 0; t < T-m; ++t) {
			for (int s = 0; s < T-m; ++s) {
				sum1 += K1(t, s) * K2(t+m, s+m) / pow(T-m, 2.);
				sum2a += K1(t,s) / pow(T-m, 2.);
				sum2b += K2(t+m, s+m) / pow(T-m, 2.);
				for (int tau = 0; tau < T-m; ++tau) {
					sum3 += K1(t,s) * K2(t+m,tau+m) / pow(T-m, 3.);
				}
			}
		}

		J1 += sum1 + sum2a*sum2b - 2*sum3;

		sum1 = 0.; // reset the summations
		sum2a = 0.;
		sum2b = 0.;
		sum3 = 0.;
		for (int t = 0; t < T-m; ++t) {
			for (int s = 0; s < T-m; ++s) {
				sum1 += K1(t+m,s+m) * K2(t,s) / pow(T-m, 2.);
				sum2a += K1(t+m,s+m) / pow(T-m, 2.);
				sum2b += K2(t,s) / pow(T-m, 2.);
				for (int tau = 0; tau < T-m; ++tau) {
					sum3 += K1(t+m,s+m) * K2(t,tau) / pow(T-m, 3.);
				}
			}
		}

		J2 += sum1 + sum2a*sum2b - 2*sum3;
	}

	return {J1, J2};
}

/* Calculate the bootstrap p-values of the HSIC-based tests with residuals from multivariate processes.
INPUT: two T by 2 matrices of data */
template<	double kernel1(SGVector<double>, SGVector<double>), /*a kernel function for the first process*/
			double kernel2(SGVector<double>, SGVector<double>), /*a kernel function for the second process*/
			/*an estimator*/
			double estimator(	SGMatrix<double> &, /*T by 2 matrix of residuals*/
								SGVector<double> &, /*a vector of estimates*/
								const SGMatrix<double> &, /*T by 2 matrix of data*/
								SGVector<double>    /*a vector of initial parameters*/),
			/*generate data from residuals*/
			SGMatrix<double> gen_data(	const SGMatrix<double> &, /*T by 2 matrix of innovations*/
										const SGVector<double>    /*a vector of parameters*/),
			/*compute residuals*/
			SGMatrix<double> resid(	const SGMatrix<double> &, /*T by 2 matrix of observations*/
									SGVector<double> 		 /*a vector of parameters*/),
			/*compute the gradient vector*/
			SGVector<double> gradient(	const SGVector<double> &, /*a vector of parameters*/
										const SGMatrix<double> &, /*a T by 2 matrix of data*/
										double /*finite differential level*/),
			/*compute the Hessian matrix*/
			SGMatrix<double> hessian(	const SGVector<double> &, /*a 7 by 1 vector of parameters*/
										const SGMatrix<double> &, /*a T by 2 matrix of data*/
										double  /*a finite differential level*/) >
std::pair<double, double> HSIC::resid_bootstrap_pvalue(	const SGMatrix<double> &Y1, /*T by 2 matrix of data*/
														const SGMatrix<double> &Y2,
														const SGVector<double> theta01, /*a vector of starting values used to estimate the DGP*/
														const SGVector<double> theta02,
														int M, /*maximum number of lags*/
														int num_bts, /*number of bootstrap samples*/
														double h, /*a finite differential factor. If h = 0., then do MLE*/
														unsigned long seed) {
	int T = Y1.num_rows, dim = Y1.num_cols;
	ASSERT_(T == Y2.num_rows && dim == Y2.num_cols && theta01.vlen == theta02.vlen);

	double fmin1 = 0., fmin2 = 0., J1 = 0., J2 = 0., J1_bt = 0., J2_bt = 0.;
	SGMatrix<double> resid1(T, dim), resid2(T, dim);
	SGVector<double> theta1(theta01.vlen), theta2(theta02.vlen);

	/** 1. Estimate the original model and compute the HSIC-based statistics **/
	fmin1 = estimator(resid1, theta1, Y1, theta01);
	fmin2 = estimator(resid2, theta2, Y2, theta02);

	/* Standardize residuals */
	standardize(resid1);
	standardize(resid2);

	std::tie(J1, J2) = HSIC::J<kernel1, kernel2>(resid1, resid2, M);

	/** 2. Compute the bootstrap p-values **/

	gsl_rng *r = nullptr;
	const gsl_rng_type *gen; // call random number generator
	gsl_rng_env_setup();
	gen = gsl_rng_default;
	r = gsl_rng_alloc(gen);
	gsl_rng_set(r, seed);

	int b = 0, t = 0, i = 0, count1 = 0, count2 = 0;
	int skip = 0;
	bool positivity = true;

	SGMatrix<double> resid1_bt(T, dim), resid2_bt(T, dim);
	SGMatrix<double> Y1_bt(T, dim), Y2_bt(T, dim);
	SGVector<double> theta1_bt(theta01.vlen), theta2_bt(theta02.vlen);
	SGVector<double> theta1_mgarch_bt(7), theta2_mgarch_bt(7);
	SGVector<int> index_vec1_bt(T), index_vec2_bt(T);


	SGVector<int> index_vec(T);
	SGVector<int>::range_fill_vector(index_vec.vector, T, 0);

	//#pragma omp parallel for default(shared) reduction (+:count1, count2, skip) schedule(dynamic, CHUNK) private(b, i, t) firstprivate(fmin1, fmin2, J1_bt, J2_bt)
	for (b = 0; b < num_bts; ++b) {
		/** 2.a. Sample with replacement from the residuals **/
		gsl_ran_sample(r, index_vec1_bt.vector, T, index_vec.vector, T, sizeof(int) );
		for (t = 0; t < T; ++t) {
			for (i = 0; i < dim; ++i) {
				resid1_bt(t, i) = resid1(index_vec1_bt[t], i);
			}
		}

		gsl_ran_sample(r, index_vec2_bt.vector, T, index_vec.vector, T, sizeof(int) );
		for (t = 0; t < T; ++t) {
			for (i = 0; i < dim; ++i) {
				resid2_bt(t, i) = resid2(index_vec2_bt[t], i);
			}
		}

		/*
		resid1_bt.zero(); // reset matrices of bootstrap residuals
		resid2_bt.zero();
		for (i = 0; i < dim; ++i) {
			gsl_ran_sample( r, resid1_bt.get_column_vector(i), T, resid1.get_column_vector(i), T, sizeof(double) );
			gsl_ran_sample( r, resid2_bt.get_column_vector(i), T, resid2.get_column_vector(i), T, sizeof(double) );
		}*/

		/** 2.b. Standardize bootstrap residuals **/
		//standardize(resid1_bt);
		//standardize(resid2_bt);

		/** 2.c. Generate bootstrap samples */
		Y1_bt = gen_data(resid1_bt, theta1);
		Y2_bt = gen_data(resid2_bt, theta2);

		/** 2.d. Calculate the bootstrap statistics **/
		if (Math::fequals_abs(h, 0., 1e-5) == true) {
			// Estimate the VAR/CC-MGARCH model with bootstrap samples using the MLE (this can take a long time)
			fmin1 = estimator(resid1_bt, theta1_bt, Y1_bt, theta1); // re-estimate the model with bootstrap samples
			fmin2 = estimator(resid2_bt, theta2_bt, Y2_bt, theta2);

			resid1_bt = resid(Y1_bt, theta1_bt);
			resid2_bt = resid(Y2_bt, theta2_bt);

			// Standardize the bootstrap residuals
			standardize(resid1_bt);
			standardize(resid2_bt);

			// Calculate the bootstrap statistics
			std::tie(J1_bt, J2_bt) = HSIC::J<kernel1, kernel2>(resid1_bt, resid2_bt, M);
		}
		else {
			// Estimate the VAR/CC-MGARCH model with bootstrap samples using the asymptotic linear expansion (this can save a lot of time)
			theta1_bt = add(theta1, matrix_prod( pinv<double>( hessian(theta1, Y1_bt, h) ), gradient(theta1, Y1_bt, h) ), 1., -1.);
			theta2_bt = add(theta2, matrix_prod( pinv<double>( hessian(theta2, Y2_bt, h) ), gradient(theta2, Y2_bt, h) ), 1., -1.);

			if (theta1_bt.vlen == 7) { // if the model is CC-MGARCH
				// impose the positiveness constraint
				if ( (Math::min<double>(theta1_bt, theta1_bt.vlen-1) > 0.) && (theta1_bt[6] > -1.) && (theta1_bt[6] < 1.) \
						&& (Math::min<double>(theta2_bt, theta2_bt.vlen-1) > 0.) && (theta2_bt[6] > -1.) && (theta2_bt[6] < 1.) ) {

					positivity = true;
					resid1_bt = resid(Y1_bt, theta1_bt);
					resid2_bt = resid(Y2_bt, theta2_bt);

					// Standardize the bootstrap residuals
					standardize(resid1_bt);
					standardize(resid2_bt);

					// Calculate the bootstrap statistics
					std::tie(J1_bt, J2_bt) = HSIC::J<kernel1, kernel2>(resid1_bt, resid2_bt, M);
				}
				else {
					positivity = false;
					skip += 1;
					//cout << "HSIC::resid_bootstrap_pvalue: Skip an iteration!" << endl;
				}
			}
			else if (theta1_bt.vlen == 4) { // if the model is VAR(1)
				resid1_bt = resid(Y1_bt, theta1_bt);
				resid2_bt = resid(Y2_bt, theta2_bt);

				// Standardize the bootstrap residuals
				standardize(resid1_bt);
				standardize(resid2_bt);

				// Calculate the bootstrap statistics
				std::tie(J1_bt, J2_bt) = HSIC::J<kernel1, kernel2>(resid1_bt, resid2_bt, M);
			}
			else if (theta1_bt.vlen == 11) { // if the model is VAR-CC-MGARCH(1,1)
				theta1_mgarch_bt = get_subvector(theta1_bt, 4, 10);
				theta2_mgarch_bt = get_subvector(theta2_bt, 4, 10);

				// impose the positiveness constraint
				if ( (Math::min<double>(theta1_mgarch_bt, theta1_mgarch_bt.vlen-1) >= 0.) && (theta1_mgarch_bt[6] > -1.) && (theta1_mgarch_bt[6] < 1.) \
						&& (Math::min<double>(theta2_mgarch_bt, theta2_mgarch_bt.vlen-1) >= 0.) && (theta2_mgarch_bt[6] > -1.) && (theta2_mgarch_bt[6] < 1.) ) {

					positivity = true;
					resid1_bt = resid(Y1_bt, theta1_bt);
					resid2_bt = resid(Y2_bt, theta2_bt);

					// Standardize the bootstrap residuals
					standardize(resid1_bt);
					standardize(resid2_bt);

					// Calculate the bootstrap statistics
					std::tie(J1_bt, J2_bt) = HSIC::J<kernel1, kernel2>(resid1_bt, resid2_bt, M);
				}
				else {
					positivity = false;
					skip += 1;
					//cout << "HSIC::resid_bootstrap_pvalue: Skip an iteration!" << endl;
				}
			}
			else {
				cerr << "HSIC::resid_bootstrap_pvalue: The number of parameters must be either 4 or 7!\n";
				exit(0);
			}
		}

		//cout << J1 << " , " << J2 << endl;
		//cout << J1_bt << " , " << J2_bt << endl;

		/** 2.e. Count the number of times where a bootstrap statistic is greater or equal to the sample one **/
		if (J1_bt >= J1 && positivity == true)
			count1++;
		if (J2_bt >= J2 && positivity == true)
			count2++;
	}

	num_bts = num_bts - skip;
	//cout << "HSIC::resid_bootstrap_pvalue: num_bts = " << num_bts << endl;

	double pvalue1 = 0., pvalue2 = 0.;
	if (num_bts == 0) {
		pvalue1 = 2.;
		pvalue2 = 2.;
	}
	else {
		pvalue1 = ( (double) count1 / num_bts );
		pvalue2 = ( (double) count2 / num_bts );
	}

    gsl_rng_free(r); //free up memory

	return {pvalue1, pvalue2};
}




































#endif
