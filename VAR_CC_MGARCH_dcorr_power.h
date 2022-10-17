#ifndef VAR_CC_MGARCH_DCORR_POWER_H
#define VAR_CC_MGARCH_DCORR_POWER_H

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <shogun/lib/exception/ShogunException.h>
#include <nongaussian_dist_corr.h>
#include <kernel.h>
#include <nl_dgp.h>
#include <VAR_CC_MGARCH_reg_dcorr.h>

using namespace std;

class VAR_CC_MGARCH_dcorr_power {
	public:
		VAR_CC_MGARCH_dcorr_power (){  }; //default constructor
		~VAR_CC_MGARCH_dcorr_power () { }; //default destructor

		/* Calculate the distance-based test statistic by fitting observations to a VAR-CC-MGARCH(1,1) model*/
		template<double kernel_k(double)>
		static double do_Test(	const SGMatrix<double> &X, const SGMatrix<double> &Y, /*T by 2 matrices of observations*/
								int lag_smooth, /*kernel bandwidth*/
								double expn, /*exponent of the Euclidean distance*/
								SGVector<double> theta_mgarch0_X, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
								SGVector<double> theta_mgarch0_Y, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
								double kernel_QDSum, double kernel_QRSum /*quadratically and quartically integrate the kernel function*/);

		/* Calculate the bootstrap test statistics by fitting observations to a VAR-CC-MGARCH(1,1) model.
		OUTPUT: a bootstrap p-value (pvalue), a value of the statistic (cstat), and a vector of bootstrap statistics (cstat_bootstrap) */
		template<double kernel_k(double) /*kernel to smooth lags*/>
		static std::tuple<	double /*p-value*/,
							double /*statistic*/,
							SGVector<double> /*bootstrap statistics*/> do_Test_bt(	const SGMatrix<double> &X, const SGMatrix<double> &Y, /*T by 2 matrices of observations*/
																					int lag_smooth, /*kernel bandwidth*/
																					double expn, /*exponent of the Euclidean distance*/
																					SGVector<double> theta_mgarch0_X, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																					SGVector<double> theta_mgarch0_Y, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																					const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices
																																				of auxiliary random variables
																																				used for bootstrapping*/
																					double kernel_QDSum, double kernel_QRSum /*quadratically and quartically integrate
																															   the kernel function*/);

		/* Calculate 5%- and 10%- asymptotic sizes and empirical critical values. */
		template <	/*simulate data*/
					void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
									const SGVector<double>, /*a vector of parameters for the first process*/
									const SGVector<double>, /*a vector of parameters for the second process*/
									const int, /*select how the innovations are generated*/
									unsigned long /*a seed used to generate random numbers*/),
					double kernel(double) /* a kernel function*/>
		static void cValue (Matrix &asymp_REJF, Matrix &empir_CV,
							const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
							const SGVector<double> &theta01, const SGVector<double> &theta02, /*7 by 1 vector of initial values
																								used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
							const int number_sampl, /*number of random samples drawn*/
							const int T, /*sample size*/
							const int lag_smooth, /*kernel bandwidth*/
							const double expn, /*exponent of the Euclidean distance*/
							const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
							int seed,
							ofstream &size_out);

		/* Calculate 5%- and 10%- empirical and asymptotic rejection frequencies */
		template <	/*simulate data*/
					void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
									const SGVector<double>, /*a vector of parameters for the first process*/
									const SGVector<double>, /*a vector of parameters for the second process*/
									const int, /*select how the innovations are generated*/
									unsigned long /*a seed used to generate random numbers*/),
					double kernel(double) /* a kernel function*/ >
		static void power_f(Matrix &empir_REJF, Matrix &asymp_REJF,
							const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
							const SGVector<double> &theta01, const SGVector<double> &theta02, /*7 by 1 vector of initial values
																								used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
							const int number_sampl, /*number of random samples drawn*/
							const int T, /*sample size*/
							const int lag_smooth, /*kernel bandwidth*/
							const double expn, /*exponent of the Euclidean distance*/
							const Matrix &empir_CV, /*5% and 10% empirical critical values*/
							const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
							const int choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
							int seed,
							ofstream &pwr_out);


		/* Calculate 5%- and 10%- asymptotic and bootstrap sizes, and empirical critical values. */
		template <	/*simulate data*/
					void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
									const SGVector<double>, /*a vector of parameters for the first process*/
									const SGVector<double>, /*a vector of parameters for the second process*/
									const int, /*select how the innovations are generated*/
									unsigned long /*a seed used to generate random numbers*/),
					double kernel(double) /* a kernel function*/>
		static void cValue (Matrix &asymp_REJF, Matrix &empir_CV,
							Matrix &bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
							const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
							const SGVector<double> &theta01, const SGVector<double> &theta02, /*7 by 1 vector of initial values
																								used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
							const int number_sampl, /*number of random samples drawn*/
							const int T, /*sample size*/
							const int lag_smooth, /*kernel bandwidth*/
							const double expn, /*exponent of the Euclidean distance*/
							const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
							const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices of auxiliary random variables
																															  used for bootstrapping*/
							int seed,
							ofstream &size_out);


		/* Calculate 5%- and 10%- empirical, asymptotic, and bootstrap rejection frequencies */
		template <	/*simulate data*/
					void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
									const SGVector<double>, /*a vector of parameters for the first process*/
									const SGVector<double>, /*a vector of parameters for the second process*/
									const int, /*select how the innovations are generated*/
									unsigned long /*a seed used to generate random numbers*/),
					double kernel(double) /*kernel to smooth lags*/>
		static void power_f(Matrix &empir_REJF, Matrix &asymp_REJF, /*5% and 10% empirical and asymptotic rejection frequencies*/
							Matrix &bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
							const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
							const SGVector<double> &theta01, const SGVector<double> &theta02, /*7 by 1 vector of initial values
																								used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
							const int number_sampl, /*number of random samples drawn*/
							const int T, /*sample size*/
							const int lag_smooth, /*kernel bandwidth*/
							const double expn, /*exponent of the Euclidean distance*/
							const Matrix &empir_CV, /*5% and 10% empirical critical values*/
							const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
							const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices of auxiliary random variables
																															  used for bootstrapping*/
							const int choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
							int seed,
							ofstream &pwr_out);

	private:
		/* Integrate quadratic and quartic functions of a kernel weight */
		template <double kernel_k (double )>
		static void integrate_Kernel (double &kernel_QDSum, double &kernel_QRSum);

};

/* Integrate quadratic and quartic functions of a kernel weight */
template <double kernel_k (double )>
void VAR_CC_MGARCH_dcorr_power::integrate_Kernel (double &kernel_QDSum, double &kernel_QRSum) {
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

/* Calculate the distance-based test statistic by fitting observations to a VAR-CC-MGARCH(1,1) model*/
template<double kernel_k(double)>
double VAR_CC_MGARCH_dcorr_power::do_Test(	const SGMatrix<double> &X, const SGMatrix<double> &Y, /*T by 2 matrices of observations*/
											int lag_smooth, /*kernel bandwidth*/
											double expn, /*exponent of the Euclidean distance*/
											SGVector<double> theta_mgarch0_X, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
											SGVector<double> theta_mgarch0_Y, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
											double kernel_QDSum, double kernel_QRSum /*quadratically and quartically integrate the kernel function*/) {
	int T = X.num_rows;
	int t = 1, s = 1, tau = 1, lag = 0;
	ASSERT_(T == Y.num_rows && X.num_cols == 2);
	int L = 1, T0 = T-L;

	SGMatrix<double> var_U_x(T0, T0), var_U_y(T0, T0);
	//#pragma omp parallel sections num_threads(2)
	{
		//calculate \H{U}_{t,s}^{(x)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
//		//#pragma omp section
		var_U_x = VAR_CC_MGARCH_REG_DCORR::var_U(X, expn, theta_mgarch0_X);
		//var_U_x.display_matrix("var_U_x");

		//calculate \H{U}_{t,s}^{(y)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
		//#pragma omp section
		var_U_y = VAR_CC_MGARCH_REG_DCORR::var_U(Y, expn, theta_mgarch0_Y);
		//var_U_y.display_matrix("var_U_y");


	}

	/* calculate the asymptotic variance */
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
		if (lag == 0) weight = 1.;
		else weight = kernel_k ((double) lag/lag_smooth);

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

/* Calculate the test statistic and bootstrap p-values by fitting observations to a VAR-CC-MGARCH(1,1) model.
OUTPUT: a bootstrap p-value (pvalue), a value of the statistic (cstat), and a vector of bootstrap statistics (cstat_bootstrap) */
template<double kernel_k(double) /*kernel to smooth lags*/>
std::tuple<	double /*p-value*/,
			double /*statistic*/,
			SGVector<double> /*bootstrap statistics*/> VAR_CC_MGARCH_dcorr_power::do_Test_bt(const SGMatrix<double> &X, const SGMatrix<double> &Y, /*T by 2 matrices of observations*/
																							int lag_smooth, /*kernel bandwidth*/
																							double expn, /*exponent of the Euclidean distance*/
																							SGVector<double> theta_mgarch0_X, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																							SGVector<double> theta_mgarch0_Y, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																							const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices
																																						of auxiliary random variables
																																						used for bootstrapping*/
																							double kernel_QDSum, double kernel_QRSum /*quadratically and quartically integrate
																																	   the kernel function*/) {
	int T = X.num_rows, num_B = xi_x.num_rows;
	int t = 1, s = 1, tau = 1, lag = 0, i = 0;

    ASSERT_(num_B == xi_y.num_rows && T == Y.num_rows && T == xi_x.num_cols && T == xi_y.num_cols); //check for the size consistency of matrices

	int T0 = T-1;

	SGMatrix<double> var_U_x(T0,T0), var_U_y(T0,T0);

	//#pragma omp parallel sections num_threads(2)
	{
		//calculate \H{U}_{t,s}^{(x)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
//		//#pragma omp section
		var_U_x = VAR_CC_MGARCH_REG_DCORR::var_U(X, expn, theta_mgarch0_X);

		//calculate \H{U}_{t,s}^{(y)}, t = 0, ..., T0-1 and s = 0, ..., T0-1
		//#pragma omp section
		var_U_y = VAR_CC_MGARCH_REG_DCORR::var_U(Y, expn, theta_mgarch0_Y);
	}

	/* calculate the asymptotic variance */
	double aVar = 0.;
	for (s = 0; s < T0; ++s) {
		for (t = s+1; t < T0; ++t) {
			aVar += ( (double) 1/pow(T0, 2.) ) * pow(var_U_x(t, s)*var_U_y(t, s), 2.);
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


	double cstat =  ( (double) sum / sqrt(lag_smooth) ); // compute the centered statistic
	double tstat = sum / ( 2 * sqrt(lag_smooth * kernel_QRSum * aVar) ); // compute the t-statistic
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



/* Calculate 5%- and 10%- asymptotic sizes and empirical critical values. */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			double kernel(double) /* a kernel function*/>
void VAR_CC_MGARCH_dcorr_power::cValue (Matrix &asymp_REJF, Matrix &empir_CV,
										const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
										const SGVector<double> &theta01, const SGVector<double> &theta02, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
										const int number_sampl, /*number of random samples drawn*/
										const int T, /*sample size*/
										const int lag_smooth, /*kernel bandwidth*/
										const double expn, /*exponent of the Euclidean distance*/
										const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
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
	VAR_CC_MGARCH_dcorr_power::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);

	size_out << std::fixed << std::setprecision(5);
	size_out << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;

    cout << "Calculating empirical critical values ..." << endl;

    int i = 1, choose_alt = 0;
    //SGMatrix<double> X(T,2), Y(T,2);
    Matrix tvalue(number_sampl,1);
    int asymp_REJ_5 = 0, asymp_REJ_10 = 0;
    int skip = 0;

    size_out << std::fixed << std::setprecision(5);

	#pragma omp parallel for default(shared) reduction (+:skip,asymp_REJ_5,asymp_REJ_10) schedule(dynamic,CHUNK) private(i) firstprivate(rseed)
	for (i = 1; i <= number_sampl; i++) {
		try {
			SGMatrix<double> X(T,2), Y(T,2);

			rseed = gsl_rng_get(r); //generate a random seed

			//draw two independent random samples from the dgp of X and Y each using a random seed
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);

			//calculate the test statistic
			tvalue(i) = VAR_CC_MGARCH_dcorr_power::do_Test<kernel>(	X, Y, /*T by 2 matrices of observations*/
																	lag_smooth, /*kernel bandwidth*/
																	expn, /*exponent of the Euclidean distance*/
																	theta01, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																	theta02, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																	kernel_QDSum, kernel_QRSum /*quadratically and quartically integrate the kernel function*/);
		}
		catch(shogun::ShogunException) { //catch a Shogun exception
			cerr << "VAR_CC_MGARCH_dcorr_power::cValue: An exception occurs!" << endl;
			tvalue(i) = 0;
			++skip;
		}

		#pragma omp critical
		{
			size_out << tvalue(i) << endl;
		}

		if (tvalue(i) >= asymp_CV(1)) ++asymp_REJ_5; //using 5%-critical value
		if (tvalue(i) >= asymp_CV(2)) ++asymp_REJ_10; //using 10%-critical value
	}


	asymp_REJF(1) = ( (double) asymp_REJ_5 / (number_sampl-skip) ); //calculate sizes
	asymp_REJF(2) = ( (double) asymp_REJ_10/ (number_sampl-skip) );

	empir_CV(1) = quantile (tvalue, 0.95); //calculate quantiles
	empir_CV(2) = quantile (tvalue, 0.90);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << empir_CV(2) << endl;
	gsl_rng_free (r); //free up memory
}


/* Calculate 5%- and 10%- empirical and asymptotic rejection frequencies */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			double kernel(double) /* a kernel function*/ >
void VAR_CC_MGARCH_dcorr_power::power_f(Matrix &empir_REJF, Matrix &asymp_REJF,
										const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
										const SGVector<double> &theta01, const SGVector<double> &theta02, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
										const int number_sampl, /*number of random samples drawn*/
										const int T, /*sample size*/
										const int lag_smooth, /*kernel bandwidth*/
										const double expn, /*exponent of the Euclidean distance*/
										const Matrix &empir_CV, /*5% and 10% empirical critical values*/
										const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
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
	VAR_CC_MGARCH_dcorr_power::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);
	cout << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;

	int i = 1, empir_REJ_5 = 0, asymp_REJ_5 = 0, empir_REJ_10 = 0, asymp_REJ_10 = 0;
	int skip = 0;

	Matrix tvalue(number_sampl,1);
	//SGMatrix<double> X(T,2), Y(T,2);

	pwr_out << std::fixed << std::setprecision(5);


	#pragma omp parallel for default(shared) reduction(+:skip,empir_REJ_5,empir_REJ_10,asymp_REJ_5,asymp_REJ_10) schedule(dynamic,CHUNK) private(i) \
																																	firstprivate(rseed)
	for (i = 1; i <= number_sampl; ++i) {
		try {
			SGMatrix<double> X(T,2), Y(T,2);

			rseed = gsl_rng_get(r); //generate a random seed

			//draw two independent random samples from the dgp of X and Y each using a random seed
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);


			//calculate the test statistic
			tvalue(i) = VAR_CC_MGARCH_dcorr_power::do_Test<kernel>(	X, Y, /*T by 2 matrices of observations*/
																	lag_smooth, /*kernel bandwidth*/
																	expn, /*exponent of the Euclidean distance*/
																	theta01, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																	theta02, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																	kernel_QDSum, kernel_QRSum /*quadratically and quartically integrate the kernel function*/);

		}
		catch(shogun::ShogunException) { //catch a Shogun exception
			cerr << "VAR_CC_MGARCH_dcorr_power::power_f: An exception occurs!" << endl;
			tvalue(i) = 0;
			++skip;
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

	empir_REJF(1) = ((double) empir_REJ_5/(number_sampl-skip) );
	empir_REJF(2) = ((double) empir_REJ_10/(number_sampl-skip) );
	asymp_REJF(1) = ((double) asymp_REJ_5/(number_sampl-skip) );
	asymp_REJF(2) = ((double) asymp_REJ_10/(number_sampl-skip) );
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << empir_REJF(1) << " and " << empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << asymp_REJF(1) << " and " << asymp_REJF(2) << endl;

	gsl_rng_free (r); // free up memory
}

#if 0
/* Calculate 5% and 10% bootstrap rejection frequencies */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			double kernel(double) /*kernel to smooth lags*/>
void VAR_CC_MGARCH_dcorr_power::power_f(Matrix &bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
										const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
										const SGVector<double> &theta01, const SGVector<double> &theta02, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
										const int number_sampl, /*number of random samples drawn*/
										const int lag_smooth, /*kernel bandwidth*/
										const double expn, /*exponent of the Euclidean distance*/
										const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices of auxiliary random variables
																															  used for bootstrapping*/
										const int choose_alt, /*alternative hypotheses: 1-2 (linear correlation) and 2-5 (non-correlation) as defined in Wang et al. (2021, page 22)*/
										int seed,
										ofstream &pwr_out) {

	int num_B = xi_x.num_rows, T = xi_x.num_cols;

	gsl_rng * r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

	//SGMatrix<double> X(T,2), Y(T,2);
	int i = 1, bootstrap_REJ_5 = 0, bootstrap_REJ_10 = 0;
	int skip = 0;
	Matrix pvalue(number_sampl, 1), cstat(number_sampl, 1);
	//SGVector<double> cstat_bootstrap(num_B);

	pwr_out << std::fixed << std::setprecision(5);
	pwr_out << " pvalue " << "   ,   " << "statistics" << endl;

	#pragma omp parallel for default(shared) reduction (+:skip,bootstrap_REJ_5,bootstrap_REJ_10) schedule(static,CHUNK) private(i) firstprivate(rseed)
	for (i = 1; i <= number_sampl; ++i) {
		try {
			SGMatrix<double> X(T,2), Y(T,2);

			rseed = gsl_rng_get(r); //generate a random seed

			//draw two independent random samples from the dgp of X and Y each using a random seed
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);


			SGVector<double> cstat_bootstrap(num_B);

			//calculate bootstrap p-values
			std::tie(pvalue(i), cstat(i), cstat_bootstrap) = \
														VAR_CC_MGARCH_dcorr_power::do_Test_bt<kernel>(	X, Y, /*T by 2 matrices of observations*/
																										lag_smooth, /*kernel bandwidth*/
																										expn, /*exponent of the Euclidean distance*/
																										theta01, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																										theta02, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																										xi_x, xi_y);
		}
		catch(shogun::ShogunException) { //catch a Shogun exception
			cerr << "VAR_CC_MGARCH_dcorr_power::power_f: An exception occurs!" << endl;
			pvalue(i) = 1.;
			++skip;
		}

		#pragma omp critical
		{
			pwr_out <<  pvalue(i) << "   ,   " << cstat(i) << endl;
		}

		if (pvalue(i) <= 0.05)  ++bootstrap_REJ_5;//using 5%-critical value
		if (pvalue(i) <= 0.10)  ++bootstrap_REJ_10;//using 10%-critical value
	}
	bootstrap_REJF(1) = ((double) bootstrap_REJ_5/(number_sampl-skip) );
	bootstrap_REJF(2) = ((double) bootstrap_REJ_10/(number_sampl-skip) );

	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% bootstrap reject frequencies for choose_alt = " << choose_alt << " are "
	        << bootstrap_REJF(1) << " and " << bootstrap_REJF(2) << endl;

	gsl_rng_free (r); // free up memory
}
#endif

/* Calculate 5%- and 10%- asymptotic and bootstrap sizes, and empirical critical values. */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			double kernel(double) /* a kernel function*/>
void VAR_CC_MGARCH_dcorr_power::cValue (Matrix &asymp_REJF, Matrix &empir_CV,
										Matrix &bootstrap_REJF, /*5%- and 10%- bootstrap rejection frequencies*/
										const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
										const SGVector<double> &theta01, const SGVector<double> &theta02, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
										const int number_sampl, /*number of random samples drawn*/
										const int T, /*sample size*/
										const int lag_smooth, /*kernel bandwidth*/
										const double expn, /*exponent of the Euclidean distance*/
										const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
										const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices of auxiliary random variables
																															  used for bootstrapping*/
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
	VAR_CC_MGARCH_dcorr_power::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);

	size_out << std::fixed << std::setprecision(5);
	size_out << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;

    cout << "Calculating empirical critical values ..." << endl;

    int i = 1, choose_alt = 0, num_B = xi_x.num_rows;
    //SGMatrix<double> X(T,2), Y(T,2);
    Matrix pvalue(number_sampl,1), tvalue(number_sampl,1);
    int asymp_REJ_5 = 0, asymp_REJ_10 = 0, bootstrap_REJ_5 = 0, bootstrap_REJ_10 = 0;
    int skip = 0;

    size_out << std::fixed << std::setprecision(5);
    size_out << "bootstrap pvalue" << " , " << "t-stat" << endl;

	#pragma omp parallel for default(shared) reduction (+:skip,asymp_REJ_5,asymp_REJ_10,bootstrap_REJ_5,bootstrap_REJ_10) \
																						schedule(dynamic,CHUNK) private(i) firstprivate(rseed)
	for (i = 1; i <= number_sampl; i++) {
		try {
			SGMatrix<double> X(T,2), Y(T,2);

			rseed = gsl_rng_get(r); //generate a random seed

			//draw two independent random samples from the dgp of X and Y each using a random seed
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);

			SGVector<double> cstat_bootstrap(num_B);

			//calculate the test statistic
			std::tie(pvalue(i), tvalue(i), cstat_bootstrap) = \
														VAR_CC_MGARCH_dcorr_power::do_Test_bt<kernel>(	X, Y, /*T by 2 matrices of observations*/
																										lag_smooth, /*kernel bandwidth*/
																										expn, /*exponent of the Euclidean distance*/
																										theta01, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																										theta02, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																										xi_x, xi_y,
																										kernel_QDSum, kernel_QRSum);
		}
		catch(shogun::ShogunException) { //catch a Shogun exception
			cerr << "VAR_CC_MGARCH_dcorr_power::cValue: An exception occurs!" << endl;
			pvalue(i) = 1.;
			tvalue(i) = 0.;
			++skip;
		}

		#pragma omp critical
		{
			size_out << pvalue(i) << " , " << tvalue(i) << endl;
		}

		if (tvalue(i) >= asymp_CV(1)) ++asymp_REJ_5; //using 5%-critical value
		if (tvalue(i) >= asymp_CV(2)) ++asymp_REJ_10; //using 10%-critical value
		if (pvalue(i) <= 0.05)  ++bootstrap_REJ_5;//using 5%-critical value
		if (pvalue(i) <= 0.10)  ++bootstrap_REJ_10;//using 10%-critical value
	}


	asymp_REJF(1) = ( (double) asymp_REJ_5 / (number_sampl-skip) ); //calculate sizes
	asymp_REJF(2) = ( (double) asymp_REJ_10/ (number_sampl-skip) );

	empir_CV(1) = quantile (tvalue, 0.95); //calculate quantiles
	empir_CV(2) = quantile (tvalue, 0.90);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << empir_CV(2) << endl;

	bootstrap_REJF(1) = ((double) bootstrap_REJ_5/(number_sampl-skip) );
	bootstrap_REJF(2) = ((double) bootstrap_REJ_10/(number_sampl-skip) );
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-bootstrap size for T =" << T << " is " << bootstrap_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-bootstrap size for T = " << T << " is " << bootstrap_REJF(2) << endl;

	gsl_rng_free (r); //free up memory
}


/* Calculate 5%- and 10%- empirical, asymptotic, and bootstrap rejection frequencies */
template <	/*simulate data*/
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),
			double kernel(double) /* a kernel function*/ >
void VAR_CC_MGARCH_dcorr_power::power_f(Matrix &empir_REJF, Matrix &asymp_REJF, /*5% and 10% empirical and asymptotic rejection frequencies*/
										Matrix &bootstrap_REJF, /*5% and 10% bootstrap rejection frequencies*/
										const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
										const SGVector<double> &theta01, const SGVector<double> &theta02, /*7 by 1 vector of initial values
																											used to initialize the MLE estimator for VAR-CC-MGARCH(1,1)*/
										const int number_sampl, /*number of random samples drawn*/
										const int T, /*sample size*/
										const int lag_smooth, /*kernel bandwidth*/
										const double expn, /*exponent of the Euclidean distance*/
										const Matrix &empir_CV, /*5% and 10% empirical critical values*/
										const Matrix &asymp_CV, /*5% and 10% asymptotic critical values*/
										const SGMatrix<double> &xi_x, const SGMatrix<double> &xi_y, /*num_B by T matrices of auxiliary random variables
																															  used for bootstrapping*/
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
	VAR_CC_MGARCH_dcorr_power::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);
	cout << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;

	int i = 1, empir_REJ_5 = 0, asymp_REJ_5 = 0, empir_REJ_10 = 0, asymp_REJ_10 = 0, bootstrap_REJ_5 = 0, bootstrap_REJ_10 = 0;
	int skip = 0;
	int num_B = xi_x.num_rows;

	Matrix tvalue(number_sampl,1), pvalue(number_sampl, 1);
	//SGMatrix<double> X(T,2), Y(T,2);

	pwr_out << std::fixed << std::setprecision(5);
	pwr_out << " bootstrap pvalue " << "   ,   " << "t-stat" << endl;

	#pragma omp parallel for default(shared) reduction(+:skip,empir_REJ_5,empir_REJ_10,asymp_REJ_5,asymp_REJ_10,bootstrap_REJ_5,bootstrap_REJ_10) \
																								schedule(dynamic,CHUNK) private(i) firstprivate(rseed)
	for (i = 1; i <= number_sampl; ++i) {
		try {
			SGMatrix<double> X(T,2), Y(T,2);

			rseed = gsl_rng_get(r); //generate a random seed

			//draw two independent random samples from the dgp of X and Y each using a random seed
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);

			SGVector<double> cstat_bootstrap(num_B);

			//calculate the test statistic
			std::tie(pvalue(i), tvalue(i), cstat_bootstrap) = \
														VAR_CC_MGARCH_dcorr_power::do_Test_bt<kernel>(	X, Y, /*T by 2 matrices of observations*/
																										lag_smooth, /*kernel bandwidth*/
																										expn, /*exponent of the Euclidean distance*/
																										theta01, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																										theta02, /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/
																										xi_x, xi_y,
																										kernel_QDSum, kernel_QRSum);

		}
		catch(shogun::ShogunException) { //catch a Shogun exception
			cerr << "VAR_CC_MGARCH_dcorr_power::power_f: An exception occurs!" << endl;
			pvalue(i) = 1.;
			tvalue(i) = 0.;
			++skip;
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

	empir_REJF(1) = ((double) empir_REJ_5/(number_sampl-skip) );
	empir_REJF(2) = ((double) empir_REJ_10/(number_sampl-skip) );
	asymp_REJF(1) = ((double) asymp_REJ_5/(number_sampl-skip) );
	asymp_REJF(2) = ((double) asymp_REJ_10/(number_sampl-skip) );
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << empir_REJF(1) << " and " << empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << asymp_REJF(1) << " and " << asymp_REJF(2) << endl;

	bootstrap_REJF(1) = ((double) bootstrap_REJ_5/(number_sampl-skip) );
	bootstrap_REJF(2) = ((double) bootstrap_REJ_10/(number_sampl-skip) );
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% bootstrap reject frequencies for choose_alt = " << choose_alt << " are "
	        << bootstrap_REJF(1) << " and " << bootstrap_REJF(2) << endl;

	gsl_rng_free (r); // free up memory
}






#endif

