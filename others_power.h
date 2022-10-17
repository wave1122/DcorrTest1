#ifndef OTHERS_POWER_H
#define OTHERS_POWER_H

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <nongaussian_dist_corr.h>
#include <kernel.h>
#include <nl_dgp.h>
#include <dep_tests.h>
#include <hsic_test.h>

using namespace std;

class Others_power {
	public:
		Others_power (){  }; //default constructor
		~Others_power () { }; //default destructor

		/* Calculate 5%- and 10%- asymptotic sizes and empirical critical values for Wang et al.'s (2021) HSIC-based test, El Himdi and Roy's (1997) test,
		Bouhaddioui and Roy's (2006) test, and Tchahou and Duchesne's (2013) L1 test.
		OUTPUT: 5%- and 10%- sizes (asymp_REJF) and empirical critical values (empir_CV) */
		template<	//![simulate data]
					void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
									const SGVector<double>, /*a vector of parameters for the first process*/
									const SGVector<double>, /*a vector of parameters for the second process*/
									const int, /*select how the innovations are generated*/
									unsigned long /*a seed used to generate random numbers*/),

					//![set parameters for the HSIC test statistics]
					double kernel1(SGVector<double>, SGVector<double>), /*a kernel function for the first process*/
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
												double  /*a finite differential level*/),
					//![kernel function for Hong-type tests]
					double kernel(double) >
		static void cValue (Matrix &HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							Matrix &HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							Matrix &ER_asymp_REJF, /*asymptotic rejection rates of El Himdi and Roy's (1997) test*/
							Matrix &ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							Matrix &BR_asymp_REJF, /*asymptotic rejection rates of Bouhaddioui and Roy's (2006) test*/
							Matrix &BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							Matrix &RbF_asymp_REJF, /*asymptotic rejection rates of Robbins and Fisher's (2015) test*/
							Matrix &RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							Matrix &TD_L1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							Matrix &TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							Matrix &TD_T1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							Matrix &TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
							const SGVector<double> &theta01, const SGVector<double> &theta02, /*initial values used to estimate DGPs*/
							int number_sampl, /*number of random samples drawn*/
							const int T, /*sample size*/
							const int lag_smooth, /*kernel bandwidth*/
							const int num_bts, /*number of bootstrap samples*/
							const Matrix &ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							const Matrix &BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							const Matrix &RbF_asymp_CV, /*5% and 10% asymptotic critical values of and Robbins and Fisher's (2015) test*/
							const Matrix &TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							const Matrix &TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							double h, /*a finite differential factor*/
							unsigned long seed,
							ofstream &size_out,
							bool use_bartlett_ker = true);

		/* Calculate 5% and 10% empirical and asymptotic rejection frequencies at 5% and 10% empirical critical values (empir_CV) and
		5% and 10% asymptotic critical values (asymp_CV) Wang et al.'s (2021) HSIC-based test, El Himdi and Roy's (1997) test,
		Bouhaddioui and Roy's (2006) test, and Tchahou and Duchesne's (2013) L1 test.
		OUTPUT: 5% and 10% empirical rejection frequencies (empir_REJF) and asymptotic rejection frequencies (asymp_REJF). */
		template<	//![simulate data]
					void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
									const SGVector<double>, /*a vector of parameters for the first process*/
									const SGVector<double>, /*a vector of parameters for the second process*/
									const int, /*select how the innovations are generated*/
									unsigned long /*a seed used to generate random numbers*/),

					//![set parameters for the HSIC test statistics]
					double kernel1(SGVector<double>, SGVector<double>), /*a kernel function for the first process*/
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
												double  /*a finite differential level*/),
					//![kernel function for Hong-type tests]
					double kernel(double) >
		static void power_f (Matrix &HSIC_J1_empir_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							Matrix &HSIC_J2_empir_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							Matrix &ER_empir_REJF, Matrix &ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							Matrix &BR_empir_REJF, Matrix &BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							Matrix &RbF_empir_REJF, Matrix &RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							Matrix &TD_L1_empir_REJF, Matrix &TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							Matrix &TD_T1_empir_REJF, Matrix &TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
							const SGVector<double> &theta01, const SGVector<double> &theta02, /*initial values used to estimate DGPs*/
							int number_sampl, /*number of random samples drawn*/
							const int T, /*sample size*/
							const int lag_smooth, /*kernel bandwidth*/
							int num_bts, /*number of bootstrap samples*/
							const Matrix &ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							const Matrix &ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							const Matrix &BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							const Matrix &BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							const Matrix &RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							const Matrix &RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							const Matrix &TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							const Matrix &TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							const Matrix &TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							const Matrix &TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							const int choose_alt, /*set a degree of dependence between two DGPs*/
							double h, /*a finite differential factor*/
							unsigned long seed, /*seed for random number generator*/
							ofstream &pwr_out,
							bool use_bartlett_ker = true);

		//calculate 5%- and 10%- asymptotic sizes and empirical critical values when each X or Y is generated by a non-Gaussian AR(2) process. INPUT: number of random samples
        //generated (number_sampl), a sample size (T), a lag truncation (TL), a lag-smoothing parameter (lag_smooth), 6x1 vectors of AR coefficients (alpha_X and alpha_Y),
        //vectors of random errors (epsilon_t, epsilon_s, eta_t and eta_s) generated by gen_RAN (epsilon, eta, delta, 0., 0, rseed) with random seeds (rseed),
        //a delta in (-1,1), an exponent for the Euclidean distance (expn in (-1,1)), a 5%- and 10%- asymptotic critical values (asymp_CV),  a kernel weight (kernel),
        //a data-generating process (gen_DGP), a conditional mean function (cmean), and an OLS estimator (est_DGP).
        //OUTPUT: 5%- and 10%- sizes (asymp_REJF) and empirical critical values (empir_CV).
        template <double kernel (double), void gen_DGP (Matrix &, Matrix &, const Matrix, const Matrix, const double, const double, const int, unsigned long),
		          void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
        static void cValue (Matrix &asymp_REJF, Matrix &empir_CV, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                                         const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta_t, const Matrix &eta_s, const double delta,
                                         const double expn, const Matrix &asymp_CV, unsigned long seed, ofstream &size_out);
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

/* Calculate 5%- and 10%- asymptotic sizes and empirical critical values for Wang et al.'s (2021) HSIC-based test, El Himdi and Roy's (1997) test,
Bouhaddioui and Roy's (2006) test, and Tchahou and Duchesne's (2013) L1 test.
OUTPUT: 5%- and 10%- sizes (asymp_REJF) and empirical critical values (empir_CV) */
template<	//![simulate data]
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),

			//![set parameters for the HSIC test statistics]
			double kernel1(SGVector<double>, SGVector<double>), /*a kernel function for the first process*/
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
										double  /*a finite differential level*/),
			//![kernel function for Hong-type tests]
			double kernel(double) >
void Others_power::cValue (	Matrix &HSIC_J1_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							Matrix &HSIC_J2_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							Matrix &ER_asymp_REJF, /*asymptotic rejection rates of El Himdi and Roy's (1997) test*/
							Matrix &ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							Matrix &BR_asymp_REJF, /*asymptotic rejection rates of Bouhaddioui and Roy's (2006) test*/
							Matrix &BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							Matrix &RbF_asymp_REJF, /*asymptotic rejection rates of Robbins and Fisher's (2015) test*/
							Matrix &RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							Matrix &TD_L1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							Matrix &TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							Matrix &TD_T1_asymp_REJF, /*asymptotic rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							Matrix &TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
							const SGVector<double> &theta01, const SGVector<double> &theta02, /*initial values used to estimate DGPs*/
							int number_sampl, /*number of random samples drawn*/
							const int T, /*sample size*/
							const int lag_smooth, /*kernel bandwidth*/
							const int num_bts, /*number of bootstrap samples*/
							const Matrix &ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							const Matrix &BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							const Matrix &RbF_asymp_CV, /*5% and 10% asymptotic critical values of and Robbins and Fisher's (2015) test*/
							const Matrix &TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							const Matrix &TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							double h, /*a finite differential factor*/
							unsigned long seed,
							ofstream &size_out,
							bool use_bartlett_ker) {
	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

    cout << "Calculating empirical critical values ..." << endl;

    int HSIC_J1_REJ_5 = 0, HSIC_J1_REJ_10 = 0, HSIC_J2_REJ_5 = 0, HSIC_J2_REJ_10 = 0, ER_asymp_REJ_5 = 0, ER_asymp_REJ_10 = 0, BR_asymp_REJ_5 = 0, \
		BR_asymp_REJ_10 = 0, RbF_asymp_REJ_5 = 0, RbF_asymp_REJ_10 = 0, TD_L1_asymp_REJ_5 = 0, TD_L1_asymp_REJ_10 = 0, TD_T1_asymp_REJ_5 = 0, TD_T1_asymp_REJ_10 = 0;
    int i = 1, skip = 0, choose_alt = 0;
    SGMatrix<double> X(T,2), Y(T,2), eta1(T,2), eta2(T,2);
    SGVector<double> theta_est1(theta01.vlen), theta_est2(theta02.vlen);
    double fmin1 = 0., fmin2 = 0.;
    Matrix J1_pvalue(number_sampl, 1), J2_pvalue(number_sampl, 1), ER_stat(number_sampl, 1), BR_stat(number_sampl, 1), \
														RbF_stat(number_sampl, 1), TD_L1_stat(number_sampl, 1), TD_T1_stat(number_sampl, 1);
	J1_pvalue.set(0.);
	J2_pvalue.set(0.);

    size_out << std::fixed << std::setprecision(5);
    size_out << "J1_pvalue" << " , " << "J2_pvalue" << " , " << "ER_stat" << " , " << "BR_stat" << " , " << "RbF_stat" << " , " << "TD_L1_stat" << " , "
																				<< "TD_T1_stat" << endl;

	#pragma omp parallel for default(shared) reduction (+:skip,HSIC_J1_REJ_5,HSIC_J1_REJ_10,HSIC_J2_REJ_5,HSIC_J2_REJ_10,ER_asymp_REJ_5,ER_asymp_REJ_10,\
														BR_asymp_REJ_5,BR_asymp_REJ_10,RbF_asymp_REJ_5,RbF_asymp_REJ_10,TD_L1_asymp_REJ_5,\
														TD_L1_asymp_REJ_10,TD_T1_asymp_REJ_5,TD_T1_asymp_REJ_10) schedule(dynamic,CHUNK) private(i) \
														firstprivate(rseed,fmin1,fmin2)
	for (i = 1; i <= number_sampl; i++) {
		SGMatrix<double> X(T,2), Y(T,2), eta1(T,2), eta2(T,2);
		SGVector<double> theta_est1(theta01.vlen), theta_est2(theta02.vlen);
		//#pragma omp critical
		{
			rseed = gsl_rng_get (r);

			/* Generate random samples */
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);

			/* Compute residuals of each model */
			fmin1 = estimator(eta1, theta_est1, X, theta01);
			fmin2 = estimator(eta2, theta_est2, Y, theta02);
		}

		if (use_bartlett_ker == true) {
			/* Calculate the HSIC test pvalues */
			std::tie( J1_pvalue(i), J2_pvalue(i) ) =  HSIC::resid_bootstrap_pvalue<kernel1, kernel2, estimator, gen_data, resid, gradient, hessian>(X, Y, \
																											theta01, theta02, lag_smooth, num_bts, h, seed);

			//cout << J1_pvalue(i) << " , " << J2_pvalue(i) << endl;

			if (Math::fequals_abs(J1_pvalue(i), 2., 1e-2) == true && Math::fequals_abs(J2_pvalue(i), 2., 1e-2) == true) {
				skip += 1; // skip an iteration when the positivity constraint in the CC-MGARCH model is violated
			}
		}
		else {
			if (i == 1)
				size_out << "********************* The HSIC-based test statistics are not calculated for this lag-smoothing kernel ******************" << endl;
		}

		/* Compute El Himdi and Roy's (1997), Bouhaddioui and Roy's (2006), and Robbins and Fisher's (2015) test statistics */
		std::tie( ER_stat(i), BR_stat(i), RbF_stat(i) ) = Dep_tests::do_ElHimdiBouhaddiouiRoy<kernel>(lag_smooth, lag_smooth, eta1, eta2);

		/*	Compute Tchahou and Duchesne's (2013) test statistic */
		std::tie( TD_L1_stat(i), TD_T1_stat(i) ) = Dep_tests::do_TchahouDuchesne(lag_smooth, eta1, eta2);


		if (J1_pvalue(i) <= 0.05) ++HSIC_J1_REJ_5;  //using 5%-critical value
		if (J1_pvalue(i) <= 0.10) ++HSIC_J1_REJ_10; //using 10%-critical value
		if (J2_pvalue(i) <= 0.05) ++HSIC_J2_REJ_5;  //using 5%-critical value
		if (J2_pvalue(i) <= 0.10) ++HSIC_J2_REJ_10; //using 10%-critical value

		if ( ER_stat(i) >= ER_asymp_CV(1) ) ++ER_asymp_REJ_5; //using 5%-critical value
		if ( ER_stat(i) >= ER_asymp_CV(2) ) ++ER_asymp_REJ_10; //using 10%-critical value
		if ( BR_stat(i) >= BR_asymp_CV(1) ) ++BR_asymp_REJ_5; //using 5%-critical value
		if ( BR_stat(i) >= BR_asymp_CV(2) ) ++BR_asymp_REJ_10; //using 10%-critical value
		if ( RbF_stat(i) >= RbF_asymp_CV(1) ) ++RbF_asymp_REJ_5; //using 5%-critical value
		if ( RbF_stat(i) >= RbF_asymp_CV(2) ) ++RbF_asymp_REJ_10; //using 10%-critical value


		if ( TD_L1_stat(i) >= TD_L1_asymp_CV(1) ) ++TD_L1_asymp_REJ_5; //using 5%-critical value
		if ( TD_L1_stat(i) >= TD_L1_asymp_CV(2) ) ++TD_L1_asymp_REJ_10; //using 10%-critical value
		if ( TD_T1_stat(i) >= TD_T1_asymp_CV(1) ) ++TD_T1_asymp_REJ_5; //using 5%-critical value
		if ( TD_T1_stat(i) >= TD_T1_asymp_CV(2) ) ++TD_T1_asymp_REJ_10; //using 10%-critical value

		#pragma omp critical
		{
			size_out << J1_pvalue(i) << " , " << J2_pvalue(i) << " , " << ER_stat(i) << " , " << BR_stat(i) << " , " << RbF_stat(i) \
																		<< " , " << TD_L1_stat(i) << " , " << TD_T1_stat(i) << endl;
		}
	}

	number_sampl = number_sampl - skip;
	cout << "number_sampl = " << number_sampl << endl;

	size_out << "Wang et al.'s (2021) HSIC-based J1 test: " << endl;
	HSIC_J1_REJF(1) = ((double) HSIC_J1_REJ_5/number_sampl); //calculate sizes
	HSIC_J1_REJF(2) = ((double) HSIC_J1_REJ_10/number_sampl);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << HSIC_J1_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << HSIC_J1_REJF(2) << endl;

	size_out << "================================================================================================================================" << endl;

	size_out << "Wang et al.'s (2021) HSIC-based J2 test: " << endl;
	HSIC_J2_REJF(1) = ((double) HSIC_J2_REJ_5/number_sampl); //calculate sizes
	HSIC_J2_REJF(2) = ((double) HSIC_J2_REJ_10/number_sampl);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << HSIC_J2_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << HSIC_J2_REJF(2) << endl;

	size_out << "================================================================================================================================" << endl;

	size_out << "El Himdi and Roy's (1997) test: " << endl;
	ER_empir_CV(1) = quantile (ER_stat, 0.95); //calculate quantiles
	ER_empir_CV(2) = quantile (ER_stat, 0.90);
	ER_asymp_REJF(1) = ((double) ER_asymp_REJ_5/number_sampl); //calculate sizes
	ER_asymp_REJF(2) = ((double) ER_asymp_REJ_10/number_sampl);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << ER_asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << ER_asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << ER_empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << ER_empir_CV(2) << endl;

	size_out << "===============================================================================================================================" << endl;

	size_out << "Bouhaddioui and Roy's (2006) test: " << endl;
	BR_empir_CV(1) = quantile (BR_stat, 0.95); //calculate quantiles
	BR_empir_CV(2) = quantile (BR_stat, 0.90);
	BR_asymp_REJF(1) = ((double) BR_asymp_REJ_5/number_sampl); //calculate sizes
	BR_asymp_REJF(2) = ((double) BR_asymp_REJ_10/number_sampl);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << BR_asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << BR_asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << BR_empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << BR_empir_CV(2) << endl;

	size_out << "==============================================================================================================================" << endl;

	size_out << "Robbins and Fisher's (2015) test: " << endl;
	RbF_empir_CV(1) = quantile (RbF_stat, 0.95); //calculate quantiles
	RbF_empir_CV(2) = quantile (RbF_stat, 0.90);
	RbF_asymp_REJF(1) = ((double) RbF_asymp_REJ_5/number_sampl); //calculate sizes
	RbF_asymp_REJF(2) = ((double) RbF_asymp_REJ_10/number_sampl);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << RbF_asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << RbF_asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << RbF_empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << RbF_empir_CV(2) << endl;

	size_out << "==============================================================================================================================" << endl;

	size_out << "Tchahou and Duchesne's (2013) L1 test: " << endl;
	TD_L1_empir_CV(1) = quantile (TD_L1_stat, 0.95); //calculate quantiles
	TD_L1_empir_CV(2) = quantile (TD_L1_stat, 0.90);
	TD_L1_asymp_REJF(1) = ((double) TD_L1_asymp_REJ_5/number_sampl); //calculate sizes
	TD_L1_asymp_REJF(2) = ((double) TD_L1_asymp_REJ_10/number_sampl);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << TD_L1_asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << TD_L1_asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << TD_L1_empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << TD_L1_empir_CV(2) << endl;

	size_out << "==============================================================================================================================" << endl;

	size_out << "Tchahou and Duchesne's (2013) T1 test: " << endl;
	TD_T1_empir_CV(1) = quantile (TD_T1_stat, 0.95); //calculate quantiles
	TD_T1_empir_CV(2) = quantile (TD_T1_stat, 0.90);
	TD_T1_asymp_REJF(1) = ((double) TD_T1_asymp_REJ_5/number_sampl); //calculate sizes
	TD_T1_asymp_REJF(2) = ((double) TD_T1_asymp_REJ_10/number_sampl);
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-asymptotic size for T =" << T << " is " << TD_T1_asymp_REJF(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-asymptotic size for T = " << T << " is " << TD_T1_asymp_REJF(2) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "5%-empirical critical value for T =" << T << " is " << TD_T1_empir_CV(1) << endl;
	size_out << "T = " << T << " and M_T = " << lag_smooth  << ": " << "10%-empirical critical value for T = " << T << " is " << TD_T1_empir_CV(2) << endl;

	gsl_rng_free (r); // free up memory
}


/* Calculate 5% and 10% empirical and asymptotic rejection frequencies at 5% and 10% empirical critical values (empir_CV) and
5% and 10% asymptotic critical values (asymp_CV) Wang et al.'s (2021) HSIC-based test, El Himdi and Roy's (1997) test,
Bouhaddioui and Roy's (2006) test, and Tchahou and Duchesne's (2013) L1 test.
OUTPUT: 5% and 10% empirical rejection frequencies (empir_REJF) and asymptotic rejection frequencies (asymp_REJF). */
template<	//![simulate data]
			void gen_DGP(	SGMatrix<double> &, SGMatrix<double> &, /*T by 2 matrices of realizations*/
							const SGVector<double>, /*a vector of parameters for the first process*/
							const SGVector<double>, /*a vector of parameters for the second process*/
							const int, /*select how the innovations are generated*/
							unsigned long /*a seed used to generate random numbers*/),

			//![set parameters for the HSIC test statistics]
			double kernel1(SGVector<double>, SGVector<double>), /*a kernel function for the first process*/
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
										double  /*a finite differential level*/),
			//![kernel function for Hong-type tests]
			double kernel(double) >
void Others_power::power_f (Matrix &HSIC_J1_empir_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J1 test*/
							Matrix &HSIC_J2_empir_REJF, /*rejection rates of Wang et al.'s (2021) HSIC-based J2 test*/
							Matrix &ER_empir_REJF, Matrix &ER_asymp_REJF, /*rejection rates of El Himdi and Roy's (1997) test*/
							Matrix &BR_empir_REJF, Matrix &BR_asymp_REJF, /*rejection rates of Bouhaddioui and Roy's (2006) test*/
							Matrix &RbF_empir_REJF, Matrix &RbF_asymp_REJF, /*rejection rates of and Robbins and Fisher's (2015) test*/
							Matrix &TD_L1_empir_REJF, Matrix &TD_L1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) L1 test*/
							Matrix &TD_T1_empir_REJF, Matrix &TD_T1_asymp_REJF, /*rejection rates of Tchahou and Duchesne's (2013) T1 test*/
							const SGVector<double> &theta1, const SGVector<double> &theta2, /*true parameters of two DGPs*/
							const SGVector<double> &theta01, const SGVector<double> &theta02, /*initial values used to estimate DGPs*/
							int number_sampl, /*number of random samples drawn*/
							const int T, /*sample size*/
							const int lag_smooth, /*kernel bandwidth*/
							int num_bts, /*number of bootstrap samples*/
							const Matrix &ER_empir_CV, /*5% and 10% empirical critical values of El Himdi and Roy's (1997) test*/
							const Matrix &ER_asymp_CV, /*5% and 10% asymptotic critical values of El Himdi and Roy's (1997) test*/
							const Matrix &BR_empir_CV, /*5% and 10% empirical critical values of Bouhaddioui and Roy's (2006) test*/
							const Matrix &BR_asymp_CV, /*5% and 10% asymptotic critical values of Bouhaddioui and Roy's (2006) test*/
							const Matrix &RbF_empir_CV, /*5% and 10% empirical critical values of Robbins and Fisher's (2015) test*/
							const Matrix &RbF_asymp_CV, /*5% and 10% asymptotic critical values of Robbins and Fisher's (2015) test*/
							const Matrix &TD_L1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) L1 test*/
							const Matrix &TD_L1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) L1 test*/
							const Matrix &TD_T1_empir_CV, /*5% and 10% empirical critical values of Tchahou and Duchesne's (2013) T1 test*/
							const Matrix &TD_T1_asymp_CV, /*5% and 10% asymptotic critical values of Tchahou and Duchesne's (2013) T1 test*/
							const int choose_alt, /*set a degree of dependence between two DGPs*/
							double h, /*a finite differential factor*/
							unsigned long seed, /*seed for random number generator*/
							ofstream &pwr_out,
							bool use_bartlett_ker)  {
	gsl_rng * r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long int rseed = 1;

	cout << "Calculating rejection frequencies for T = " << T << endl;

	int HSIC_J1_REJ_5 = 0, HSIC_J1_REJ_10 = 0, HSIC_J2_REJ_5 = 0, HSIC_J2_REJ_10 = 0, ER_asymp_REJ_5 = 0, ER_asymp_REJ_10 = 0, \
		BR_asymp_REJ_5 = 0, BR_asymp_REJ_10 = 0, RbF_asymp_REJ_5 = 0, RbF_asymp_REJ_10 = 0, TD_L1_asymp_REJ_5 = 0, TD_L1_asymp_REJ_10 = 0, \
		TD_T1_asymp_REJ_5 = 0, TD_T1_asymp_REJ_10 = 0;
	int ER_empir_REJ_5 = 0, ER_empir_REJ_10 = 0, BR_empir_REJ_5 = 0, BR_empir_REJ_10 = 0, RbF_empir_REJ_5 = 0, RbF_empir_REJ_10 = 0, \
											TD_L1_empir_REJ_5 = 0, TD_L1_empir_REJ_10 = 0, TD_T1_empir_REJ_5 = 0, TD_T1_empir_REJ_10 = 0;
    int i = 0, skip = 0;
    SGMatrix<double> X(T,2), Y(T,2), eta1(T,2), eta2(T,2);
    SGVector<double> theta_est1(theta01.vlen), theta_est2(theta02.vlen);
    double fmin1 = 0., fmin2 = 0.;
    Matrix J1_pvalue(number_sampl, 1), J2_pvalue(number_sampl, 1), ER_stat(number_sampl, 1), BR_stat(number_sampl, 1), \
														RbF_stat(number_sampl, 1), TD_L1_stat(number_sampl, 1), TD_T1_stat(number_sampl, 1);
	J1_pvalue.set(0.);
	J2_pvalue.set(0.);

    pwr_out << std::fixed << std::setprecision(5);
    pwr_out << "J1_pvalue" << " , " << "J2_pvalue" << " , " << "ER_stat" << " , " << "BR_stat" << " , " << "RbF_stat" << " , " \
																						<< "TD_L1_stat" << " , " << "TD_T1_stat" << endl;


	#pragma omp parallel for default(shared) reduction(+:skip,HSIC_J1_REJ_5,HSIC_J1_REJ_10,HSIC_J2_REJ_5,HSIC_J2_REJ_10,ER_asymp_REJ_5,ER_asymp_REJ_10,\
														BR_asymp_REJ_5,BR_asymp_REJ_10,RbF_asymp_REJ_5,RbF_asymp_REJ_10,TD_L1_asymp_REJ_5,TD_L1_asymp_REJ_10,\
														TD_T1_asymp_REJ_5,TD_T1_asymp_REJ_10,ER_empir_REJ_5,ER_empir_REJ_10,BR_empir_REJ_5,BR_empir_REJ_10,\
														RbF_empir_REJ_5,RbF_empir_REJ_10,TD_L1_empir_REJ_5,TD_L1_empir_REJ_10,TD_T1_empir_REJ_5,\
														TD_T1_empir_REJ_10) \
														schedule(dynamic,CHUNK) private(i) firstprivate(rseed,fmin1,fmin2)
	for (i = 1; i <= number_sampl; ++i) {
		SGMatrix<double> X(T,2), Y(T,2), eta1(T,2), eta2(T,2);
		SGVector<double> theta_est1(theta01.vlen), theta_est2(theta02.vlen);
		//#pragma omp critical
		{
			rseed = gsl_rng_get (r); //set a random seed for the random number generator

					/* Generate random samples */
			gen_DGP(X, Y, theta1, theta2, choose_alt, rseed);

			 /* Compute residuals of each model */
			fmin1 = estimator(eta1, theta_est1, X, theta01);
			fmin2 = estimator(eta2, theta_est2, Y, theta02);
		}

		if (use_bartlett_ker == true) {
			/* Calculate the HSIC test pvalues */
			std::tie( J1_pvalue(i), J2_pvalue(i) ) =  HSIC::resid_bootstrap_pvalue<kernel1, kernel2, estimator, gen_data, resid, gradient, hessian>(X, Y, \
																												theta01, theta02, lag_smooth, num_bts, h, seed);

			//cout << J1_pvalue(i) << " , " << J2_pvalue(i) << endl;

			if (Math::fequals_abs(J1_pvalue(i), 2., 1e-2) == true && Math::fequals_abs(J2_pvalue(i), 2., 1e-2) == true) {
				skip += 1; // skip an iteration when the positivity constraint in the CC-MGARCH model is violated
			}
		}
		else {
			if (i == 1)
				pwr_out << "********************* The HSIC-based test statistics are not calculated for this lag-smoothing kernel ******************" << endl;
		}

		if (J1_pvalue(i) <= 0.05) ++HSIC_J1_REJ_5;  //using 5%-critical value
        if (J1_pvalue(i) <= 0.10) ++HSIC_J1_REJ_10; //using 10%-critical value
        if (J2_pvalue(i) <= 0.05) ++HSIC_J2_REJ_5;  //using 5%-critical value
        if (J2_pvalue(i) <= 0.10) ++HSIC_J2_REJ_10; //using 10%-critical value


		/* Compute El Himdi and Roy's (1997), Bouhaddioui and Roy's (2006), and Robbins and Fisher's (2015) test statistics */
		std::tie( ER_stat(i), BR_stat(i), RbF_stat(i) ) = Dep_tests::do_ElHimdiBouhaddiouiRoy<kernel>(lag_smooth, lag_smooth, eta1, eta2);
		if ( ER_stat(i) >= ER_empir_CV(1) ) ++ER_empir_REJ_5; //using 5%-critical value
		if ( ER_stat(i) >= ER_asymp_CV(1) ) ++ER_asymp_REJ_5; //using 5%-critical value
		if ( ER_stat(i) >= ER_empir_CV(2) ) ++ER_empir_REJ_10; //using 10%-critical value
		if ( ER_stat(i) >= ER_asymp_CV(2) ) ++ER_asymp_REJ_10; //using 10%-critical value

		if ( BR_stat(i) >= BR_empir_CV(1) ) ++BR_empir_REJ_5; //using 5%-critical value
		if ( BR_stat(i) >= BR_asymp_CV(1) ) ++BR_asymp_REJ_5; //using 5%-critical value
		if ( BR_stat(i) >= BR_empir_CV(2) ) ++BR_empir_REJ_10; //using 10%-critical value
		if ( BR_stat(i) >= BR_asymp_CV(2) ) ++BR_asymp_REJ_10; //using 10%-critical value

		if ( RbF_stat(i) >= RbF_empir_CV(1) ) ++RbF_empir_REJ_5; //using 5%-critical value
		if ( RbF_stat(i) >= RbF_asymp_CV(1) ) ++RbF_asymp_REJ_5; //using 5%-critical value
		if ( RbF_stat(i) >= RbF_empir_CV(2) ) ++RbF_empir_REJ_10; //using 10%-critical value
		if ( RbF_stat(i) >= RbF_asymp_CV(2) ) ++RbF_asymp_REJ_10; //using 10%-critical value


		/*	Compute Tchahou and Duchesne's (2013) test statistic */
		std::tie( TD_L1_stat(i), TD_T1_stat(i) ) = Dep_tests::do_TchahouDuchesne(lag_smooth, eta1, eta2);
		if ( TD_L1_stat(i) >= TD_L1_empir_CV(1) ) ++TD_L1_empir_REJ_5; //using 5%-critical value
		if ( TD_L1_stat(i) >= TD_L1_asymp_CV(1) ) ++TD_L1_asymp_REJ_5; //using 5%-critical value
		if ( TD_L1_stat(i) >= TD_L1_empir_CV(2) ) ++TD_L1_empir_REJ_10; //using 10%-critical value
		if ( TD_L1_stat(i) >= TD_L1_asymp_CV(2) ) ++TD_L1_asymp_REJ_10; //using 10%-critical value

		if ( TD_T1_stat(i) >= TD_T1_empir_CV(1) ) ++TD_T1_empir_REJ_5; //using 5%-critical value
		if ( TD_T1_stat(i) >= TD_T1_asymp_CV(1) ) ++TD_T1_asymp_REJ_5; //using 5%-critical value
		if ( TD_T1_stat(i) >= TD_T1_empir_CV(2) ) ++TD_T1_empir_REJ_10; //using 10%-critical value
		if ( TD_T1_stat(i) >= TD_T1_asymp_CV(2) ) ++TD_T1_asymp_REJ_10; //using 10%-critical value

		#pragma omp critical
		{
			pwr_out << J1_pvalue(i) << " , " << J2_pvalue(i) << " , " << ER_stat(i) << " , " << BR_stat(i) << " , " \
											<< RbF_stat(i) << " , " << TD_L1_stat(i) << " , " << TD_T1_stat(i) << endl;
		}
	}

	number_sampl = number_sampl - skip;
	cout << "number_sampl = " << number_sampl << endl;

	pwr_out << "Wang et al.'s (2021) HSIC-based J1 test: " << endl;
	HSIC_J1_empir_REJF(1) = ((double) HSIC_J1_REJ_5/number_sampl); //calculate sizes
	HSIC_J1_empir_REJF(2) = ((double) HSIC_J1_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << HSIC_J1_empir_REJF(1) << " and " << HSIC_J1_empir_REJF(2) << endl;

	pwr_out << "================================================================================================================================" << endl;

	pwr_out << "Wang et al.'s (2021) HSIC-based J2 test: " << endl;
	HSIC_J2_empir_REJF(1) = ((double) HSIC_J2_REJ_5/number_sampl); //calculate sizes
	HSIC_J2_empir_REJF(2) = ((double) HSIC_J2_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << HSIC_J2_empir_REJF(1) << " and " << HSIC_J2_empir_REJF(2) << endl;

	pwr_out << "================================================================================================================================" << endl;


	pwr_out << "El Himdi and Roy's (1997) test: " << endl;
	ER_empir_REJF(1) = ((double) ER_empir_REJ_5/number_sampl); //calculate sizes
	ER_empir_REJF(2) = ((double) ER_empir_REJ_10/number_sampl);
	ER_asymp_REJF(1) = ((double) ER_asymp_REJ_5/number_sampl); //calculate sizes
	ER_asymp_REJF(2) = ((double) ER_asymp_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << ER_empir_REJF(1) << " and " << ER_empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << ER_asymp_REJF(1) << " and " << ER_asymp_REJF(2) << endl;

	pwr_out << "===============================================================================================================================" << endl;

	pwr_out << "Bouhaddioui and Roy's (2006) test: " << endl;
	BR_empir_REJF(1) = ((double) BR_empir_REJ_5/number_sampl); //calculate sizes
	BR_empir_REJF(2) = ((double) BR_empir_REJ_10/number_sampl);
	BR_asymp_REJF(1) = ((double) BR_asymp_REJ_5/number_sampl); //calculate sizes
	BR_asymp_REJF(2) = ((double) BR_asymp_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << BR_empir_REJF(1) << " and " << BR_empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << BR_asymp_REJF(1) << " and " << BR_asymp_REJF(2) << endl;

	pwr_out << "==============================================================================================================================" << endl;

	pwr_out << "Robbins and Fisher's (2015) test: " << endl;
	RbF_empir_REJF(1) = ((double) RbF_empir_REJ_5/number_sampl); //calculate sizes
	RbF_empir_REJF(2) = ((double) RbF_empir_REJ_10/number_sampl);
	RbF_asymp_REJF(1) = ((double) RbF_asymp_REJ_5/number_sampl); //calculate sizes
	RbF_asymp_REJF(2) = ((double) RbF_asymp_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << RbF_empir_REJF(1) << " and " << RbF_empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << RbF_asymp_REJF(1) << " and " << RbF_asymp_REJF(2) << endl;

	pwr_out << "==============================================================================================================================" << endl;

	pwr_out << "Tchahou and Duchesne's (2013) L1 test: " << endl;
	TD_L1_empir_REJF(1) = ((double) TD_L1_empir_REJ_5/number_sampl); //calculate sizes
	TD_L1_empir_REJF(2) = ((double) TD_L1_empir_REJ_10/number_sampl);
	TD_L1_asymp_REJF(1) = ((double) TD_L1_asymp_REJ_5/number_sampl); //calculate sizes
	TD_L1_asymp_REJF(2) = ((double) TD_L1_asymp_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << TD_L1_empir_REJF(1) << " and " << TD_L1_empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << TD_L1_asymp_REJF(1) << " and " << TD_L1_asymp_REJF(2) << endl;

	pwr_out << "==============================================================================================================================" << endl;

	pwr_out << "Tchahou and Duchesne's (2013) T1 test: " << endl;
	TD_T1_empir_REJF(1) = ((double) TD_T1_empir_REJ_5/number_sampl); //calculate sizes
	TD_T1_empir_REJF(2) = ((double) TD_T1_empir_REJ_10/number_sampl);
	TD_T1_asymp_REJF(1) = ((double) TD_T1_asymp_REJ_5/number_sampl); //calculate sizes
	TD_T1_asymp_REJF(2) = ((double) TD_T1_asymp_REJ_10/number_sampl);
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% empirical reject frequencies for choose_alt = " << choose_alt << " are "
	        << TD_T1_empir_REJF(1) << " and " << TD_T1_empir_REJF(2) << endl;
	pwr_out << "T = " << T << " and M_T = " << lag_smooth << ": 5% and 10% asymptotic reject frequencies for choose_alt = " << choose_alt << " are "
	        << TD_T1_asymp_REJF(1) << " and " << TD_T1_asymp_REJF(2) << endl;

	gsl_rng_free (r); // free up memory
}



//calculate 5%- and 10%- asymptotic sizes and empirical critical values when each X or Y is generated by a non-Gaussian AR(2) process. INPUT: number of random samples
//generated (number_sampl), a sample size (T), a lag truncation (TL), a lag-smoothing parameter (lag_smooth), 6x1 vectors of AR coefficients (alpha_X and alpha_Y),
//vectors of random errors (epsilon_t, epsilon_s, eta_t and eta_s) generated by gen_RAN (epsilon, eta, delta, 0., 0, rseed) with random seeds (rseed),
//a delta in (-1,1), an exponent for the Euclidean distance (expn in (-1,1)), a 5%- and 10%- asymptotic critical values (asymp_CV),  a kernel weight (kernel),
//a data-generating process (gen_DGP), a conditional mean function (cmean), and an OLS estimator (est_DGP).
//OUTPUT: 5%- and 10%- sizes (asymp_REJF) and empirical critical values (empir_CV).
template <double kernel (double), void gen_DGP (Matrix &, Matrix &, const Matrix, const Matrix, const double, const double, const int, unsigned long),
		  void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
void Others_power::cValue (Matrix &asymp_REJF, Matrix &empir_CV, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                                          const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta_t, const Matrix &eta_s, const double delta,
					                 const double expn, const Matrix &asymp_CV, unsigned long seed, ofstream &size_out) {
	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long rseed = 1;
    //calculate integrals of quadratic and quartic functions of the kernel weight
	auto kernel_QDSum = 0., kernel_QRSum = 0.;
	NGDist_corr::integrate_Kernel <kernel> (kernel_QDSum, kernel_QRSum);
	cout << "kernel_QDSum and kernel_QRSum =" << kernel_QDSum << " , " << kernel_QRSum << endl;
    cout << "Calculating empirical critical values ..." << endl;
    int i = 1, size_alpha_X = alpha_X.nRow(), size_alpha_Y = alpha_Y.nRow();
    Matrix X(T,1), Y(T,1), alpha_X_hat(size_alpha_X,1), alpha_Y_hat(size_alpha_Y,1), resid_X(T-2,1), resid_Y(T-2,1), tvalue(number_sampl,1);
    int asymp_REJ_5 = 0, asymp_REJ_10 = 0;
	#pragma omp parallel for default(shared) reduction (+:asymp_REJ_5,asymp_REJ_10) schedule(dynamic,CHUNK) private(i) \
                                                                                                                                                                     firstprivate(rseed,X,Y,resid_X,resid_Y,alpha_X_hat,alpha_Y_hat)
	for (i = 1; i <= number_sampl; i++) {
		rseed = gsl_rng_get (r);
		gen_DGP (X, Y, alpha_X, alpha_Y, delta, 0., 0, rseed); //draw two independent random samples from the dgp of X and Y each using a random seed
		//then use these samples to estimate the AR coefficients
		est_DGP (resid_X, alpha_X_hat, X);
		est_DGP (resid_Y, alpha_Y_hat, Y);
		//cout << alpha_X_hat(1) << " , " << alpha_Y_hat(1) << endl;
		//calculate the test statistic
		tvalue(i) = NGDist_corr::do_Test<kernel, cmean> (X, Y, TL, lag_smooth, kernel_QRSum, alpha_X_hat, alpha_Y_hat, epsilon_t, epsilon_s, eta_t, eta_s, expn);
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

//calculate 5%- and 10%- asymptotic sizes and empirical critical values when X and Y are generated by a univariate non-Gaussian AR(2) process
//and a bivariate non-Gaussian AR(2) process respectively. INPUT: number of random samples generated (number_sampl), a sample size (T), a lag truncation (TL),
//a lag-smoothing parameter (lag_smooth), A 6x1 vector of coefficients for X (alpha_X), a 6x2 matrix of coefficients for Y1 and Y2 (alpha_Y),
//Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed)
//with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, an exponent for the Euclidean distance (expn in (1,2)), and a 5%- and 10%- asymptotic critical
//values (asymp_CV), a kernel weight (kernel), a data-generating process (gen_DGP), a conditional mean function (cmean), an OLS estimator (est_DGP).
//OUTPUT: 5%- and 10%- sizes (asymp_REJF) and empirical critical values (empir_CV).
template <double kernel (double), void gen_DGP (Matrix &, Matrix &, Matrix &, const Matrix, const Matrix, const Matrix &, const double, const double, const int, unsigned long),
		  void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &), void est_DGP (Matrix &, Matrix &, const Matrix)>
void Others_power::cValue (Matrix &asymp_REJF, Matrix &empir_CV, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                      const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t,
					  const Matrix &eta2_s, const Matrix &delta, const double expn, const Matrix &asymp_CV, unsigned long seed, ofstream &size_out) {
	gsl_rng *r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
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
void Others_power::power_f (Matrix &empir_REJF, Matrix &asymp_REJF, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                       const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta_t, const Matrix &eta_s, const double delta,
					   const double rho, const double expn, const Matrix &empir_CV, const Matrix &asymp_CV, const int choose_alt, unsigned long seed,
					   ofstream &pwr_out)  {
	gsl_rng * r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
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
void Others_power::power_f (Matrix &empir_REJF, Matrix &asymp_REJF, const int number_sampl, const int T, const int TL, const int lag_smooth, const Matrix &alpha_X,
                                            const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t,
                                            const Matrix &eta2_s, const Matrix &delta, const double rho12, const double rho13, const double expn, const Matrix &empir_CV, const Matrix &asymp_CV,
                                            const int choose_alt, unsigned long seed, ofstream &pwr_out)  {
	gsl_rng * r = nullptr;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
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
void Others_power::power_f (Matrix &bootstrap_REJF, const int number_sampl, const int TL, const int lag_smooth, const Matrix &alpha_X, const Matrix &alpha_Y,
								    const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t, const Matrix &eta2_s,
								    const Matrix &xi_x, const Matrix &xi_y, const Matrix &delta, const double rho12, const double rho13, const double expn, const int choose_alt,
								    unsigned long seed, ofstream &pwr_out)  {
	gsl_rng * r = nullptr;
	const gsl_rng_type * gen;//random number generator
	gsl_rng_env_setup();
	gen = gsl_rng_default;
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
