#ifndef NL_DGP_H_
#define NL_DGP_H_

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sf_expint.h>
#include <asserts.h>


#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/util/factory.h>

using namespace std;
using namespace shogun;
using namespace shogun::linalg;

class NL_Dgp {
	public:
		NL_Dgp () {   }; //default constructor
		~NL_Dgp () {   };//default destructor

        //generate TWO threshold AR processes of the second order. INPUT: A 5x1 vector of coefficients for X (alpha_X), a 5x1 vector of coefficients for Y (alpha_Y),
        //and a template function to generate two random errors for both the processes (gen_RAN). OUTPUT: Tx1 matrices (X and Y).
        template <void gen_RAN (double &, double &, const double, const double, const int, unsigned long)>
        static void gen_TAR (Matrix &X, Matrix &Y, const Matrix alpha_X, const Matrix alpha_Y, const double delta, const double rho, const int choose_alt,
		                     unsigned long seed);
		//generate a univariate threshold AR process and a bivariate threshold AR process of the second orders.
        //INPUT: A 5x1 vector of coefficients for X (alpha_X), a 5x2 matrix of coefficients for Y1 and Y2 (alpha_Y),
        //and a template function to generate two random errors for both the processes (gen_RAN). OUTPUT: a Tx1 vector (X) and Tx1 vectors (Y1 and Y2).
        template <void gen_RAN (double &, double &, double &, const Matrix &, const double, const double, const int, unsigned long)>
        static void gen_TAR (Matrix &X, Matrix &Y1, Matrix &Y2, const Matrix alpha_X, const Matrix alpha_Y, const Matrix &delta, const double alpha5,
                             const double rho1, const int choose_alt, unsigned long seed);
        //generate TWO bilinear processes of the second order. INPUT: A 6x1 vector of coefficients for X (alpha_X), a 6x1 vector of coefficients for Y (alpha_Y),
        //and a template function to generate two random errors for both the processes (gen_RAN). OUTPUT: Tx1 matrices (X and Y).
        template <void gen_RAN (double &, double &, const double, const double, const int, unsigned long)>
        static void gen_Bilinear (Matrix &X, Matrix &Y, const Matrix alpha_X, const Matrix alpha_Y, const double delta, const double rho, const int choose_alt,
		                          unsigned long seed);
		//generate a univariate bilinear process and a bivariate bilinear process of the second orders that may have some dependency.
        //INPUT: A 6x1 vector of coefficients for X (alpha_X), a 6x2 matrix of coefficients for Y1 and Y2 (alpha_Y), and a template function
        //to generate random errors for each individual process (gen_RAN). OUTPUT: a Tx1 vector (X) and Tx1 vectors (Y1 and Y2).
        template <void gen_RAN (double &, double &, double &, const Matrix &, const double, const double, const int, unsigned long)>
        static void gen_Bilinear (Matrix &X, Matrix &Y1, Matrix &Y2, const Matrix alpha_X, const Matrix alpha_Y, const Matrix &delta, const double alpha5,
		                          const double rho1, const int choose_alt, unsigned long seed);
        //generate centered skew-normal random error terms. INPUT: delta in (-1,1), a correlation between xi_1_eps and xi_1_eta (rho), an alternative:
        //choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
        //dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: two random error terms (epsilon and eta).
        static void gen_SN (double &epsilon, double &eta, const double delta, const double rho, const int choose_alt, unsigned long seed);
        //generate a trivariate skew-normal random variable. INPUT: a 3x1 vector (delta) in (-1,1)^3, correlations (rho12 and rho13), an alternative:
        //set choose_alt = 0 to generate independent random variables, choose_alt = 1 to generate *epsilon uncorrelated and dependent with *eta1 and *eta2,
        //choose_alt = 2 to generate dependent random variables. OUTPUT: three random variables (epsilon, eta1, and eta2).
        static void gen_TriSN (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double rho12, const double rho13, const int choose_alt,
		                       unsigned long seed);
        //generate two mixtures of N(0,1) random variables. INPUT: a correlation between xi_1_eps and xi_1_eta (rho) and an alternative:
		//choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
        //dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: two random error terms (epsilon and eta).
        static void gen_MN (double &epsilon, double &eta, const double delta, const double rho, const int choose_alt, unsigned long seed);
        //generate three mixtures of N(0,1) random variables. INPUT: a correlation between xi_1_eps and xi_1_eta1 (rho) and an alternative:
        //choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
        //dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: three random error terms (epsilon, eta1, and eta2).
        static void gen_TriMN (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double alpha5, const double rho1, const int choose_alt,
                               unsigned long seed);
        //generate two centered chi-squared random variables. INPUT: a correlation between xi_1_eps and xi_1_eta (rho) and an alternative:
        //choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
        //dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: two random error terms (epsilon and eta).
        static void gen_ChiSq (double &epsilon, double &eta, const double delta, const double rho, const int choose_alt, unsigned long seed);
        //generate three centered chi-squared random variables. INPUT: a correlation between xi_1_eps and xi_1_eta1 (rho) and an alternative:
        //choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
        //dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: three random error terms (epsilon, eta1, and eta2).
        static void gen_TriChiSq (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double alpha5, const double rho1, const int choose_alt,
                                  unsigned long seed);
        //generate two centered Beta random variables. INPUT: a shape parameter for the bivariate Beta distribution (alpha5 = 0.0001 gives corr = -0.145, alpha5 = 0.1 gives corr = -0.005,
        //alpha5 = 1. gives corr = 0.365) and   choose_alt = 0 and 1. OUTPUT: two random error terms (epsilon and eta).
        static void gen_Beta (double &epsilon, double &eta, const double delta, const double alpha5, const int choose_alt, unsigned long seed);
        //generate three centered Beta random variables. INPUT: choose_alt = 0, 1, 2, and 3.
        //OUTPUT: three random error terms (epsilon, eta1, and eta2).
        static void gen_TriBeta (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double rho, const double rho1, const int choose_alt, unsigned long seed);
        //generate two centered exponential random variables. INPUT: an alternative: choose_alt = 0 to generate independent error terms,
        //choose_alt = 1 to generate very weakly dependent error terms, choose_alt = 2 to generate weak dependent error terms, choose_alt = 3 to generate dependent error terms.
        //OUTPUT: two random error terms (epsilon and eta).
        static void gen_Exp (double &epsilon, double &eta, const double delta, const double rho, const int choose_alt, unsigned long seed);
        //generate three centered exponential random variables. INPUT: an alternative: choose_alt = 0 to generate independent error terms,
        //choose_alt = 1 to generate very weakly dependent error terms, choose_alt = 2 to generate weak dependent error terms, choose_alt = 3 to generate dependent error terms.
        //OUTPUT: three random error terms (epsilon, eta1, and eta2).
        static void gen_TriExp (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double rho, const double rho1, const int choose_alt, unsigned long seed);
        //generate two vectors of random errors (epsilon and eta). INPUT: constants (delta and rho in (-1,1)), an alternative (choose_alt = 0, 1, 2), and a
        //random generator template (gen_RAN: gen_SN, gen_MN, gen_ChiSq or gen_Beta). OUTPUT: 2 vectors
        template <void gen_RAN (double &, double &, const double, const double, const int, unsigned long)>
        static void gen_RANV (Matrix &epsilon, Matrix &eta, const double delta, const double rho, const int choose_alt, unsigned long seed);
        //generate three vectors of random errors (epsilon, eta1 and eta2). INPUT: a vector (delta in (-1,1)^3) and constants (rho12 and rho13 in (-1,1)),
        //an alternative (choose_alt = 0, 1, 2), and a random generator template (gen_RAN: gen_TriSN, gen_TriMN, gen_TriChiSq or gen_TriBeta). OUTPUT: 3 vectors
        template <void gen_RAN (double &, double &, double &, const Matrix &, const double, const double, const int, unsigned long)>
        static void gen_RANV (Matrix &epsilon, Matrix &eta1, Matrix &eta2, const Matrix &delta, const double rho12, const double rho13, const int choose_alt,
                              unsigned long seed);

		/* Generate 6-dimensional centered normal random variables defined on page 31 in Wang, Li & Zhu (2021).
		OUTPUT: T independent realizations of two scalar random numbers (u1 and u2) and two 2 by 1 vectors (u3 and u4) */
		static std::tuple< SGVector<double>, SGVector<double>, SGMatrix<double>, SGMatrix<double> > gen_MNorm(	int T,
																												double rho1,
																												double rho4,
																												unsigned long seed);

		/* Generate data from a CC-MGARCH model as defined on page 367 in Tse (2002).
		OUTPUT: two T by 2 random matrices (Y1 and Y2) */
		static void gen_CC_MGARCH(	SGMatrix<double> &Y1, SGMatrix<double> &Y2, /*T by 2 matrices of realizations from the CC-GARCH*/
									const SGVector<double> theta1, /*a 7 by 1 vector of parameters for the first process*/
									const SGVector<double> theta2, /*a 7 by 1 vector of parameters for the second process*/
									const int choose_alt, /*select how the innovations are generated*/
									unsigned long seed);

		/* Calculate residuals from a CC-MGARCH process.
		OUTPUT: a T by 2 matrix of residuals. */
		static SGMatrix<double> resid_CC_MGARCH(const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
												SGVector<double> theta /*7 by 1 vector of parameters*/);

		/* Generate data from a CC-MGARCH process given a sample of innovations */
		static SGMatrix<double> gen_CC_MGARCH(	const SGMatrix<double> &eta, /*T by 2 matrix of innovations*/
												const SGVector<double> theta /*7 by 1 vector of parameters*/);

		/* Generate data from a VAR(1) defined in El Himdl and Roy (1997).
		OUTPUT: two T by 2 random matrices (Y1 and Y2) */
		static void gen_VAR(SGMatrix<double> &Y1, SGMatrix<double> &Y2, /*T by 2 matrices of realizations from two VAR(1) */
							const SGVector<double> theta1, /*a 4 by 1 vector of parameters for the first process*/
							const SGVector<double> theta2, /*a 4 by 1 vector of parameters for the second process*/
							const int choose_alt, /*select how the innovations are generated*/
							unsigned long seed);

		/* Generate data from a VAR(1) process given a sample of innovations */
		static SGMatrix<double> gen_VAR(const SGMatrix<double> &eta, /*T by 2 matrix of innovations*/
										const SGVector<double> theta /*4 by 1 vector of parameters*/);


		/* Generate data from a VAR-CC-MGARCH(1,1) model as defined on on page 367 in Tse (2002).
		OUTPUT: two T by 2 random matrices (Y1 and Y2) */
		static void gen_VAR_CC_MGARCH(	SGMatrix<double> &Y1, SGMatrix<double> &Y2, /*T by 2 matrices of realizations from the CC-GARCH*/
										const SGVector<double> theta1, /*a 11 by 1 vector of parameters for the first process*/
										const SGVector<double> theta2, /*a 11 by 1 vector of parameters for the second process*/
										const int choose_alt, /*select how the innovations are generated*/
										unsigned long seed);

		/* Generate data from a VAR-CC-MGARCH(1,1) process given a sample of innovations */
		static SGMatrix<double> gen_VAR_CC_MGARCH(	const SGMatrix<double> &eta, /*T by 2 matrix of innovations*/
													const SGVector<double> theta /*11 by 1 vector of parameters*/);

		/* Fit data to a VAR(p) model and get residuals */
		static SGMatrix<double>	resid_VAR(	SGMatrix<double> &Phi, /*dim by dim*p+1 matrix of parameter estimates*/
											const SGMatrix<double> &Y, /*T by dim matrix of observations*/
											int p /*maximum VAR lag*/);

		/* Calculate residuals from a VAR(1) process.
		OUTPUT: a T by 2 matrix of residuals. */
		static SGMatrix<double> resid_VAR1(	const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
											SGVector<double> theta /*4 by 1 vector of parameters*/);

		/* Calculate residuals from a VAR-CC-MGARCH(1,1) process.
		OUTPUT: a T by 2 matrix of residuals. */
		static SGMatrix<double> resid_VAR_CC_MGARCH(const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
													SGVector<double> theta /*11 by 1 vector of parameters*/);

		/* Estimate VAR(1) */
		static double estimate_VAR1(SGMatrix<double> &resid, /*T by 2 matrix of residuals*/
									SGVector<double> &theta, /*4 by 1 vector of estimates*/
									const SGMatrix<double> &Y, /*T by 2 matrix of data*/
									SGVector<double> theta0    /*4 by 1 vector of initial parameters*/);

		/* Compute the average SSE for VAR(1) */
		static double sse_VAR1(	const SGVector<double> &theta, /*a 4 by 1 vector of parameters*/
								const SGMatrix<double> &Y /*a T by 2 matrix of data*/);

		/* Approximate the first-order derivative of the SSE for VAR(1) */
		static SGVector<double> sse_VAR1_gradient(	const SGVector<double> &theta, /*a 4 by 1 vector of parameters*/
													const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
													double h /*finite differential level*/);

		/* Approximate the second-order derivatives of the SSE for VAR(1) */
		static SGMatrix<double> sse_VAR1_hessian(	const SGVector<double> &theta, /*a 4 by 1 vector of parameters*/
													const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
													double h /*finite differential level*/ );

		//generate a 3x1 normal random vector with a mean zero and a variance-covariance matrix (x).
        static Matrix multi_norm (gsl_matrix *x, unsigned long seed);
        //generate a vector of independent standard normal random variables.
        static void gen_unorm (Matrix &X, unsigned long seed);
        //run multivariate OLS. INPUT: a Tx1 vector of data on the dependent (Y) and a TxN matrix of data on the independent (X).
        //OUTPUT: a Tx1 vector of residuals (resid) and a Nx1 vector of the OLS estimates (slope)
        static void gen_Resid (Matrix &resid, Matrix &slope, const Matrix X, const Matrix Y);
        //estimate the bilinear regression model by the OLS. INPUT: a Tx1 vector of data on X.
        //OUTPUT: a (T-2)x1 vector of residuals (resid) and a 6x1 vector of the OLS estimates (slope)
        static void est_BL (Matrix &resid, Matrix &slope, const Matrix X);
        //estimate the TAR regression model by the OLS. INPUT: a Tx1 vector of data on X.
        //OUTPUT: a (T-2)x1 vector of residuals (resid) and a 5x1 vector of the OLS estimates (slope)
        static void est_TAR (Matrix &resid, Matrix &slope, const Matrix X);

	private:

		/* Standardize the columns of a matrix */
		static SGMatrix<double> standardize(const SGMatrix<double> &eta);

};

/* Standardize the columns of a matrix */
SGMatrix<double> NL_Dgp::standardize(const SGMatrix<double> &eta) {
	int T = eta.num_rows, dim = eta.num_cols;

	SGVector<double> mean(dim), std_dev(dim);
	mean = Statistics::matrix_mean(eta);
	std_dev = Statistics::matrix_std_deviation(eta);
	SGMatrix<double> eta_std(T, dim);
	for (int t = 0; t < T; ++t) {
		for (int i = 0; i < dim; ++i) {
			eta_std(t, i) = (eta(t,i) - mean[i]) / std_dev[i];
		}
	}
	return eta_std;
}



 //generate a vector of independent standard normal random variables.
void NL_Dgp::gen_unorm (Matrix &X, unsigned long seed) {
	gsl_rng *r = nullptr;
     const gsl_rng_type *gen; //random number generator
     gsl_rng_env_setup();
     gen = gsl_rng_default;
     r = gsl_rng_alloc(gen);
     gsl_rng_set(r, seed);
     for (int i = 1; i <= X.nRow(); i++)
		X(i) = gsl_ran_ugaussian (r);
	gsl_rng_free (r); //free memory
}

//generate a 3x1 normal random vector with a mean zero and a variance-covariance matrix (x).
Matrix NL_Dgp::multi_norm (gsl_matrix *x, unsigned long seed) { //x is a var-cov matrix
    gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);

    gsl_vector *mean = gsl_vector_calloc (3); //vector of zero means
    gsl_vector_set_all(mean, 0.);

    gsl_matrix * L = gsl_matrix_calloc (3, 3);
    gsl_matrix_memcpy (L, x); //copy x into L
    gsl_linalg_cholesky_decomp1 (L); //get a Cholesky decomposition matrix
    gsl_vector *xi_vec = gsl_vector_calloc (3);
	gsl_ran_multivariate_gaussian (r, mean, L, xi_vec); //call the multivariate normal random generator
	Matrix result(3, 1);
	result(1) = gsl_vector_get (xi_vec, 0);
	result(2) = gsl_vector_get (xi_vec, 1);
	result(3) = gsl_vector_get (xi_vec, 2);
	gsl_vector_free (mean); //free memory
	gsl_matrix_free (L);
    gsl_vector_free (xi_vec);
    gsl_rng_free (r);
    return result;
}

/* Generate 6-dimensional centered normal random variables defined on page 31 in Wang, Li & Zhu (2021).
OUTPUT: T independent realizations of two scalar random numbers (u1 and u2) and two 2 by 1 vectors (u3 and u4) */
std::tuple< SGVector<double>, SGVector<double>, SGMatrix<double>, SGMatrix<double> > NL_Dgp::gen_MNorm(	int T,
																										double rho1,
																										double rho4,
																										unsigned long seed) {

	/* Construct variance-covariance matrix */
	int n = 6;
	double rho2 = 0.5, rho3 = 0.75;
	gsl_matrix *Omega = gsl_matrix_alloc (n, n);
	gsl_matrix_set(Omega, 0, 0,  1.);
	gsl_matrix_set(Omega, 0, 1,  rho1);
	gsl_matrix_set(Omega, 0, 2,  0.);
	gsl_matrix_set(Omega, 0, 3,  0.);
	gsl_matrix_set(Omega, 0, 4,  0.);
	gsl_matrix_set(Omega, 0, 5,  0.);

	gsl_matrix_set(Omega, 1, 0,  rho1);
	gsl_matrix_set(Omega, 1, 1,  1.);
	gsl_matrix_set(Omega, 1, 2,  0.);
	gsl_matrix_set(Omega, 1, 3,  0.);
	gsl_matrix_set(Omega, 1, 4,  0.);
	gsl_matrix_set(Omega, 1, 5,  0.);

	gsl_matrix_set(Omega, 2, 0,  0.);
	gsl_matrix_set(Omega, 2, 1,  0.);
	gsl_matrix_set(Omega, 2, 2,  1.);
	gsl_matrix_set(Omega, 2, 3,  rho2);
	gsl_matrix_set(Omega, 2, 4,  rho4);
	gsl_matrix_set(Omega, 2, 5,  rho4);

	gsl_matrix_set(Omega, 3, 0,  0.);
	gsl_matrix_set(Omega, 3, 1,  0.);
	gsl_matrix_set(Omega, 3, 2,  rho2);
	gsl_matrix_set(Omega, 3, 3,  1.);
	gsl_matrix_set(Omega, 3, 4,  rho4);
	gsl_matrix_set(Omega, 3, 5,  rho4);

	gsl_matrix_set(Omega, 4, 0,  0.);
	gsl_matrix_set(Omega, 4, 1,  0.);
	gsl_matrix_set(Omega, 4, 2,  rho4);
	gsl_matrix_set(Omega, 4, 3,  rho4);
	gsl_matrix_set(Omega, 4, 4,  1.);
	gsl_matrix_set(Omega, 4, 5,  rho3);

	gsl_matrix_set(Omega, 5, 0,  0.);
	gsl_matrix_set(Omega, 5, 1,  0.);
	gsl_matrix_set(Omega, 5, 2,  rho4);
	gsl_matrix_set(Omega, 5, 3,  rho4);
	gsl_matrix_set(Omega, 5, 4,  rho3);
	gsl_matrix_set(Omega, 5, 5,  1.);

//	for (int i = 0; i < n; ++i) {
//		for (int j = 0; j < n; ++j) {
//			cout << gsl_matrix_get(Omega, i, j) << " , ";
//		}
//		cout << "\n";
//	}


	gsl_vector *mean = gsl_vector_calloc(n); //vector of zero means
    gsl_vector_set_all(mean, 0.);


	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);


    gsl_matrix * L = gsl_matrix_calloc (n, n);
    gsl_matrix_memcpy (L, Omega); //copy Omega into L
    gsl_linalg_cholesky_decomp1 (L); //get a Cholesky decomposition matrix

    gsl_vector *xi_vec = gsl_vector_calloc (n);
    SGVector<double> u1(T), u2(T);
    SGMatrix<double> u3(T, 2), u4(T, 2);
    for (int t = 0; t < T; ++t) {
		gsl_vector_set_all(xi_vec, 0.); // reset 'xi_vec'
		gsl_ran_multivariate_gaussian (r, mean, L, xi_vec); //call the multivariate normal random generator

		u1[t] = gsl_vector_get (xi_vec, 0);
		u2[t] = gsl_vector_get (xi_vec, 1);
		u3(t, 0) = gsl_vector_get (xi_vec, 2);
		u3(t, 1) = gsl_vector_get (xi_vec, 3);
		u4(t, 0) = gsl_vector_get (xi_vec, 4);
		u4(t, 1) = gsl_vector_get (xi_vec, 5);
    }

	gsl_matrix_free(Omega);//free memory
	gsl_vector_free(mean);
	gsl_matrix_free(L);
    gsl_vector_free(xi_vec);
    gsl_rng_free(r);

    return {u1, u2, u3, u4};
}

/* Generate data from a CC-MGARCH model as defined on on page 367 in Tse (2002).
OUTPUT: two T by 2 random matrices (Y1 and Y2) */
void NL_Dgp::gen_CC_MGARCH(	SGMatrix<double> &Y1, SGMatrix<double> &Y2, /*T by 2 matrices of realizations from the CC-GARCH*/
							const SGVector<double> theta1, /*a 7 by 1 vector of parameters for the first process*/
							const SGVector<double> theta2, /*a 7 by 1 vector of parameters for the second process*/
							const int choose_alt, /*select how the innovations are generated*/
							unsigned long seed) {
	int T = Y1.num_rows;
	ASSERT_(T == Y2.num_rows);

	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    int B = 200; //burning the first 200 observations

    int lag = 3, T1 = T+B;
    SGVector<double> u1(T1+lag), u2(T1+lag);
    SGMatrix<double> u3(T1+lag, 2), u4(T1+lag, 2);
    SGMatrix<double> eta1(T1, 2), eta2(T1, 2);

	/* Generate random errors */
    switch (choose_alt) {
        case 0: { // draw from EGP1
        	double rho1 = 0., rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
			for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u3(t, i);
					eta2(t, i) = u4(t, i);
				}
			}
			break;
        }
        case 1: { // draw from EGP2
			double rho1 = 0., rho4 = 0.3;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
			for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u3(t, i);
					eta2(t, i) = u4(t, i);
				}
			}
			break;
        }
        case 2: { // draw from EGP3
			double rho1 = 0.8, rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u1[t] * (u3(t,i) - 1.) / sqrt(2.);
					eta2(t, i) = u2[t] * (u3(t,i) + 1.) / sqrt(2.);
				}
        	}
			break;
        }
	    case 3: { // draw from EGP4
			double rho1 = 0.8, rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1+lag, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u1[t] * (u3(t,i) - 1.) / sqrt(2.);
					eta2(t, i) = u2[t+lag] * (u3(t,i) + 1.) / sqrt(2.);
				}
        	}
        	break;
	    }
		case 4: { // draw from EGP5
			double rho1 = 0.8, rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = ( (pow(u1[t],2.) + 1) / sqrt(6) ) * u3(t,i); // ( fabs( u3(t,i) ) - sqrt(2./ M_PI) ) / (1. - 2./M_PI);
					eta2(t, i) = fabs(u2[t]) * u4(t,i); // ( fabs( u4(t,i) ) - sqrt(2./ M_PI) ) / (1. - 2./M_PI);
				}
        	}
			break;
		}
		case 5: { // draw from EGP6
			double rho1 = 0.8, rho4 = 0.8;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u1[t] * u3(t,i);
					eta2(t, i) = u2[t] * u4(t,i);
				}
        	}
			break;
		}
	    default: {
    		    cerr << "NL_Dgp::gen_CC_GARCH: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	    }
	}

	//eta1.display_matrix("eta1");

	/* Generate conditional variance-covariance matrices */
	SGMatrix<double> v1(2, 2), v2(2, 2);
	v1.set_const(0.);
	v2.set_const(0.);
	SGMatrix<double> Y1a(2, T1), Y2a(2, T1);
	Y1a.set_const(0.);
	Y2a.set_const(0.);
	SGMatrix<double> eigenvect_mat1(2, 2), eigenvect_mat2(2, 2), v1_sqrt(2, 2), v2_sqrt(2, 2), eigenval_mat1(2, 2), eigenval_mat2(2, 2);
	eigenval_mat1.set_const(0.);
	eigenval_mat2.set_const(0.);
	SGVector<double> eigenval1(2), eigenval2(2);
	for (int t = 1; t < T1; ++t) {
		v1(0,0) = theta1[0] + theta1[1]*v1(0,0) + theta1[2]*pow(Y1a(0,t-1), 2.);
		v1(1,1) = theta1[3] + theta1[4]*v1(1,1) + theta1[5]*pow(Y1a(1,t-1), 2.);
		v1(0,1) = theta1[6] * sqrt( v1(0,0) * v1(1,1) );
		v1(1,0) = v1(0,1);

		//use the eigenvalue decomposition
		eigen_solver(v1, eigenval1, eigenvect_mat1);
		for (int i = 0; i < 2; ++i)
			eigenval_mat1(i,i) = sqrt(eigenval1[i]);
		v1_sqrt = matrix_prod( eigenvect_mat1, matrix_prod(eigenval_mat1, eigenvect_mat1, false, true) );
		Y1a.set_column( t, matrix_prod( v1_sqrt, eta1.get_row_vector(t) ) );


		/*// use the Cholesky decomposition
		Y1a(0,t) = sqrt( v1(0,0) ) * eta1(t,0);
		Y1a(1,t) = v1(0,1) * eta1(t,0) / sqrt( v1(0,0) ) + pow( ( v1(0,0)*v1(1,1) - pow(v1(0,1), 2.) ) / v1(0,0), 0.5 ) * eta1(t,1);*/

		v2(0,0) = theta2[0] + theta2[1]*v2(0,0) + theta2[2]*pow(Y2a(0,t-1), 2.);
		v2(1,1) = theta2[3] + theta2[4]*v2(1,1) + theta2[5]*pow(Y2a(1,t-1), 2.);
		v2(0,1) = theta2[6] * sqrt( v2(0,0) * v2(1,1) );
		v2(1,0) = v2(0,1);

		//use the eigenvalue decomposition
		eigen_solver(v2, eigenval2, eigenvect_mat2);
		for (int i = 0; i < 2; ++i)
			eigenval_mat2(i,i) = sqrt(eigenval2[i]);
		v2_sqrt = matrix_prod( eigenvect_mat2, matrix_prod(eigenval_mat2, eigenvect_mat2, false, true) );
		Y2a.set_column( t, matrix_prod( v2_sqrt, eta2.get_row_vector(t) ) );


		/*// use the Cholesky decomposition
		Y2a(0,t) = sqrt( v2(0,0) ) * eta2(t,0);
		Y2a(1,t) = v2(0,1) * eta2(t,0) / sqrt( v2(0,0) ) + pow( ( v2(0,0)*v2(1,1) - pow(v2(0,1), 2.) ) / v2(0,0), 0.5 ) * eta2(t,1);*/
	}
	//v2.display_matrix("v2");

	for (int t = B; t < T1; ++t) {
		for (int i = 0; i < 2; ++i) {
			Y1(t-B,i) = Y1a(i,t);
			Y2(t-B,i) = Y2a(i,t);
		}
	}

    gsl_rng_free (r); //free memory
}

/* Calculate residuals from a CC-MGARCH process.
OUTPUT: a T by 2 matrix of residuals. */
SGMatrix<double> NL_Dgp::resid_CC_MGARCH(	const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
											SGVector<double> theta /*7 by 1 vector of parameters*/) {
	ASSERT_(theta.vlen == 7);
	int T = Y.num_rows;

	SGVector<double> eigenval(2);
	SGMatrix<double> v(2, 2), eigenval_mat(2, 2), eigenvect_mat(2, 2), v_sqrt(2, 2);
	v.set_const(0.);
	eigenval_mat.zero();
	SGMatrix<double> resid(2, T);
	resid.zero();
	for (int t = 1; t < T; ++t) {
		v(0,0) = theta[0] + theta[1]*v(0,0) + theta[2]*pow(Y(t-1,0), 2.);
		v(1,1) = theta[3] + theta[4]*v(1,1) + theta[5]*pow(Y(t-1,1), 2.);
		v(0,1) = theta[6] * sqrt( v(0,0) * v(1,1) );
		v(1,0) = v(0,1);

		//use the eigenvalue decomposition to compute the square root of the inverse variance-covariance matrix
		eigen_solver(v, eigenval, eigenvect_mat);
		for (int i = 0; i < 2; ++i)
			eigenval_mat(i,i) = pow(eigenval[i], -0.5);
		v_sqrt = matrix_prod( eigenvect_mat, matrix_prod(eigenval_mat, eigenvect_mat, false, true) );
		resid.set_column( t, matrix_prod( v_sqrt, Y.get_row_vector(t) ) );


		/*// use the Cholesky decomposition
		resid(0, t-1) = Y(t,0) / sqrt( v(0,0) );
		resid(1, t-1) = pow( ( v(0,0)*v(1,1) - pow(v(0,1), 2.) ) / v(0,0), -0.5 ) * ( Y(t,1) - v(0,1) * resid(0,t-1) / sqrt( v(0,0) ) );*/
	}

	return transpose_matrix(resid); // return a T-1 by 2 matrix
}

/* Generate data from a CC-MGARCH process given a sample of innovations */
SGMatrix<double> NL_Dgp::gen_CC_MGARCH(	const SGMatrix<double> &eta, /*T by 2 matrix of innovations*/
										const SGVector<double> theta /*7 by 1 vector of parameters*/) {
	int T = eta.num_rows, dim = eta.num_cols;
	ASSERT_(dim == 2 && theta.vlen == 7); // check if the dimensions are correct!

	SGVector<double> eigenval(dim);
	SGMatrix<double> v(dim, dim), eigenvect_mat(dim, dim), eigenval_mat(dim, dim), Y(dim, T), v_sqrt(dim, dim);
	v.zero();
	eigenvect_mat.zero();
	eigenval_mat.zero();
	Y.zero(); // set all initial values to zeros
	for (int t = 1; t < T; ++t) {
		v(0,0) = theta[0] + theta[1]*v(0,0) + theta[2]*pow(Y(0,t-1), 2.);
		v(1,1) = theta[3] + theta[4]*v(1,1) + theta[5]*pow(Y(1,t-1), 2.);
		v(0,1) = theta[6] * sqrt( v(0,0) * v(1,1) );
		v(1,0) = v(0,1);

		//use the eigenvalue decomposition
		eigen_solver(v, eigenval, eigenvect_mat);
		for (int i = 0; i < 2; ++i)
			eigenval_mat(i,i) = sqrt(eigenval[i]);
		v_sqrt = matrix_prod( eigenvect_mat, matrix_prod(eigenval_mat, eigenvect_mat, false, true) );
		Y.set_column( t, matrix_prod( v_sqrt, eta.get_row_vector(t) ) );


		/*// use the Cholesky decomposition
		Y(0,t) = sqrt( v(0,0) ) * eta(t,0);
		Y(1,t) = v(0,1) * eta(t,0) / sqrt( v(0,0) ) + pow( ( v(0,0)*v(1,1) - pow(v(0,1), 2.) ) / v(0,0), 0.5 ) * eta(t,1);*/
	}
	return transpose_matrix(Y);
}

/* Generate data from a VAR(1) defined in El Himdl and Roy (1997).
OUTPUT: two T by 2 random matrices (Y1 and Y2) */
void NL_Dgp::gen_VAR(	SGMatrix<double> &Y1, SGMatrix<double> &Y2, /*T by 2 matrices of realizations from two VAR(1) */
						const SGVector<double> theta1, /*a 4 by 1 vector of parameters for the first process*/
						const SGVector<double> theta2, /*a 4 by 1 vector of parameters for the second process*/
						const int choose_alt, /*select how the innovations are generated*/
						unsigned long seed) {
	int T = Y1.num_rows;
	ASSERT_(T == Y2.num_rows);

    int B = 200; //burning the first 200 observations

    int lag = 3, T1 = T+B;
    SGVector<double> u1(T1+lag), u2(T1+lag);
    SGMatrix<double> u3(T1+lag, 2), u4(T1+lag, 2);
    SGMatrix<double> eta1(T1, 2), eta2(T1, 2);

	/* Generate random errors */
    switch (choose_alt) {
        case 0: { // draw from EGP1
        	double rho1 = 0., rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
			for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u3(t, i);
					eta2(t, i) = u4(t, i);
				}
			}
			break;
        }
        case 1: { // draw from EGP2
			double rho1 = 0., rho4 = 0.3;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
			for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u3(t, i);
					eta2(t, i) = u4(t, i);
				}
			}
			break;
        }
        case 2: { // draw from EGP3
			double rho1 = 0.8, rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u1[t] * (u3(t,i) - 1.) / sqrt(2.);
					eta2(t, i) = u2[t] * (u3(t,i) + 1.) / sqrt(2.);
				}
        	}
			break;
        }
	    case 3: { // draw from EGP4
			double rho1 = 0.8, rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1+lag, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u1[t] * (u3(t,i) - 1.) / sqrt(2.);
					eta2(t, i) = u2[t+lag] * (u3(t,i) + 1.) / sqrt(2.);
				}
        	}
        	break;
	    }
		case 4: { // draw from EGP5
			double rho1 = 0.8, rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = ( (pow(u1[t],2.) + 1) / sqrt(6) ) * u3(t,i); // ( fabs( u3(t,i) ) - sqrt(2./ M_PI) ) / (1. - 2./M_PI);
					eta2(t, i) = fabs(u2[t]) * u4(t,i); // ( fabs( u4(t,i) ) - sqrt(2./ M_PI) ) / (1. - 2./M_PI);
				}
        	}
			break;
		}
		case 5: { // draw from EGP6
			double rho1 = 0.8, rho4 = 0.8;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u1[t] * u3(t,i);
					eta2(t, i) = u2[t] * u4(t,i);
				}
        	}
			break;
		}
	    default: {
    		    cerr << "NL_Dgp::gen_VAR: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	    }
	}

	//eta1.display_matrix("eta1");

	/* Simulate data from VAR(1) processes */
	SGMatrix<double> theta1a(theta1, 2, 2), theta2a(theta2, 2, 2);
	theta1a = transpose_matrix(theta1a);
	theta2a = transpose_matrix(theta2a);

	SGMatrix<double> Y1a(2, T1), Y2a(2, T1);
	Y1a.set_const(1e-2);
	Y2a.set_const(1e-2);
	for (int t = 1; t < T1; ++t) {
		Y1a.set_column( t, add(matrix_prod( theta1a, Y1a.get_column(t-1) ), eta1.get_row_vector(t), 1., 1.) );
		Y2a.set_column( t, add(matrix_prod( theta2a, Y2a.get_column(t-1) ), eta2.get_row_vector(t), 1., 1.) );
	}

	for (int t = B; t < T1; ++t) {
		for (int i = 0; i < 2; ++i) {
			Y1(t-B,i) = Y1a(i,t);
			Y2(t-B,i) = Y2a(i,t);
		}
	}
}

/* Generate data from a VAR(1) process given a sample of innovations */
SGMatrix<double> NL_Dgp::gen_VAR(	const SGMatrix<double> &eta, /*T by 2 matrix of innovations*/
									const SGVector<double> theta /*4 by 1 vector of parameters*/) {
	int T = eta.num_rows, dim = eta.num_cols;
	ASSERT_(dim == 2 && theta.vlen == 4); // check if the dimensions are correct!

	SGMatrix<double> theta_mat(theta, 2, 2);
	theta_mat = transpose_matrix(theta_mat);

	SGMatrix<double> Y(dim, T);
	Y.set_const(1e-2); // set all initial values to zeros
	for (int t = 1; t < T; ++t) {
		Y.set_column( t, add(matrix_prod( theta_mat, Y.get_column(t-1) ), eta.get_row_vector(t), 1., 1.) );
	}

	return transpose_matrix(Y);
}

/* Generate data from a VAR-CC-MGARCH(1,1) model as defined on on page 367 in Tse (2002).
OUTPUT: two T by 2 random matrices (Y1 and Y2) */
void NL_Dgp::gen_VAR_CC_MGARCH(	SGMatrix<double> &Y1, SGMatrix<double> &Y2, /*T by 2 matrices of realizations from the CC-GARCH*/
								const SGVector<double> theta1, /*a 11 by 1 vector of parameters for the first process*/
								const SGVector<double> theta2, /*a 11 by 1 vector of parameters for the second process*/
								const int choose_alt, /*select how the innovations are generated*/
								unsigned long seed) {
	int T = Y1.num_rows, dim = Y1.num_cols;
	ASSERT_(T == Y2.num_rows);
	ASSERT_(dim == 2 && theta1.vlen == 11 && theta2.vlen == 11); // check if the dimensions are correct!

	// retrieve the parameter values of the VAR and CC-MGARCH part
	SGVector<double> theta_var1(4), theta_var2(4), theta_mgarch1(7), theta_mgarch2(7);
	theta_var1 = get_subvector(theta1, 0, 3);
	theta_var2 = get_subvector(theta2, 0, 3);

	SGMatrix<double> theta_var_mat1(theta_var1, 2, 2), theta_var_mat2(theta_var2, 2, 2);
	theta_var_mat1 = transpose_matrix(theta_var_mat1);
	theta_var_mat2 = transpose_matrix(theta_var_mat2);

	theta_mgarch1 = get_subvector(theta1, 4, 10);
	theta_mgarch2 = get_subvector(theta2, 4, 10);

	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    int B = 200; //burning the first 200 observations

    int lag = 3, T1 = T+B;
    SGVector<double> u1(T1+lag), u2(T1+lag);
    SGMatrix<double> u3(T1+lag, 2), u4(T1+lag, 2);
    SGMatrix<double> eta1(T1, 2), eta2(T1, 2);

	/* Generate random errors */
    switch (choose_alt) {
        case 0: { // draw from EGP1
        	double rho1 = 0., rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
			for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u3(t, i);
					eta2(t, i) = u4(t, i);
				}
			}
			break;
        }
        case 1: { // draw from EGP2
			double rho1 = 0., rho4 = 0.3;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
			for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u3(t, i);
					eta2(t, i) = u4(t, i);
				}
			}
			break;
        }
        case 2: { // draw from EGP3
			double rho1 = 0.8, rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u1[t] * (u3(t,i) - 1.) / sqrt(2.);
					eta2(t, i) = u2[t] * (u3(t,i) + 1.) / sqrt(2.);
				}
        	}
			break;
        }
	    case 3: { // draw from EGP4
			double rho1 = 0.8, rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1+lag, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u1[t] * (u3(t,i) - 1.) / sqrt(2.);
					eta2(t, i) = u2[t+lag] * (u3(t,i) + 1.) / sqrt(2.);
				}
        	}
        	break;
	    }
		case 4: { // draw from EGP5
			double rho1 = 0.8, rho4 = 0.;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = ( (pow(u1[t],2.) + 1) / sqrt(6) ) * u3(t,i); // ( fabs( u3(t,i) ) - sqrt(2./ M_PI) ) / (1. - 2./M_PI);
					eta2(t, i) = fabs(u2[t]) * u4(t,i); // ( fabs( u4(t,i) ) - sqrt(2./ M_PI) ) / (1. - 2./M_PI);
				}
        	}
			break;
		}
		case 5: { // draw from EGP6
			double rho1 = 0.8, rho4 = 0.8;
        	std::tie(u1, u2, u3, u4) = NL_Dgp::gen_MNorm(T1, rho1, rho4, seed);
        	for (int t = 0; t < T1; ++t) {
				for (int i = 0; i < 2; ++i) {
					eta1(t, i) = u1[t] * u3(t,i);
					eta2(t, i) = u2[t] * u4(t,i);
				}
        	}
			break;
		}
	    default: {
    		    cerr << "NL_Dgp::gen_CC_GARCH: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	    }
	}

	//eta1.display_matrix("eta1");

	/* Generate conditional variance-covariance matrices */
	SGMatrix<double> v1(2, 2), v2(2, 2);
	v1.set_const(0.);
	v2.set_const(0.);

	SGMatrix<double> Y1a(2, T1), Y2a(2, T1);
	Y1a.zero();
	Y2a.zero();

	SGMatrix<double> eigenvect_mat1(2, 2), eigenvect_mat2(2, 2), v1_sqrt(2, 2), v2_sqrt(2, 2), eigenval_mat1(2, 2), eigenval_mat2(2, 2);
	eigenval_mat1.set_const(0.);
	eigenval_mat2.set_const(0.);
	SGVector<double> mu1_t(2), mu2_t(2), eigenval1(2), eigenval2(2);
	mu1_t.zero();
	mu2_t.zero();
	for (int t = 1; t < T1; ++t) {
		if (t > 1) {
			mu1_t = matrix_prod( theta_var_mat1, Y1a.get_column(t-2) );
		}
		v1(0,0) = theta_mgarch1[0] + theta_mgarch1[1]*v1(0,0) + theta_mgarch1[2]*pow(Y1a(0,t-1)-mu1_t[0], 2.);
		v1(1,1) = theta_mgarch1[3] + theta_mgarch1[4]*v1(1,1) + theta_mgarch1[5]*pow(Y1a(1,t-1)-mu1_t[1], 2.);
		v1(0,1) = theta_mgarch1[6] * sqrt( v1(0,0) * v1(1,1) );
		v1(1,0) = v1(0,1);

		//use the eigenvalue decomposition
		eigen_solver(v1, eigenval1, eigenvect_mat1);
		for (int i = 0; i < 2; ++i)
			eigenval_mat1(i,i) = sqrt(eigenval1[i]);
		v1_sqrt = matrix_prod( eigenvect_mat1, matrix_prod(eigenval_mat1, eigenvect_mat1, false, true) );
		Y1a.set_column( t, add(mu1_t, matrix_prod( v1_sqrt, eta1.get_row_vector(t) ), 1., 1.) );


		/*// use the Cholesky decomposition
		Y1a(0,t) = mu1_t[0] + sqrt( v1(0,0) ) * eta1(t,0);
		Y1a(1,t) = mu1_t[1] + v1(0,1) * eta1(t,0) / sqrt( v1(0,0) ) + pow( ( v1(0,0)*v1(1,1) - pow(v1(0,1), 2.) ) / v1(0,0), 0.5 ) * eta1(t,1);*/

		if (t > 1) {
			mu2_t = matrix_prod( theta_var_mat2, Y2a.get_column(t-2) );
		}
		v2(0,0) = theta_mgarch2[0] + theta_mgarch2[1]*v2(0,0) + theta_mgarch2[2]*pow(Y2a(0,t-1)-mu2_t[0], 2.);
		v2(1,1) = theta_mgarch2[3] + theta_mgarch2[4]*v2(1,1) + theta_mgarch2[5]*pow(Y2a(1,t-1)-mu2_t[1], 2.);
		v2(0,1) = theta_mgarch2[6] * sqrt( v2(0,0) * v2(1,1) );
		v2(1,0) = v2(0,1);

		//use the eigenvalue decomposition
		eigen_solver(v2, eigenval2, eigenvect_mat2);
		for (int i = 0; i < 2; ++i)
			eigenval_mat2(i,i) = sqrt(eigenval2[i]);
		v2_sqrt = matrix_prod( eigenvect_mat2, matrix_prod(eigenval_mat2, eigenvect_mat2, false, true) );
		Y2a.set_column( t, add(mu2_t, matrix_prod( v2_sqrt, eta2.get_row_vector(t) ), 1., 1.) );


		/*// use the Cholesky decomposition
		Y2a(0,t) = mu2_t[0] + sqrt( v2(0,0) ) * eta2(t,0);
		Y2a(1,t) = mu2_t[1] + v2(0,1) * eta2(t,0) / sqrt( v2(0,0) ) + pow( ( v2(0,0)*v2(1,1) - pow(v2(0,1), 2.) ) / v2(0,0), 0.5 ) * eta2(t,1);*/
	}
	//v2.display_matrix("v2");

	for (int t = B; t < T1; ++t) {
		for (int i = 0; i < 2; ++i) {
			Y1(t-B,i) = Y1a(i,t);
			Y2(t-B,i) = Y2a(i,t);
		}
	}

    gsl_rng_free (r); //free memory
}



/* Generate data from a VAR-CC-MGARCH(1,1) process given a sample of innovations */
SGMatrix<double> NL_Dgp::gen_VAR_CC_MGARCH(	const SGMatrix<double> &eta, /*T by 2 matrix of innovations*/
											const SGVector<double> theta /*11 by 1 vector of parameters*/) {
	int T = eta.num_rows, dim = eta.num_cols;
	ASSERT_(dim == 2 && theta.vlen == 11); // check if the dimensions are correct!

	SGVector<double> theta_var(4), theta_mgarch(7);
	theta_var = get_subvector(theta, 0, 3);
	SGMatrix<double> theta_var_mat(theta_var, 2, 2);
	theta_var_mat = transpose_matrix(theta_var_mat);

	theta_mgarch = get_subvector(theta, 4, 10);

	SGVector<double> eigenval(dim), mu(dim);
	mu.zero();
	SGMatrix<double> v(dim, dim), eigenvect_mat(dim, dim), eigenval_mat(dim, dim), Y(dim, T), v_sqrt(dim, dim);
	v.zero();
	eigenvect_mat.zero();
	eigenval_mat.zero();
	Y.zero(); // set all initial values to zeros
	for (int t = 1; t < T; ++t) {
		if (t > 1)
			mu = matrix_prod( theta_var_mat, Y.get_column(t-2) ); //calculate the conditional mean

		v(0,0) = theta_mgarch[0] + theta_mgarch[1]*v(0,0) + theta_mgarch[2]*pow(Y(0,t-1)-mu[0], 2.);
		v(1,1) = theta_mgarch[3] + theta_mgarch[4]*v(1,1) + theta_mgarch[5]*pow(Y(1,t-1)-mu[1], 2.);
		v(0,1) = theta_mgarch[6] * sqrt( v(0,0) * v(1,1) );
		v(1,0) = v(0,1);

		//use the eigenvalue decomposition
		eigen_solver(v, eigenval, eigenvect_mat);
		for (int i = 0; i < 2; ++i)
			eigenval_mat(i,i) = sqrt(eigenval[i]);
		v_sqrt = matrix_prod( eigenvect_mat, matrix_prod(eigenval_mat, eigenvect_mat, false, true) );
		Y.set_column( t, add(mu, matrix_prod( v_sqrt, eta.get_row_vector(t) ), 1., 1.) );


		/*// use the Cholesky decomposition
		Y(0,t) = mu[0] + sqrt( v(0,0) ) * eta(t,0);
		Y(1,t) = mu[1] + v(0,1) * eta(t,0) / sqrt( v(0,0) ) + pow( ( v(0,0)*v(1,1) - pow(v(0,1), 2.) ) / v(0,0), 0.5 ) * eta(t,1);*/
	}
	return transpose_matrix(Y);
}



/* Calculate residuals from a VAR(1) process.
OUTPUT: a T by 2 matrix of residuals. */
SGMatrix<double> NL_Dgp::resid_VAR1(const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
									SGVector<double> theta /*4 by 1 vector of parameters*/) {
	ASSERT_(theta.vlen == 4);
	int T = Y.num_rows;

	SGMatrix<double> theta_mat(theta, 2, 2);
	theta_mat = transpose_matrix(theta_mat);

	SGMatrix<double> resid(2, T);
	resid.zero();
	for (int t = 1; t < T; ++t) {
		resid.set_column( t, add(Y.get_row_vector(t), matrix_prod( theta_mat, Y.get_row_vector(t-1) ), 1., -1.) );
	}

	return transpose_matrix(resid);
}

/* Calculate residuals from a VAR-CC-MGARCH(1,1) process.
OUTPUT: a T by 2 matrix of residuals. */
SGMatrix<double> NL_Dgp::resid_VAR_CC_MGARCH(	const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
												SGVector<double> theta /*11 by 1 vector of parameters*/) {
	ASSERT_(theta.vlen == 11);
	int T = Y.num_rows;

	SGVector<double> theta_var(4), theta_mgarch(7);
	theta_var = get_subvector(theta, 0, 3);
	theta_mgarch = get_subvector(theta, 4, 10);

	// Generate residuals from a VAR(1) process
	SGMatrix<double> resid_var = NL_Dgp::resid_VAR1(Y, theta_var);

	SGVector<double> eigenval(2);
	SGMatrix<double> v(2, 2), eigenval_mat(2, 2), eigenvect_mat(2, 2), v_sqrt(2, 2);
	v.set_const(0.);
	eigenval_mat.zero();
	SGMatrix<double> resid(2, T);
	resid.zero();
	for (int t = 1; t < T; ++t) {
		v(0,0) = theta[0] + theta[1]*v(0,0) + theta[2]*pow(resid_var(t-1,0), 2.);
		v(1,1) = theta[3] + theta[4]*v(1,1) + theta[5]*pow(resid_var(t-1,1), 2.);
		v(0,1) = theta[6] * sqrt( v(0,0) * v(1,1) );
		v(1,0) = v(0,1);

		//use the eigenvalue decomposition to compute the square root of the inverse variance-covariance matrix
		eigen_solver(v, eigenval, eigenvect_mat);
		for (int i = 0; i < 2; ++i)
			eigenval_mat(i,i) = pow(eigenval[i], -0.5);
		v_sqrt = matrix_prod( eigenvect_mat, matrix_prod(eigenval_mat, eigenvect_mat, false, true) );
		resid.set_column( t, matrix_prod( v_sqrt, resid_var.get_row_vector(t) ) );


		/*// use the Cholesky decomposition
		resid(0, t-1) = resid_var(t,0) / sqrt( v(0,0) );
		resid(1, t-1) = pow( ( v(0,0)*v(1,1) - pow(v(0,1), 2.) ) / v(0,0), -0.5 ) * ( resid_var(t,1) - v(0,1) * resid(0,t-1) / sqrt( v(0,0) ) );*/
	}

	return transpose_matrix(resid); // return a T-1 by 2 matrix
}


/* Estimate VAR(1) */
double NL_Dgp::estimate_VAR1(	SGMatrix<double> &resid, /*T by 2 matrix of residuals*/
								SGVector<double> &theta, /*4 by 1 vector of estimates*/
								const SGMatrix<double> &Y, /*T by 2 matrix of data*/
								SGVector<double> theta0    /*4 by 1 vector of initial parameters*/) {
	(void) theta0;

	SGMatrix<double> Phi(2, 3);
	resid = NL_Dgp::resid_VAR(Phi, Y, 1);
	theta[0] = Phi(0,1);
	theta[1] = Phi(0,2);
	theta[2] = Phi(1,1);
	theta[3] = Phi(1,2);
	return 0.;
}

/* Fit data to a VAR(p) model and get residuals */
SGMatrix<double> NL_Dgp::resid_VAR(	SGMatrix<double> &Phi, /*dim by dim*p+1 matrix of parameter estimates*/
									const SGMatrix<double> &Y, /*T by dim matrix of observations*/
									int p /*maximum VAR lag*/) {
	int T = Y.num_rows, dim = Y.num_cols;

	SGMatrix<double> X_t(dim*p+1, 1), Y_t(dim, 1);
	X_t[0] = 1.;
	SGMatrix<double> A(dim, dim*p+1), B(dim*p+1, dim*p+1);
	A.zero();
	B.zero();
	for (int t = p; t < T; ++t) {
		for (int i = 0; i < p; ++i) {
			for (int j = 0; j < dim; ++j) {
				X_t(i*dim + j + 1, 0) = Y(t-i-1, j);
			}
		}
		Y_t.set_column( 0, Y.get_row_vector(t) );
		A = add(A, matrix_prod( Y_t, transpose_matrix(X_t) ), 1., 1.);
		B = add(B, matrix_prod( X_t, transpose_matrix(X_t) ), 1., 1.);
	}

	Phi = matrix_prod( A, pinv<double>(B) );

	//Phi.display_matrix("Phi");

	SGMatrix<double> resid(dim, T), resid_t(dim, 1);
	resid.zero();
	for (int t = p; t < T; ++t) {
		for (int i = 0; i < p; ++i) {
			for (int j = 0; j < dim; ++j) {
				X_t(i*dim + j + 1, 0) = Y(t-i-1, j);
			}
		}
		Y_t.set_column( 0, Y.get_row_vector(t) );
		resid_t = add(Y_t, matrix_prod(Phi, X_t), 1., -1.);

		resid.set_column( t, resid_t.get_column(0) );
	}
	return transpose_matrix(resid);
}


/* Compute the average SSE for VAR(1) */
double NL_Dgp::sse_VAR1(const SGVector<double> &theta, /*a 4 by 1 vector of parameters*/
						const SGMatrix<double> &Y /*a T by 2 matrix of data*/) {
	int T = Y.num_rows, dim = theta.vlen;
	ASSERT_(Y.num_cols == 2 && dim == 4);

	SGMatrix<double> theta_mat(theta, 2, 2);
	theta_mat = transpose_matrix(theta_mat);

	SGVector<double> resid(2);
	double sse = 0.;
	for (int t = 1; t < T; ++t) {
		resid = add(Y.get_row_vector(t), matrix_prod( theta_mat, Y.get_row_vector(t-1) ), 1., -1.);
		sse += dot(resid, resid);
	}
	return sse / T;
}

/* Approximate the first-order derivative of the SSE for VAR(1) */
SGVector<double> NL_Dgp::sse_VAR1_gradient(	const SGVector<double> &theta, /*a 4 by 1 vector of parameters*/
											const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
											double h /*finite differential level*/) {
	int dim = theta.vlen;
	ASSERT_(dim == 4);

	SGVector<double> theta1(dim), theta2(dim), df(dim);

	for (int i = 0; i < dim; ++i) {
		theta1 = add(theta, base_vec(i, dim), 1., h);
		theta2 = add(theta, base_vec(i, dim), 1., -h);
		df[i] = ( NL_Dgp::sse_VAR1(theta1, Y) - NL_Dgp::sse_VAR1(theta2, Y) ) / (2*h);
	}

	return df;
}


/* Approximate the second-order derivatives of the SSE for VAR(1) */
SGMatrix<double> NL_Dgp::sse_VAR1_hessian(	const SGVector<double> &theta, /*a 4 by 1 vector of parameters*/
											const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
											double h /*finite differential level*/ ) {
	int dim = theta.vlen;
	ASSERT_(dim == 4);

	SGVector<double> theta1(dim), theta2(dim), theta3(dim);
	SGMatrix<double> df(dim, dim);

	for (int i = 0; i < dim; ++i) {
		theta2 = add(theta, base_vec(i,dim), 1., h);
		for (int j = 0; j < dim; ++j) {
			theta1 = add(theta2, base_vec(j,dim), 1., h);
			theta3 = add(theta, base_vec(j,dim), 1., h);
			df(i, j) = ( NL_Dgp::sse_VAR1(theta1, Y) - NL_Dgp::sse_VAR1(theta2, Y) - NL_Dgp::sse_VAR1(theta3, Y) \
																						+ NL_Dgp::sse_VAR1(theta, Y) ) / pow(h,2.);
		}
	}
	return df;
}


//generate TWO bilinear processes of the second order. INPUT: A 6x1 vector of coefficients for X (alpha_X), a 6x1 vector of coefficients for Y (alpha_Y),
//and a template function to generate two random errors for both the processes (gen_RAN). OUTPUT: Tx1 matrices (X and Y).
template <void gen_RAN (double &, double &, const double, const double, const int, unsigned long)>
void NL_Dgp::gen_Bilinear (Matrix &X, Matrix &Y, const Matrix alpha_X, const Matrix alpha_Y, const double delta, const double rho,
                           const int choose_alt, unsigned long seed) {
    gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    auto T = X.nRow();
    ASSERT(T == Y.nRow());
    auto B = 200; //burning the first 200 observations
    Matrix X_tmp(T+B,1), Y_tmp(T+B,1);
    unsigned long rseed = gsl_rng_get (r); //a random seed
	gen_RAN (X_tmp(1), Y_tmp(1), delta, rho, choose_alt, seed); //generate initial values
	gen_RAN (X_tmp(2), Y_tmp(2), delta, rho, choose_alt, rseed);
	auto epsilon = 0., eta = 0.;
	for (auto t = 3; t <= T+B; ++t) {
		rseed = gsl_rng_get (r); //a random seed
		gen_RAN (epsilon, eta, delta, rho, choose_alt, rseed); //generate random error terms (with different seeds)
		X_tmp(t) = alpha_X(1) + alpha_X(2)*X_tmp(t-1) + alpha_X(3)*pow(X_tmp(t-1),2.) + alpha_X(4)*X_tmp(t-2) + alpha_X(5)*pow(X_tmp(t-2),2.)
		       + alpha_X(6)*X_tmp(t-1)*X_tmp(t-2) + epsilon;
		Y_tmp(t) = alpha_Y(1) + alpha_Y(2)*Y_tmp(t-1) + alpha_Y(3)*pow(Y_tmp(t-1),2.) + alpha_Y(4)*Y_tmp(t-2) + alpha_Y(5)*pow(Y_tmp(t-2),2.)
		       + alpha_Y(6)*Y_tmp(t-1)*Y_tmp(t-2) + eta;
		if (t > B) {
	        X(t-B) = X_tmp(t);
	    	Y(t-B) = Y_tmp(t);
		}
	}
	gsl_rng_free (r); //free memory
}

//generate a univariate bilinear process and a bivariate bilinear process of the second orders that may have some dependency.
//INPUT: A 6x1 vector of coefficients for X (alpha_X), a 6x2 matrix of coefficients for Y1 and Y2 (alpha_Y), and a template function
//to generate random errors for each individual process (gen_RAN). OUTPUT: a Tx1 vector (X) and Tx1 vectors (Y1 and Y2).
template <void gen_RAN (double &, double &, double &, const Matrix &, const double, const double, const int, unsigned long)>
void NL_Dgp::gen_Bilinear (Matrix &X, Matrix &Y1, Matrix &Y2, const Matrix alpha_X, const Matrix alpha_Y, const Matrix &delta,
                           const double alpha5, const double rho1, const int choose_alt, unsigned long seed) {
    gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    auto T = X.nRow();
    ASSERT(T == Y1.nRow() && T == Y2.nRow());
    auto B = 200; //burning the first 200 observations
    Matrix X_tmp(T+B,1), Y_tmp(T+B,2);
    unsigned long rseed = gsl_rng_get (r); //a random seed
    gen_RAN (X_tmp(1), Y_tmp(1,1), Y_tmp(1,2), delta, alpha5, rho1, choose_alt, seed); //generate initial values
    gen_RAN (X_tmp(2), Y_tmp(2,1), Y_tmp(2,2), delta, alpha5, rho1, choose_alt, rseed);
    auto epsilon = 0., eta1 = 0., eta2 = 0.;
    for (auto t = 3; t <= T + B; ++t) {
    	rseed = gsl_rng_get (r); //a random seed
    	gen_RAN (epsilon, eta1, eta2, delta, alpha5, rho1, choose_alt, rseed); //generate random errors (with different seeds)
    	X_tmp(t) = alpha_X(1) + alpha_X(2)*X_tmp(t-1) + alpha_X(3)*pow(X_tmp(t-1),2.) + alpha_X(4)*X_tmp(t-2) + alpha_X(5)*pow(X_tmp(t-2),2.)
    	           + alpha_X(6)*X_tmp(t-1)*X_tmp(t-2) + epsilon;
    	Y_tmp(t,1) = alpha_Y(1,1) + alpha_Y(2,1)*Y_tmp(t-1,1) + alpha_Y(3,1)*pow(Y_tmp(t-1,1),2.) + alpha_Y(4,1)*Y_tmp(t-2,1)
		           + alpha_Y(5,1)*pow(Y_tmp(t-2,1),2.) + alpha_Y(6,1)*Y_tmp(t-1,1)*Y_tmp(t-2,1) + eta1;
		Y_tmp(t,2) = alpha_Y(1,2) + alpha_Y(2,2)*Y_tmp(t-1,2) + alpha_Y(3,2)*pow(Y_tmp(t-1,2),2.) + alpha_Y(4,2)*Y_tmp(t-2,2)
		           + alpha_Y(5,2)*pow(Y_tmp(t-2,2),2.) + alpha_Y(6,2)*Y_tmp(t-1,2)*Y_tmp(t-2,2) + eta2;
		if (t > B) {
	    	X(t-B) = X_tmp(t);
	    	Y1(t-B) = Y_tmp(t,1);
	    	Y2(t-B) = Y_tmp(t,2);
		}
    }
    gsl_rng_free (r); //free memory
}

//generate TWO threshold AR processes of the second order. INPUT: A 5x1 vector of coefficients for X (alpha_X), a 5x1 vector of coefficients for Y (alpha_Y),
//and a template function to generate two random errors for both the processes (gen_RAN). OUTPUT: Tx1 matrices (X and Y).
template <void gen_RAN (double &, double &, const double, const double, const int, unsigned long)>
void NL_Dgp::gen_TAR (Matrix &X, Matrix &Y, const Matrix alpha_X, const Matrix alpha_Y, const double delta, const double rho, const int choose_alt,
                      unsigned long seed) {
    gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    auto T = X.nRow();
    ASSERT(T == Y.nRow());
    auto B = 200; //burning the first 200 observations
    Matrix X_tmp(T+B,1), Y_tmp(T+B,1);
    unsigned long rseed = gsl_rng_get (r); //a random seed
	gen_RAN (X_tmp(1), Y_tmp(1), delta, rho, choose_alt, seed); //generate initial values
	gen_RAN (X_tmp(2), Y_tmp(2), delta, rho, choose_alt, rseed);
	auto epsilon = 0., eta = 0.;
	auto pos = [](double x) { //get the positive part of a real number (x)
    	if (x > 0.)
    		return x;
    	else
    		return 0.;
	};
	for (auto t = 3; t <= T+B; ++t) {
		rseed = gsl_rng_get (r); //a random seed
		gen_RAN (epsilon, eta, delta, rho, choose_alt, rseed); //generate random error terms (with different seeds)
		X_tmp(t) = alpha_X(1) + alpha_X(2)*X_tmp(t-1) + alpha_X(3)*pos(X_tmp(t-1)) + alpha_X(4)*X_tmp(t-2) + alpha_X(5)*pos(X_tmp(t-2)) + epsilon;
		Y_tmp(t) = alpha_Y(1) + alpha_Y(2)*Y_tmp(t-1) + alpha_Y(3)*pos(Y_tmp(t-1)) + alpha_Y(4)*Y_tmp(t-2) + alpha_Y(5)*pos(Y_tmp(t-2)) + eta;
		if (t > B) {
	        X(t-B) = X_tmp(t);
	    	Y(t-B) = Y_tmp(t);
		}
	}
	gsl_rng_free (r); //free memory
}

//generate a univariate threshold AR process and a bivariate threshold AR process of the second orders.
//INPUT: A 5x1 vector of coefficients for X (alpha_X), a 5x2 matrix of coefficients for Y1 and Y2 (alpha_Y),
//and a template function to generate two random errors for both the processes (gen_RAN). OUTPUT: a Tx1 vector (X) and Tx1 vectors (Y1 and Y2).
template <void gen_RAN (double &, double &, double &, const Matrix &, const double, const double, const int, unsigned long)>
void NL_Dgp::gen_TAR (Matrix &X, Matrix &Y1, Matrix &Y2, const Matrix alpha_X, const Matrix alpha_Y, const Matrix &delta, const double alpha5, const double rho1,
                                            const int choose_alt, unsigned long seed) {
    gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
	auto T = X.nRow();
    ASSERT(T == Y1.nRow() && T == Y2.nRow());
    auto B = 200; //burning the first 200 observations
    Matrix X_tmp(T+B, 1), Y_tmp(T+B, 2);
    unsigned long rseed = gsl_rng_get (r); //a random seed
    gen_RAN (X_tmp(1), Y_tmp(1,1), Y_tmp(1,2), delta, alpha5, rho1, choose_alt, seed); //generate initial values
    gen_RAN (X_tmp(2), Y_tmp(2,1), Y_tmp(2,2), delta, alpha5, rho1, choose_alt, rseed);
    double epsilon = 0., eta1 = 0., eta2 = 0.;
    auto pos = [](double x) { //get the positive part of a real number (x)
    	if (x > 0.)
    		return x;
    	else
    		return 0.;
	};
    for (auto t = 3; t <= T + B; ++t) {
          rseed = gsl_rng_get (r); //a random seed
          gen_RAN (epsilon, eta1, eta2, delta, alpha5, rho1, choose_alt, rseed); //generate random errors (with different seeds)
          X_tmp(t) = alpha_X(1) + alpha_X(2)*X_tmp(t-1) + alpha_X(3)*pos(X_tmp(t-1)) + alpha_X(4)*X_tmp(t-2) + alpha_X(5)*pos(X_tmp(t-2)) + epsilon;
          Y_tmp(t,1) = alpha_Y(1,1) + alpha_Y(2,1)*Y_tmp(t-1,1) + alpha_Y(3,1)*pos(Y_tmp(t-1,1)) + alpha_Y(4,1)*Y_tmp(t-2,1) + alpha_Y(5,1)*pos(Y_tmp(t-2,1))
                                 + eta1;
          Y_tmp(t,2) = alpha_Y(1,2) + alpha_Y(2,2)*Y_tmp(t-1,2) + alpha_Y(3,2)*pos(Y_tmp(t-1,2)) + alpha_Y(4,2)*Y_tmp(t-2,2) + alpha_Y(5,2)*pos(Y_tmp(t-2,2))
		                       + eta2;
		if (t > B) {
	        X(t-B) = X_tmp(t);
	    	   Y1(t-B) = Y_tmp(t,1);
	    	   Y2(t-B) = Y_tmp(t,2);
		}
    }
	gsl_rng_free (r); //free memory
}

//generate centered skew-normal random error terms. INPUT: delta in (-1,1), a correlation between xi_1_eps and xi_1_eta (rho), an alternative:
//choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
//dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: two random error terms (epsilon and eta).
void NL_Dgp::gen_SN (double &epsilon, double &eta, const double delta, const double rho, const int choose_alt, unsigned long seed) {
	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double xi_0_eps = gsl_ran_ugaussian (r);
    double xi_0_eta = gsl_ran_ugaussian (r);
    double xi_1_eps = gsl_ran_ugaussian (r);
	double xi_1_eta = 0.;
	switch (choose_alt) {
        case 0:
                xi_1_eta = gsl_ran_ugaussian (r); //independent error terms
                break;
        case 1:
        	    gsl_ran_bivariate_gaussian (r, 1., 1., rho, &xi_1_eps, &xi_1_eta); //correlated error terms
        	    break;
        case 2:
        	    if (fabs(xi_1_eps) <= 1.54) //set threshold equal to 1.54 for zero correlation
	                xi_1_eta = xi_1_eps;
	            else
	                xi_1_eta = -xi_1_eps;
	            break;
	    case 3:
	            if (fabs(xi_1_eps) <= 2.) //set threshold equal to 2.0 for non-zero correlation
	                xi_1_eta = xi_1_eps;
	            else
	                xi_1_eta = -xi_1_eps;
	            break;
	    default:
    		    cerr << "NL_Dgp::gen_SM: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	}
	epsilon = delta * fabs(xi_0_eps) + sqrt(1 - pow(delta,2.)) * xi_1_eps - sqrt((double) 2/M_PI) * delta;
	eta =     delta * fabs(xi_0_eta) + sqrt(1 - pow(delta,2.)) * xi_1_eta - sqrt((double) 2/M_PI) * delta;
	gsl_rng_free (r); //free memory
}

//generate a trivariate skew-normal random variable. INPUT: a 3x1 vector (delta) in (-1,1)^3, correlations (rho12 and rho13), an alternative:
//set choose_alt = 0 to generate independent random variables, choose_alt = 1 to generate *epsilon uncorrelated and dependent with *eta1 and *eta2,
//choose_alt = 2 to generate dependent random variables. OUTPUT: three random variables (epsilon, eta1, and eta2).
void NL_Dgp::gen_TriSN (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double rho12, const double rho13, const int choose_alt,
                        unsigned long seed) {
    gsl_rng *r = nullptr;
	const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double xi_0 = gsl_ran_ugaussian (r);
    Matrix xi(3, 1);
    xi(1) = gsl_ran_ugaussian (r);
    xi(2) = gsl_ran_ugaussian (r);
    xi(3) = gsl_ran_ugaussian (r);
    switch (choose_alt) {
    	case 0: {
                    double xi_0_star = gsl_ran_ugaussian (r);
                    epsilon = delta(1) * fabs(xi_0) + sqrt(1 - pow(delta(1),2.)) * xi(1) - sqrt((double) 2/M_PI) * delta(1);
                    eta1    = delta(2) * fabs(xi_0_star) + sqrt(1 - pow(delta(2),2.)) * xi(2) - sqrt((double) 2/M_PI) * delta(2);
                    eta2    = delta(3) * fabs(xi_0_star) + sqrt(1 - pow(delta(3),2.)) * xi(3) - sqrt((double) 2/M_PI) * delta(3);
                    gsl_rng_free (r); //free memory
	            }
                 break;
    	case 1: {
    		         gsl_matrix *variance = gsl_matrix_alloc (3,3);
    		         gsl_matrix_set (variance, 0, 0, 1.);
    		         double _rho12 = -delta(1)*delta(2)*(1 - 2/M_PI) / sqrt((1 - pow(delta(1),2.))*(1 - pow(delta(2),2.)));
    		         //cout << "_rho12 = " << _rho12 << endl;
    		         gsl_matrix_set (variance, 0, 1, _rho12);
    		         double _rho13 = -delta(1)*delta(3)*(1 - 2/M_PI) / sqrt((1 - pow(delta(1),2.))*(1 - pow(delta(3),2.)));
    		          //cout << "_rho13 = " << _rho13 << endl;
    		          gsl_matrix_set (variance, 0, 2, _rho13);
                    gsl_matrix_set (variance, 1, 0, _rho12);
                    gsl_matrix_set (variance, 1, 1, 1.);
                    gsl_matrix_set (variance, 1, 2, 0.6);
                    gsl_matrix_set (variance, 2, 0, _rho13);
                    gsl_matrix_set (variance, 2, 1, 0.6);
                    gsl_matrix_set (variance, 2, 2, 1.);
                    xi = NL_Dgp::multi_norm (variance, seed); //generate a trivariate normal random variable
                    epsilon = delta(1) * fabs(xi_0) + sqrt(1 - pow(delta(1),2.)) * xi(1) - sqrt((double) 2/M_PI) * delta(1);
	                eta1    = delta(2) * fabs(xi_0) + sqrt(1 - pow(delta(2),2.)) * xi(2) - sqrt((double) 2/M_PI) * delta(2);
	                eta2    = delta(3) * fabs(xi_0) + sqrt(1 - pow(delta(3),2.)) * xi(3) - sqrt((double) 2/M_PI) * delta(3);
	                gsl_matrix_free (variance);
	                gsl_rng_free (r); //free memory
	            }
	            break;
    	case 2: {
                    gsl_matrix *variance = gsl_matrix_alloc (3,3);
                    gsl_matrix_set (variance, 0, 0, 1.);
                    gsl_matrix_set (variance, 0, 1, rho12);
                    gsl_matrix_set (variance, 0, 2, rho13);
                    gsl_matrix_set (variance, 1, 0, rho12);
                    gsl_matrix_set (variance, 1, 1, 1.);
                    gsl_matrix_set (variance, 1, 2, 0.6);
                    gsl_matrix_set (variance, 2, 0, rho13);
                    gsl_matrix_set (variance, 2, 1, 0.6);
                    gsl_matrix_set (variance, 2, 2, 1.);
                    xi = NL_Dgp::multi_norm (variance, seed); //generate a trivariate normal random variable
                    epsilon = delta(1) * fabs(xi_0) + sqrt(1 - pow(delta(1),2.)) * xi(1) - sqrt((double) 2/M_PI) * delta(1);
	                eta1    = delta(2) * fabs(xi_0) + sqrt(1 - pow(delta(2),2.)) * xi(2) - sqrt((double) 2/M_PI) * delta(2);
	                eta2    = delta(3) * fabs(xi_0) + sqrt(1 - pow(delta(3),2.)) * xi(3) - sqrt((double) 2/M_PI) * delta(3);
	                gsl_matrix_free (variance);
	                gsl_rng_free (r); //free memory
	            }
    		    break;
    	default:
    		    cerr << "NL_Dgp::gen_TriSN: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	}
}

//generate two mixtures of N(0,1) random variables. INPUT: a correlation between xi_1_eps and xi_1_eta (rho) and an alternative:
//choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
//dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: two random error terms (epsilon and eta).
void NL_Dgp::gen_MN (double &epsilon, double &eta, const double delta, const double rho, const int choose_alt, unsigned long seed) {
	(void) delta; //unused parameter
	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double xi_0_eps = gsl_ran_ugaussian (r);
    double xi_0_eta = gsl_ran_ugaussian (r);
    double xi_1_eps = gsl_ran_ugaussian (r);
	double xi_1_eta = 0.;
	switch (choose_alt) {
        case 0:
                xi_1_eta = gsl_ran_ugaussian (r); //independent error terms
                break;
        case 1:
        	    gsl_ran_bivariate_gaussian (r, 1., 1., rho, &xi_1_eps, &xi_1_eta); //correlated error terms
        	    break;
        case 2:
        	    if (fabs(xi_1_eps) <= 1.54) //set threshold equal to 1.54 for zero correlation
	                xi_1_eta = xi_1_eps;
	            else
	                xi_1_eta = -xi_1_eps;
	            break;
	    case 3:
	            if (fabs(xi_1_eps) <= 2.) //set threshold equal to 2.0 for non-zero correlation
	                xi_1_eta = xi_1_eps;
	            else
	                xi_1_eta = -xi_1_eps;
	            break;
	    default:
    		    cerr << "NL_Dgp::gen_MN: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	}
	auto z_eps = gsl_ran_bernoulli (r, (double) 3/4);
	if (z_eps == 1)
	    epsilon = xi_1_eps;
	else
	    epsilon = xi_0_eps;
	auto z_eta = gsl_ran_bernoulli (r, (double) 2/3);
	if (z_eta == 1)
	    eta = xi_1_eta;
	else
	    eta = xi_0_eta;
	gsl_rng_free (r); //free memory
}

//generate three mixtures of N(0,1) random variables. INPUT: a correlation between xi_1_eps and xi_1_eta1 (rho) and an alternative:
//choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
//dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: three random error terms (epsilon, eta1, and eta2).
void NL_Dgp::gen_TriMN (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double rho, const double rho1, const int choose_alt,
                        unsigned long seed) {
	(void)delta; //unused parameters
	(void)rho1;
	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double xi_0_eps = gsl_ran_ugaussian (r);
    double xi_0_eta = gsl_ran_ugaussian (r);
    double xi_1_eps = gsl_ran_ugaussian (r);
	double xi_1_eta1 = gsl_ran_ugaussian (r);
	double xi_1_eta2 = gsl_ran_ugaussian (r);
	switch (choose_alt) {
        case 0: //independent error terms
                break;
        case 1:
        	    gsl_ran_bivariate_gaussian (r, 1., 1., rho, &xi_1_eps, &xi_1_eta1); //correlated error terms
        	    break;
        case 2:
        	    if (fabs(xi_1_eps) <= 1.54) {//set threshold equal to 1.54 for zero correlation
	                xi_1_eta1 = xi_1_eps;
	                xi_1_eta2 = xi_1_eps;
	            }
	            else {
	                xi_1_eta1 = -xi_1_eps;
	                xi_1_eta2 = -xi_1_eps;
	            }
	            break;
	    case 3:
	            if (fabs(xi_1_eps) <= 2.0) {//set threshold equal to 2.0 for non-zero correlation
	                xi_1_eta1 = xi_1_eps;
	                xi_1_eta2 = xi_1_eps;
	            }
	            else {
	                xi_1_eta1 = -xi_1_eps;
	                xi_1_eta2 = -xi_1_eps;
	            }
	            break;
	    default:
    		    cerr << "NL_Dgp::gen_TriMN: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	}
	auto z_eps = gsl_ran_bernoulli (r, (double) 3/4);
	if (z_eps == 1)
	    epsilon = xi_1_eps;
	else
	    epsilon = xi_0_eps;
	auto z_eta1 = gsl_ran_bernoulli (r, (double) 2/3);
	if (z_eta1 == 1)
	    eta1 = xi_1_eta1;
	else
	    eta1 = xi_0_eta;
	auto z_eta2 = gsl_ran_bernoulli (r, (double) 3/5);
	if (z_eta2 == 1)
	    eta2 = xi_1_eta2;
	else
	    eta2 = xi_0_eta;
	gsl_rng_free (r); //free memory
}

//generate two centered chi-squared random variables. INPUT: a correlation between xi_1_eps and xi_1_eta (rho) and an alternative:
//choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
//dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: two random error terms (epsilon and eta).
void NL_Dgp::gen_ChiSq (double &epsilon, double &eta, const double delta, const double rho, const int choose_alt, unsigned long seed) {
	(void)delta; //unused parameter
	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double sigma = 1.;
    double xi_0_eps = gsl_ran_gaussian (r, sigma);
    double xi_0_eta = gsl_ran_gaussian (r, sigma);
    double xi_1_eps = gsl_ran_gaussian (r, sigma);
	double xi_1_eta = 0.;
	switch (choose_alt) {
        case 0:
                xi_1_eta = gsl_ran_ugaussian (r); //independent error terms
                break;
        case 1:
        	    gsl_ran_bivariate_gaussian (r, sigma, sigma, rho, &xi_1_eps, &xi_1_eta); //correlated error terms
        	    break;
        case 2:
        	    if (fabs(xi_1_eps) <= 1.54) //set threshold equal to 1.54 for zero correlation
	                xi_1_eta = xi_1_eps;
	            else
	                xi_1_eta = -xi_1_eps;
	            break;
	    case 3:
	            if (fabs(xi_1_eps) <= 2.) //set threshold equal to 2.0 for non-zero correlation
	                xi_1_eta = xi_1_eps;
	            else
	                xi_1_eta = -xi_1_eps;
	            break;
	    default:
    		    cerr << "NL_Dgp::gen_ChiSq: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	}
	epsilon = (pow(xi_0_eps, 2.) + pow(xi_1_eps, 2.) - 2)/4;
	eta     = (pow(xi_0_eta, 2.) + pow(xi_1_eta, 2.) - 2)/4;
	gsl_rng_free (r); //free memory
}

//generate three centered chi-squared random variables. INPUT: a correlation between xi_1_eps and xi_1_eta1 (rho) and an alternative:
//choose_alt = 0 to generate independent error terms, choose_alt = 1 to generate correlated error terms, choose_alt = 2 to generate uncorrelated but
//dependent error terms, choose_alt = 3 to generate correlated and dependent error terms. OUTPUT: three random error terms (epsilon, eta1, and eta2).
void NL_Dgp::gen_TriChiSq (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double rho, const double rho1, const int choose_alt, unsigned long seed) {
     (void)delta; //unused parameters
	(void)rho1;
	gsl_rng *r = nullptr;
     const gsl_rng_type *gen; //random number generator
     gsl_rng_env_setup();
     gen = gsl_rng_default;
     r = gsl_rng_alloc(gen);
     gsl_rng_set(r, seed);
     double sigma = 1.;
     double xi_0_eps = gsl_ran_gaussian (r, sigma);
     double xi_0_eta = gsl_ran_gaussian (r, sigma);
     double xi_1_eps = gsl_ran_gaussian (r, sigma);
	double xi_1_eta1 = gsl_ran_gaussian (r, sigma);
	double xi_1_eta2 = gsl_ran_gaussian (r, sigma);
	switch (choose_alt) {
        case 0: //independent error terms
                break;
        case 1:
        	    gsl_ran_bivariate_gaussian (r, sigma, sigma, rho, &xi_1_eps, &xi_1_eta1); //correlated error terms
        	    break;
        case 2:
        	    if (fabs(xi_1_eps) <= 1.54) {//set threshold equal to 1.54 for zero correlation
	                xi_1_eta1 = xi_1_eps;
	                xi_1_eta2 = xi_1_eps;
	            }
	            else {
	                xi_1_eta1 = -xi_1_eps;
	                xi_1_eta2 = -xi_1_eps;
	            }
	            break;
	    case 3:
	            if (fabs(xi_1_eps) <= 2.0) {//set threshold equal to 2.0 for non-zero correlation
	                xi_1_eta1 = xi_1_eps;
	                xi_1_eta2 = xi_1_eps;
	            }
	            else {
	                xi_1_eta1 = -xi_1_eps;
	                xi_1_eta2 = -xi_1_eps;
	            }
	            break;
	    default:
    		     cerr << "NL_Dgp::gen_TriChiSq: This choice is not in the switch list. Make sure that your choice is valid!\n";
               exit(0);
	}
	epsilon = (pow(xi_0_eps, 2.) + pow(xi_1_eps, 2.) - 2)/4;
	eta1     = (pow(xi_0_eta, 2.) + pow(xi_1_eta1, 2.) - 2)/4;
	eta2     = (pow(xi_0_eta, 2.) + pow(xi_1_eta2, 2.) - 2)/4;
	gsl_rng_free (r); //free memory
}

//generate two centered Beta random variables. INPUT: a shape parameter for the bivariate Beta distribution (alpha5 = 0.0001 gives corr = -0.145, alpha5 = 0.1 gives corr = -0.005,
//alpha5 = 1. gives corr = 0.365) and   choose_alt = 0 and 1. OUTPUT: two random error terms (epsilon and eta).
void NL_Dgp::gen_Beta (double &epsilon, double &eta, const double delta, const double alpha5_, const int choose_alt, unsigned long seed) {
     (void)delta; //unused parameters
     (void)alpha5_;
	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    auto alpha1 = 5., alpha2 = 5., alpha3 = 0.5, alpha4 = 0.5, alpha5 = 0.;
    switch (choose_alt) {
          case 0:  { //independent error terms
                    auto a_eps = alpha1+alpha3, b_eps = 1., a_eta = alpha2+alpha4, b_eta = 1.;
                    epsilon = gsl_ran_beta(r, a_eps, b_eps) -  a_eps/(a_eps+b_eps);
                    eta = gsl_ran_beta(r, a_eta, b_eta) - a_eta/(a_eta+b_eta);
               }
               break;
          case 1: { //correlation = -0.145
                    Matrix U(5,1);
                    alpha5 = 0.0001;
                    U(1) = gsl_ran_gamma (r, alpha1, 1.);  //alpha1 = alpha2 = 5.
                    U(2) = gsl_ran_gamma (r, alpha2, 1.);
                    U(3) = gsl_ran_gamma (r, alpha3, 1.); //alpha3 = alpha4 = 0.5
                    U(4) = gsl_ran_gamma (r, alpha4, 1.);
                    U(5) = gsl_ran_gamma (r, alpha5, 1.);
                    epsilon = ((double) (U(1) + U(3))/(U(1) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
                    eta     = ((double) (U(2) + U(4))/(U(2) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
               }
               break;
          case 2: { //correlation = -0.005
                    Matrix U(5,1);
                    alpha5 = 0.1;
                    U(1) = gsl_ran_gamma (r, alpha1, 1.);  //alpha1 = alpha2 = 5.
                    U(2) = gsl_ran_gamma (r, alpha2, 1.);
                    U(3) = gsl_ran_gamma (r, alpha3, 1.); //alpha3 = alpha4 = 0.5
                    U(4) = gsl_ran_gamma (r, alpha4, 1.);
                    U(5) = gsl_ran_gamma (r, alpha5, 1.);
                    epsilon = ((double) (U(1) + U(3))/(U(1) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
                    eta     = ((double) (U(2) + U(4))/(U(2) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
               }
               break;
          case 3: { //correlation = 0.365
                    Matrix U(5,1);
                    alpha5 = 1.;
                    U(1) = gsl_ran_gamma (r, alpha1, 1.);  //alpha1 = alpha2 = 5.
                    U(2) = gsl_ran_gamma (r, alpha2, 1.);
                    U(3) = gsl_ran_gamma (r, alpha3, 1.); //alpha3 = alpha4 = 0.5
                    U(4) = gsl_ran_gamma (r, alpha4, 1.);
                    U(5) = gsl_ran_gamma (r, alpha5, 1.);
                    epsilon = ((double) (U(1) + U(3))/(U(1) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
                    eta     = ((double) (U(2) + U(4))/(U(2) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
               }
               break;
          default:
    		     cerr << "NL_Dgp::gen_Beta: This choice is not in the switch list. Make sure that your choice is valid!\n";
               exit(0);
    }
    gsl_rng_free (r);//free memory
}

//generate three centered Beta random variables. INPUT: choose_alt = 0, 1, 2, and 3.
//OUTPUT: three random error terms (epsilon, eta1, and eta2).
void NL_Dgp::gen_TriBeta (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double rho, const double rho1, const int choose_alt, unsigned long seed) {
	(void)delta; //unused parameters
	(void)rho;
	(void)rho1;
	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
     auto alpha1 = 5., alpha2 = 5., alpha3 = 0.5, alpha4 = 0.5, alpha5 = 0.;
    switch (choose_alt) {
          case 0:  { //independent error terms
                    auto a_eps = alpha1+alpha3, b_eps = 1., a_eta = alpha2+alpha4, b_eta = 1.;
                    epsilon = gsl_ran_beta(r, a_eps, b_eps) -  a_eps/(a_eps+b_eps);
                    eta1 = gsl_ran_beta(r, a_eta, b_eta) - a_eta/(a_eta+b_eta);
               }
               break;
          case 1: { //correlation = -0.145
                    Matrix U(5,1);
                    alpha5 = 0.0001;
                    U(1) = gsl_ran_gamma (r, 5., 1.);
                    U(2) = gsl_ran_gamma (r, 5., 1.);
                    U(3) = gsl_ran_gamma (r, 0.5, 1.);
                    U(4) = gsl_ran_gamma (r, 0.5, 1.);
                    U(5) = gsl_ran_gamma (r, alpha5, 1.);
                    epsilon = ((double) (U(1) + U(3))/(U(1) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
                    eta1     = ((double) (U(2) + U(4))/(U(2) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
               }
               break;
          case 2: { //correlation = -0.005
                    Matrix U(5,1);
                    alpha5 = 0.1;
                    U(1) = gsl_ran_gamma (r, 5., 1.);
                    U(2) = gsl_ran_gamma (r, 5., 1.);
                    U(3) = gsl_ran_gamma (r, 0.5, 1.);
                    U(4) = gsl_ran_gamma (r, 0.5, 1.);
                    U(5) = gsl_ran_gamma (r, alpha5, 1.);
                    epsilon = ((double) (U(1) + U(3))/(U(1) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
                    eta1     = ((double) (U(2) + U(4))/(U(2) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
               }
               break;
          case 3: { //correlation = 0.365
                    Matrix U(5,1);
                    alpha5 = 1.;
                    U(1) = gsl_ran_gamma (r, 5., 1.);
                    U(2) = gsl_ran_gamma (r, 5., 1.);
                    U(3) = gsl_ran_gamma (r, 0.5, 1.);
                    U(4) = gsl_ran_gamma (r, 0.5, 1.);
                    U(5) = gsl_ran_gamma (r, alpha5, 1.);
                    epsilon = ((double) (U(1) + U(3))/(U(1) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
                    eta1     = ((double) (U(2) + U(4))/(U(2) + U(3) + U(4) + U(5)) - (5 + 0.5)/(5 + 0.5 + 0.5 + alpha5));
               }
               break;
          default:
    		     cerr << "NL_Dgp::gen_Beta: This choice is not in the switch list. Make sure that your choice is valid!\n";
               exit(0);
    }
    double xi = gsl_ran_beta(r, alpha1+alpha3, 1.) -  (alpha1+alpha3) / (alpha1+alpha3+1.);
    eta2 = eta1 + xi;
    gsl_rng_free (r);//free memory
}

//generate two centered exponential random variables. INPUT: an alternative: choose_alt = 0 to generate independent error terms,
//choose_alt = 1 to generate very weakly dependent error terms, choose_alt = 2 to generate weak dependent error terms, choose_alt = 3 to generate dependent error terms.
//OUTPUT: two random error terms (epsilon and eta).
void NL_Dgp::gen_Exp (double &epsilon, double &eta, const double delta, const double rho, const int choose_alt, unsigned long seed) {
     (void)delta; //unused parameter
     (void)rho;
     gsl_rng *r = nullptr;
     const gsl_rng_type *gen; //random number generator
     gsl_rng_env_setup();
     gen = gsl_rng_default;
     r = gsl_rng_alloc(gen);
     gsl_rng_set(r, seed);
     double beta = 1.5, gamma = 1.5;

     //generate a bivariate random variable with Arnold & Strauss's (1988) exponential distribution by using Yu's (2009) rejection algorithm
     auto gen_BEV = [beta, gamma, r](double &x, double &y, double rho0) {
          double u1 = 0., u2 = 0., u3 = 0., x_;
          double theta = -rho0*exp(-1./rho0) / gsl_sf_expint_Ei(-1/rho0);
     restart:
          u1 = gsl_rng_uniform_pos(r);
          u2 = gsl_rng_uniform_pos(r);
          x_ = -log(u1);
          if (u2 < 1./(1.+rho0*x_)) {
               u3 = gsl_rng_uniform_pos(r);
               x = x_/beta - (theta-1)/(beta*rho0);
               y = -log(u3)/(gamma*(1+rho0*x_)) - (theta-1)/(gamma*rho0);
          }
          else {
               goto restart;
          }
          return 0;
     };
	switch (choose_alt) {
        case 0: { //independent error terms
                    epsilon = gsl_ran_exponential(r, 1./beta) - 1./beta;
                    eta = gsl_ran_exponential(r, 1./gamma) - 1./gamma;
               }
               break;
        case 1: { //very weak dependence
                    double rho1 = 0.01;
                    gen_BEV(epsilon, eta, rho1);
               }
               break;
        case 2:{ //weak dependence
                    double rho1 = 0.05;
                    gen_BEV(epsilon, eta, rho1);
               }
               break;
	    case 3: { //strong dependence
                    double rho1 = 0.5;
                    gen_BEV(epsilon, eta, rho1);
               }
               break;
	    default:
    		    cerr << "NL_Dgp::gen_Exp: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	}
	gsl_rng_free (r); //free memory
}

//generate three centered exponential random variables. INPUT: an alternative: choose_alt = 0 to generate independent error terms,
//choose_alt = 1 to generate very weakly dependent error terms, choose_alt = 2 to generate weak dependent error terms, choose_alt = 3 to generate dependent error terms.
//OUTPUT: three random error terms (epsilon, eta1, and eta2).
void NL_Dgp::gen_TriExp (double &epsilon, double &eta1, double &eta2, const Matrix &delta, const double rho, const double rho_, const int choose_alt, unsigned long seed) {
     (void)delta; //unused parameters
     (void)rho;
	(void)rho_;
	gsl_rng *r = nullptr;
     const gsl_rng_type *gen; //random number generator
     gsl_rng_env_setup();
     gen = gsl_rng_default;
     r = gsl_rng_alloc(gen);
     gsl_rng_set(r, seed);
     double beta = 1.5, gamma = 1.5;

     //generate a bivariate random variable with Arnold & Strauss's (1988) exponential distribution by using Yu's (2009) rejection algorithm
     auto gen_BEV = [beta, gamma, r](double &x, double &y, double rho0) {
          double u1 = 0., u2 = 0., u3 = 0., x_;
          double theta = -rho0*exp(-1./rho0) / gsl_sf_expint_Ei(-1/rho0);
     restart:
          u1 = gsl_rng_uniform_pos(r);
          u2 = gsl_rng_uniform_pos(r);
          x_ = -log(u1);
          if (u2 < 1./(1.+rho0*x_)) {
               u3 = gsl_rng_uniform_pos(r);
               x = x_/beta - (theta-1)/(beta*rho0);
               y = -log(u3)/(gamma*(1+rho0*x_)) - (theta-1)/(gamma*rho0);
          }
          else {
               goto restart;
          }
          return 0;
     };
	switch (choose_alt) {
        case 0: { //independent error terms
                    epsilon = gsl_ran_exponential(r, 1./beta) - 1./beta;
                    eta1 = gsl_ran_exponential(r, 1./gamma) - 1./gamma;
               }
               break;
        case 1: { //very weak dependence
                    double rho1 = 0.01;
                    gen_BEV(epsilon, eta1, rho1);
               }
               break;
        case 2:{ //weak dependence
                    double rho1 = 0.05;
                    gen_BEV(epsilon, eta1, rho1);
               }
               break;
	    case 3: { //strong dependence
                    double rho1 = 0.5;
                    gen_BEV(epsilon, eta1, rho1);
               }
               break;
	    default:
    		    cerr << "NL_Dgp::gen_TriExp: This choice is not in the switch list. Make sure that your choice is valid!\n";
                exit(0);
	}
	eta2 = eta1 + gsl_ran_exponential(r, 1./gamma) - 1./gamma;
	gsl_rng_free (r); //free memory
}

//generate two vectors of random errors (epsilon and eta). INPUT: constants (delta and rho in (-1,1)), an alternative (choose_alt = 0, 1, 2), and a
//random generator template (gen_RAN: gen_SN, gen_MN, gen_ChiSq or gen_Beta). OUTPUT: 2 vectors
template <void gen_RAN (double &, double &, const double, const double, const int, unsigned long)>
void NL_Dgp::gen_RANV (Matrix &epsilon, Matrix &eta, const double delta, const double rho, const int choose_alt, unsigned long seed) {
	gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long rseed = 1;
    for (auto i = 1; i <= epsilon.nRow(); i++) {
    	rseed = gsl_rng_get (r);
    	gen_RAN (epsilon(i), eta(i), delta, rho, choose_alt, rseed); //use a random seed
	}
    gsl_rng_free (r); //free memory
}

//generate three vectors of random errors (epsilon, eta1 and eta2). INPUT: a vector (delta in (-1,1)^3) and constants (rho12 and rho13 in (-1,1)),
//an alternative (choose_alt = 0, 1, 2), and a random generator template (gen_RAN: gen_TriSN, gen_TriMN, gen_TriChiSq or gen_TriBeta). OUTPUT: 3 vectors
template <void gen_RAN (double &, double &, double &, const Matrix &, const double, const double, const int, unsigned long)>
void NL_Dgp::gen_RANV (Matrix &epsilon, Matrix &eta1, Matrix &eta2, const Matrix &delta, const double rho12, const double rho13, const int choose_alt,
                       unsigned long seed) {
    gsl_rng *r = nullptr;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_default;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    unsigned long rseed = 1;
    for (auto i = 1; i <= epsilon.nRow(); i++) {
    	rseed = gsl_rng_get (r);
    	gen_RAN (epsilon(i), eta1(i), eta2(i), delta, rho12, rho13, choose_alt, rseed);
    	//cout << epsilon (i) << " , " << eta1(i) << " , " << eta2(i) << endl;
	}
    gsl_rng_free (r); //free memory
}

//run multivariate OLS. INPUT: a Tx1 vector of data on the dependent (Y) and a TxN matrix of data on the independent (X).
//OUTPUT: a Tx1 vector of residuals (resid) and a Nx1 vector of the OLS estimates (slope)
void NL_Dgp::gen_Resid (Matrix &resid, Matrix &slope, const Matrix X, const Matrix Y) {
	//auto T = X.nRow();
	auto N = X.nCol();
	Matrix denom(N,N), num(N,1), denom_Inv(N,N);
	denom = Tr(X)*X;
	denom_Inv = inv(denom);
	num = Tr(X)*Y;
	slope = denom_Inv*num;
	resid = Y - X*slope;
}

//estimate the bilinear regression model by the OLS. INPUT: a Tx1 vector of data on X.
//OUTPUT: a (T-2)x1 vector of residuals (resid) and a 6x1 vector of the OLS estimates (slope)
void NL_Dgp::est_BL (Matrix &resid, Matrix &slope, const Matrix X) {
	auto T = X.nRow(), row = T-2, col = 6;
	Matrix Z(row,col), Y(row,1);
	for (auto t = 1; t <= row; ++t) {
		Z(t,1) = 1.;
		Z(t,2) = X(t+1);
		Z(t,3) = pow(Z(t,2), 2.);
		Z(t,4) = X(t);
		Z(t,5) = pow(Z(t,4), 2.);
		Z(t,6) = Z(t,2) * Z(t,4);
		Y(t) = X(t+2);
	}
	NL_Dgp::gen_Resid (resid, slope, Z, Y);
}

//estimate the TAR regression model by the OLS. INPUT: a Tx1 vector of data on X.
//OUTPUT: a (T-2)x1 vector of residuals (resid) and a 5x1 vector of the OLS estimates (slope)
void NL_Dgp::est_TAR (Matrix &resid, Matrix &slope, const Matrix X) {
	auto T = X.nRow(), row = T-2, col = 5;
	Matrix Z(row,col), Y(row,1);
	auto pos = [](double x) { //get the positive part of a real number (x)
    	if (x > 0.)
    		return x;
    	else
    		return 0.;
	};
	for (auto t = 1; t <= row; ++t) {
		Z(t,1) = 1.;
		Z(t,2) = X(t+1);
		Z(t,3) = pos(Z(t,2));
		Z(t,4) = X(t);
		Z(t,5) = pos(Z(t,4));
		Y(t) = X(t+2);
	}
	NL_Dgp::gen_Resid (resid, slope, Z, Y);
}






















#endif


