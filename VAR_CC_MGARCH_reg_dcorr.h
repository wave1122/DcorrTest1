#ifndef VAR_CC_MGARCH_REG_DCORR_H_
#define VAR_CC_MGARCH_REG_DCORR_H_

#include <unistd.h>
#include <omp.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/Statistics.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include<mgarch.h>

#define CHUNK 1

using namespace std;
using namespace shogun;
using namespace shogun::linalg;

class VAR_CC_MGARCH_REG_DCORR {
	public:
		VAR_CC_MGARCH_REG_DCORR () {   };//default constructor
		~VAR_CC_MGARCH_REG_DCORR () {   };//default destructor

		/* Calculate the conditional expectations of  ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
		using VAR-MGARCH(1,1) as the model (2.15).
		OUTPUT: a T0 = T-1 by T-1 matrix (mat_reg_first, mat_reg_second, mat_breg). */
		static void reg(SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
						SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
						SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
						const SGMatrix<double> &X, /*T by 2 matrix of observations*/
						const double expn, /*exponent of distances*/
						SGVector<double> theta_mgarch0 /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/);



		/* Calculate centered Euclidean distances by ML methods.
		INPUT: T by 2 matrix of data (X), and an exponent of the distance correlation (expn).
		OUTPUT: a T0 = T-1 by T-1 matrix (mat_U). */
		static SGMatrix<double> var_U(	const SGMatrix<double> &X,
										const double expn, /*an exponent of the Euclidean distance*/
										SGVector<double> theta_mgarch0 /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/);

};




/* Calculate the conditional expectations of  ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
using VAR-MGARCH(1,1) as the model (2.15).
OUTPUT: a T0 = T-1 by T-1 matrix (mat_reg_first, mat_reg_second, mat_breg). */
void VAR_CC_MGARCH_REG_DCORR::reg(	SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
									SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
									SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
									const SGMatrix<double> &X, /*T by 2 matrix of observations*/
									const double expn, /*exponent of distances*/
									SGVector<double> theta_mgarch0 /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/) {
	auto T = X.num_rows, N = X.num_cols;
	auto L = 1, T0 = T - L;
	ASSERT_(T0 == mat_reg_first.num_rows && T0 == mat_reg_first.num_cols && T0 == mat_breg.num_rows && T0 == mat_breg.num_cols && N == 2);

	mat_reg_first.zero(); // reset the output matrix
	mat_reg_second.zero();
	mat_breg.zero();

	auto t = 0, s = 0, tau = 0, tau1 = 0, tau2 = 0;

	/* 1. Estimate E[x_t | lags of x_t] */

	SGVector<double> theta_var(4), theta_mgarch(7);
	SGMatrix<double> resid_mgarch(T0, N), mu_trans(N, T), mu(T, N);
	mu_trans.zero();
	mu.zero();


	cc_mgarch::ols_mle_simplex(	resid_mgarch, /*T-1 by 2 matrix of residuals for CC-MGARCH*/
								theta_var, /*4 by 1 vector of estimates for the VAR part*/
								theta_mgarch, /*7 by 1 vector of estimates for the CC-MGARCH part*/
								X, /*T by 2 matrix of observations*/
								theta_mgarch0 /*7 by 1 vector of initial parameters*/ );

	SGMatrix<double> theta_var_mat(theta_var, 2, 2);
	theta_var_mat = transpose_matrix(theta_var_mat);


	for (int t = 1; t < T; ++t) {
		mu_trans.set_column( t, matrix_prod( theta_var_mat, X.get_row_vector(t-1) ) );
	}
	mu = transpose_matrix(mu_trans);

	/* 2. Estimate Var[x_t | lags of x_t] and  the sqrt of Cov[x_1t, x_2t | lags of x_t] */

	SGVector<double> eigenval(N);
	SGMatrix<double> v(N, N), eigenvect_mat(N, N), eigenval_mat(N, N), cov_mat_sqrt(N, N), var_mat_sqrt(T0, N);
	SGVector<double> cov_vec_sqrt(T0);
	v.zero();
	cov_mat_sqrt.zero(); // reset all matrices
	eigenvect_mat.zero();
	eigenval_mat.zero();
	for (t = 0; t < T0; ++t) {
		v(0,0) = theta_mgarch[0] + theta_mgarch[1]*v(0,0) + theta_mgarch[2]*pow(X(t,0)-mu(t,0), 2.);
		v(1,1) = theta_mgarch[3] + theta_mgarch[4]*v(1,1) + theta_mgarch[5]*pow(X(t,1)-mu(t,1), 2.);
		v(0,1) = theta_mgarch[6] * sqrt( v(0,0) * v(1,1) );
		v(1,0) = v(0,1);

		//use the eigenvalue decomposition ('cov_mat_sqrt' will be symmetric)
		eigen_solver(v, eigenval, eigenvect_mat);
		for (int i = 0; i < N; ++i)
			eigenval_mat(i,i) = sqrt(eigenval[i]);
		cov_mat_sqrt = matrix_prod( eigenvect_mat, matrix_prod(eigenval_mat, eigenvect_mat, false, true) );


		/*// use the Cholesky decomposition ('cov_mat_sqrt' will be lower triangular)
		cov_mat_sqrt(0,0) = sqrt( v(0,0) );
		cov_mat_sqrt(1,0) = v(0,1) / sqrt( v(0,0) );
		cov_mat_sqrt(0,1) = cov_mat_sqrt(1,0);
		cov_mat_sqrt(1,1) = pow( ( v(0,0)*v(1,1) - pow(v(0,1), 2.) ) / v(0,0), 0.5 );*/

		var_mat_sqrt(t,0) = cov_mat_sqrt(0,0);
		var_mat_sqrt(t,1) = cov_mat_sqrt(1,1);
		cov_vec_sqrt[t] = cov_mat_sqrt(0,1);
	}

	/* 3. Estimate E[||x_t - x_s||^\kappa| lags of x_t] */
	SGVector<double> row_vec(N);

	//#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,i,j,s,tau) firstprivate(counter,cov_mat,cov_mat_sqrt,row_vec)
	for (t = 0; t < T0; ++t) {
		cov_mat_sqrt(0,0) = var_mat_sqrt(t,0);
		cov_mat_sqrt(1,1) = var_mat_sqrt(t,1);
		cov_mat_sqrt(0,1) = cov_vec_sqrt[t];
		cov_mat_sqrt(1,0) = cov_mat_sqrt(0,1);

		for (s = 0; s < T0; ++s) {
			for (tau = 0; tau < T0; ++tau) {
				row_vec = add(add(mu.get_row_vector(t+L), matrix_prod( cov_mat_sqrt, resid_mgarch.get_row_vector(tau) ), 1., 1.), X.get_row_vector(s+L), 1., -1.);

				//#pragma omp atomic
				mat_reg_first(t, s) += std::pow(norm(row_vec), expn) / T0;
			}
		}
	}

	/* 4. Estimate E[||x_t - x_s||^\kappa| lags of x_s] */

	mat_reg_second = transpose_matrix(mat_reg_first);
	//mat_reg_second.display_matrix("mat_reg_second");


	/* 5. Estimate E[||x_t - x_s||^\kappa| lags of x_t and x_s] */

	//SGMatrix<double> cov_mat_sqrt_t(N, N), cov_mat_sqrt_s(N, N);

	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,tau1,tau2) //firstprivate(cov_mat_sqrt_t,cov_mat_sqrt_s,row_vec)
	for (t = 0; t < T0; ++t) {
		SGMatrix<double> cov_mat_sqrt_t(N, N), cov_mat_sqrt_s(N, N);
		SGVector<double> row_vec(N);

        cov_mat_sqrt_t(0,0) = var_mat_sqrt(t,0);
		cov_mat_sqrt_t(1,1) = var_mat_sqrt(t,1);
		cov_mat_sqrt_t(0,1) = cov_vec_sqrt[t];
		cov_mat_sqrt_t(1,0) = cov_mat_sqrt_t(0,1);

        for (s = t+1; s < T0; ++s) { //first ver.: s = t
            cov_mat_sqrt_s(0,0) = var_mat_sqrt(s,0);
			cov_mat_sqrt_s(1,1) = var_mat_sqrt(s,1);
			cov_mat_sqrt_s(0,1) = cov_vec_sqrt[s];
			cov_mat_sqrt_s(1,0) = cov_mat_sqrt_s(0,1);

            for (tau1 = 0; tau1 < T0; ++tau1) {
                for (tau2 = 0; tau2 < T0; ++tau2) {
					row_vec = add(add(mu.get_row_vector(t+L), matrix_prod( cov_mat_sqrt_t, resid_mgarch.get_row_vector(tau1) ), 1., 1.), \
								  add(mu.get_row_vector(s+L), matrix_prod( cov_mat_sqrt_s, resid_mgarch.get_row_vector(tau2) ), 1., 1.), 1., -1.);

					//#pragma omp atomic
					mat_breg(t,s) += std::pow(norm(row_vec), expn) / pow(T0, 2.);
                }
            }
            mat_breg(s, t) = mat_breg(t,s);
        }
	}
	//mat_breg.display_matrix("mat_breg");
}


/* Calculate centered Euclidean distances by ML methods.
INPUT: T by 2 matrix of data (X), and an exponent of the distance correlation (expn).
OUTPUT: a T0 = T-1 by T-1 matrix (mat_U). */
SGMatrix<double> VAR_CC_MGARCH_REG_DCORR::var_U(const SGMatrix<double> &X,
												const double expn, /*an exponent of the Euclidean distance*/
												SGVector<double> theta_mgarch0 /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/) {
	auto T = X.num_rows, N = X.num_cols;
	auto L = 1, T0 = T - L;

	ASSERT_(N == 2 && theta_mgarch0.vlen == 7);

	SGMatrix<double> mat_reg_first(T0, T0), mat_reg_second(T0, T0), mat_breg(T0, T0), mat_U(T0, T0);
	mat_U.zero(); // reset the output matrix

	// Calculate conditional expectations
	VAR_CC_MGARCH_REG_DCORR::reg(	mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
									mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
									mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
									X, /*T by 2 matrix of observations*/
									expn, /*exponent of distances*/
									theta_mgarch0 /*7 by 1 vector of initial parameters to estimate the CC-MGARCH part*/);

	// Calculate 'mat_U'
	int t = 0, s = 0;
	//SGVector<double> row_vec(N);

	#pragma omp parallel for default(shared) schedule(static,CHUNK) private(t,s)
	for (t = L; t < T; ++t) {
		for (s = L; s < T; ++s) {
			//row_vec = add(X.get_row_vector(t), X.get_row_vector(s), 1., -1.);
			mat_U(t-L, s-L) = std::pow(shogun::linalg::norm( add(X.get_row_vector(t), X.get_row_vector(s), 1., -1.) ), expn) - mat_reg_first(t-L, s-L) \
																										- mat_reg_second(t-L, s-L) + mat_breg(t-L, s-L);
			//mat_U(s-L, t-L) = mat_U(t-L,s-L);
		}
	}

	// zero out the diagonal elements
	// for (t = 0; t < T0; ++t) //first ver.:
		// mat_U(t, t) = 0.;

	return mat_U;
}






#endif

