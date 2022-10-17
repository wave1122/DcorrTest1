#ifndef MGARCH_H
#define MGARCH_H

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
#include <nmsimplex.h>
#include <nl_dgp.h>


#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/util/factory.h>




using namespace std;
using namespace shogun;
using namespace shogun::linalg;

class cc_mgarch {
	public:
		cc_mgarch() {   }; //default constructor
		~cc_mgarch() {   };//default destructor

		/* Define the negative logarithm of the Gaussian quasi-likelihood function for the CC-MGARCH model defined on page 367 in Tse (2002) */
		static double neg_loglikelihood(const SGVector<double> &theta, /*a 7 by 1 vector of parameters*/
										const SGMatrix<double> &Y /*a T by 2 matrix of data*/);

		/* Define the negative logarithm of the Gaussian quasi-likelihood function for a VAR-CC-MGARCH(1,1) model */
		static double neg_loglikelihood_var_mgarch(	const SGVector<double> &theta, /*a 11 by 1 vector of parameters*/
													const SGMatrix<double> &Y /*a T by 2 matrix of data*/);

		/* Define a GSL wrapper for 'neg_loglikelihood' of the CC-MGARCH model */
		static double neg_loglikelihood_simplex(const gsl_vector *theta0, /*a 7 by 1 vector*/
												void *parms /*a 2T+1 by 1 vector*/);

		/* Approximate the first-order derivative of the log likelihood function */
		static SGVector<double> neg_loglikelihood_gradient(	const SGVector<double> &theta, /*a 7 by 1 vector of parameters*/
															const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
															double h /*finite differential level*/);

		/* Approximate the first-order derivative of the log likelihood function of a VAR-CC-MGARCH(1,1) model */
		static SGVector<double> neg_loglikelihood_gradient_var_mgarch(	const SGVector<double> &theta, /*a 11 by 1 vector of parameters*/
																		const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
																		double h /*finite differential level*/);

		/* Approximate the second-order derivatives of the log likelihood function */
		static SGMatrix<double> neg_loglikelihood_hessian(	const SGVector<double> &theta, /*a 7 by 1 vector of parameters*/
															const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
															double h /*finite differential level*/ );

		/* Approximate the second-order derivatives of the log likelihood function of a VAR-CC-MGARCH(1,1) model */
		static SGMatrix<double> neg_loglikelihood_hessian_var_mgarch(	const SGVector<double> &theta, /*a 11 by 1 vector of parameters*/
																		const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
																		double h /*finite differential level*/ );

		/* Compute the ML estimates for the CC-MGARCH model.
		OUTPUT: ML estimtes (theta) and the value of the log likelihood function */
		static double mle_simplex(	SGMatrix<double> &resid, /*T by 2 matrix of residuals*/
									SGVector<double> &theta, /*7 by 1 vector*/
									const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
									SGVector<double> theta0 /*initial parameters*/);

		/* Compute estimates for VAR-CC-MGARCH(1,1) model.
		OUTPUT: OLS estimates (theta_var) and ML estimates (theta_mgarch) and the minimum value of the negative log likelihood */
		static double ols_mle_simplex( 	SGMatrix<double> &resid_mgarch, /*T-1 by 2 matrix of residuals for CC-MGARCH*/
										SGVector<double> &theta_var, /*4 by 1 vector of estimates for the VAR part*/
										SGVector<double> &theta_mgarch, /*7 by 1 vector of estimates for the CC-MGARCH part*/
										const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
										SGVector<double> theta_mgarch0 /*7 by 1 vector of initial parameters*/ );

		/* Compute estimates for VAR-CC-MGARCH(1,1) model.*/
		double ols_mle_simplex( SGMatrix<double> &resid_mgarch, /*T by 2 matrix of residuals for CC-MGARCH*/
								SGVector<double> &theta_var_mgarch, /*11 by 1 vector of parameter estimates*/
								const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
								SGVector<double> theta_var_mgarch0 /*11 by 1 vector of initial parameters*/);

	private:
		/* Define a basis vector */
		static SGVector<double> base_vec(int i, int n);
};

SGVector<double> cc_mgarch::base_vec(int i, int n) {
	SGVector<double> e(n);
	e.set_const(0.);
	e[i] = 1.;
	return e;
}


/* Define the negative logarithm of the Gaussian quasi-likelihood function for the CC-MGARCH model defined on page 367 in Tse (2002) */
double cc_mgarch::neg_loglikelihood(const SGVector<double> &theta, /*a 7 by 1 vector of parameters*/
									const SGMatrix<double> &Y /*a T by 2 matrix of data*/) {
	int T = Y.num_rows, dim = theta.vlen;
	ASSERT_(dim == 7);

	SGMatrix<double> v(2,2), v_inv(2,2);
	v.set_const(1e-2); // set initial values for the conditional variance-covariance matrix

	SGVector<double> y_t(2);
	double ell = 0.;
	for (int t = 1; t < T; ++t) {
		v(0,0) = theta[0] + theta[1]*v(0,0) + theta[2]*pow(Y(t-1,0), 2.);
		v(1,1) = theta[3] + theta[4]*v(1,1) + theta[5]*pow(Y(t-1,1), 2.);
		v(0,1) = theta[6] * sqrt( v(0,0) * v(1,1) );
		v(1,0) = v(0,1);

		v_inv = pinv<double>(v); // invert the variance-covariance matrix
		y_t = Y.get_row_vector(t);
		ell += dot( y_t, matrix_prod(v_inv, y_t) ) + log( v(0,0)*v(1,1) - pow(v(0,1), 2.) );
	}

	return ell / T;
}

/* Define the negative logarithm of the Gaussian quasi-likelihood function for a VAR-CC-MGARCH(1,1) model */
double cc_mgarch::neg_loglikelihood_var_mgarch(	const SGVector<double> &theta, /*a 11 by 1 vector of parameters*/
												const SGMatrix<double> &Y /*a T by 2 matrix of data*/) {
	int T = Y.num_rows;
	ASSERT_(theta.vlen == 11);

	SGVector<double> theta_var(4), theta_mgarch(7);
	theta_var = get_subvector(theta, 0, 3);
	theta_mgarch = get_subvector(theta, 4, 10);

	// Compute conditional means
	SGMatrix<double> theta_var_mat(theta_var, 2, 2);
	theta_var_mat = transpose_matrix(theta_var_mat);

	SGMatrix<double> mu_trans(2, T), mu(T, 2);
	for (int t = 1; t < T; ++t) {
		mu_trans.set_column(t, matrix_prod( theta_var_mat, Y.get_row_vector(t-1) ) );
	}
	mu = transpose_matrix(mu_trans);


	// Obtain residuals from VAR(1)
	SGMatrix<double> resid_trans(2, T), resid(T, 2);
	resid_trans.zero();
	for (int t = 1; t < T; ++t) {
		resid_trans.set_column( t, add(Y.get_row_vector(t), mu.get_row_vector(t), 1., -1.) );
	}
	resid = transpose_matrix(resid_trans);

	// Compute the log likelihood function
	return cc_mgarch::neg_loglikelihood(theta_mgarch, resid);
}



/* Approximate the first-order derivative of the log likelihood function */
SGVector<double> cc_mgarch::neg_loglikelihood_gradient(	const SGVector<double> &theta, /*a 7 by 1 vector of parameters*/
														const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
														double h /*finite differential level*/) {
	int dim = theta.vlen;
	ASSERT_(dim == 7);

	SGVector<double> theta1(dim), theta2(dim), df(dim);

	for (int i = 0; i < dim; ++i) {
		theta1 = add(theta, base_vec(i, dim), 1., h);
		theta2 = add(theta, base_vec(i, dim), 1., -h);
		df[i] = ( cc_mgarch::neg_loglikelihood(theta1, Y) - cc_mgarch::neg_loglikelihood(theta2, Y) ) / (2*h);
	}

	return df;
}

/* Approximate the first-order derivative of the log likelihood function of a VAR-CC-MGARCH(1,1) model */
SGVector<double> cc_mgarch::neg_loglikelihood_gradient_var_mgarch(	const SGVector<double> &theta, /*a 11 by 1 vector of parameters*/
																	const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
																	double h /*finite differential level*/) {
	int dim = theta.vlen;
	ASSERT_(dim == 11);

	SGVector<double> theta1(dim), theta2(dim), df(dim);

	for (int i = 0; i < dim; ++i) {
		theta1 = add(theta, base_vec(i, dim), 1., h);
		theta2 = add(theta, base_vec(i, dim), 1., -h);
		df[i] = ( cc_mgarch::neg_loglikelihood_var_mgarch(theta1, Y) - cc_mgarch::neg_loglikelihood_var_mgarch(theta2, Y) ) / (2*h);
	}

	return df;
}


/* Approximate the second-order derivatives of the log likelihood function */
SGMatrix<double> cc_mgarch::neg_loglikelihood_hessian(	const SGVector<double> &theta, /*a 7 by 1 vector of parameters*/
														const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
														double h /*finite differential level*/ ) {
	int dim = theta.vlen;

	SGVector<double> theta1(dim), theta2(dim), theta3(dim);
	SGMatrix<double> df(dim, dim);

	for (int i = 0; i < dim; ++i) {
		theta2 = add(theta, base_vec(i,dim), 1., h);
		for (int j = 0; j < dim; ++j) {
			theta1 = add(theta2, base_vec(j,dim), 1., h);
			theta3 = add(theta, base_vec(j,dim), 1., h);
			df(i, j) = ( cc_mgarch::neg_loglikelihood(theta1, Y) - cc_mgarch::neg_loglikelihood(theta2, Y) - cc_mgarch::neg_loglikelihood(theta3, Y) \
																				+ cc_mgarch::neg_loglikelihood(theta, Y) ) / pow(h,2.);
		}
	}
	return df;
}

/* Approximate the second-order derivatives of the log likelihood function of a VAR-CC-MGARCH(1,1) model */
SGMatrix<double> cc_mgarch::neg_loglikelihood_hessian_var_mgarch(	const SGVector<double> &theta, /*a 11 by 1 vector of parameters*/
																	const SGMatrix<double> &Y, /*a T by 2 matrix of data*/
																	double h /*finite differential level*/ ) {
	int dim = theta.vlen;
	ASSERT_(dim == 11);

	SGVector<double> theta1(dim), theta2(dim), theta3(dim);
	SGMatrix<double> df(dim, dim);

	for (int i = 0; i < dim; ++i) {
		theta2 = add(theta, base_vec(i,dim), 1., h);
		for (int j = 0; j < dim; ++j) {
			theta1 = add(theta2, base_vec(j,dim), 1., h);
			theta3 = add(theta, base_vec(j,dim), 1., h);
			df(i, j) = ( cc_mgarch::neg_loglikelihood_var_mgarch(theta1, Y) - cc_mgarch::neg_loglikelihood_var_mgarch(theta2, Y) \
							- cc_mgarch::neg_loglikelihood_var_mgarch(theta3, Y) + cc_mgarch::neg_loglikelihood_var_mgarch(theta, Y) ) / pow(h,2.);
		}
	}
	return df;
}


/* Define a GSL wrapper for 'neg_loglikelihood'*/
double cc_mgarch::neg_loglikelihood_simplex(const gsl_vector *theta0, /*a 7 by 1 vector*/
											void *parms /*a 2T+1 by 1 vector*/) {
	double *p = (double *) parms; // p is a 2*T+1 by 1 vector

	int T = p[0];
    SGMatrix<double> Y(T,2);
    for (int t = 0; t < T; ++t) { // assign data to a SGMatrix
		Y(t,0) = p[t+1];
		Y(t,1) = p[T+t+1];
    }

	int dim = 7;
	SGVector<double> theta(dim);
	for (int i = 0; i < dim; ++i) { // get all parameter values to a vector, 'tanh_theta'
		theta[i] = gsl_vector_get(theta0, i);
	}

	/* calculate the negative log likelihood function */
	double ell = 0.;
	if ( (Math::min<double>(theta, dim-1) >= 0.) && (theta[6] > -1.) && (theta[6] < 1.) ) { // impose the positiveness constraint
		ell = cc_mgarch::neg_loglikelihood(theta, Y);
		//cout << "ell = " << ell << endl;
	}
	else
		ell = 1000;

	return ell;
}

/* Compute the ML estimates for the CC-MGARCH model.
OUTPUT: ML estimtes (theta) and the value of the (negative) log likelihood function */
double cc_mgarch::mle_simplex(	SGMatrix<double> &resid, /*T by 2 matrix of residuals*/
								SGVector<double> &theta, /*7 by 1 vector*/
								const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
								SGVector<double> theta0 /*initial parameters*/) {
	int T = Y.num_rows, dim = theta.vlen;
	ASSERT_(resid.num_rows == T && dim == theta0.vlen);

	double *par = new double[2*T+1];
	par[0] = T;
	for (int t = 0; t < T; ++t) { // assign data to a pointer array
		par[t+1] = Y(t,0);
		par[T+t+1] = Y(t,1);
    }

	gsl_vector *init = gsl_vector_alloc(dim); //set initial values for the minimizer
	for (int i = 0; i < dim; i++)
		gsl_vector_set(init, i, theta0[i]);

	double fmin = minimization<cc_mgarch::neg_loglikelihood_simplex>(theta, init, par);

	// compute residuals
	resid = NL_Dgp::resid_CC_MGARCH(Y, theta);

	delete [] par; //free up memory
	gsl_vector_free(init);

	// theta.display_vector("MLE");
	// cout << "the value of the function = " << fmin << endl;
	return fmin;
}



/* Compute estimates for VAR-CC-MGARCH(1,1) model.
OUTPUT: OLS estimates (theta_var) and ML estimates (theta_mgarch) and the minimum value of the negative log likelihood */
double cc_mgarch::ols_mle_simplex( 	SGMatrix<double> &resid_mgarch, /*T-1 by 2 matrix of residuals for CC-MGARCH*/
									SGVector<double> &theta_var, /*4 by 1 vector of estimates for the VAR part*/
									SGVector<double> &theta_mgarch, /*7 by 1 vector of estimates for the CC-MGARCH part*/
									const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
									SGVector<double> theta_mgarch0 /*7 by 1 vector of initial parameters*/ ) {
	int T = Y.num_rows, dim_var = theta_var.vlen, dim_mgarch = theta_mgarch.vlen;
	ASSERT_(resid_mgarch.num_rows == T-1 && dim_mgarch == theta_mgarch0.vlen);

	/* 1. Estimate the VAR part by the OLS */
	SGMatrix<double> Phi(2, 3);
	SGMatrix<double> resid_var = NL_Dgp::resid_VAR(Phi, Y, 1);
	theta_var[0] = Phi(0,1);
	theta_var[1] = Phi(0,2);
	theta_var[2] = Phi(1,1);
	theta_var[3] = Phi(1,2);

	/* 2. Fit 'resid_var' to the CC-MGARCH model */
	SGMatrix<double> resid_mgarch1(T, 2);

	double fmin = cc_mgarch::mle_simplex(resid_mgarch1, theta_mgarch, resid_var, theta_mgarch0);

	resid_mgarch = get_submatrix(resid_mgarch1, 1, T-1);

	return fmin;
}

//#if 0
/* Compute estimates for VAR-CC-MGARCH(1,1) model.*/
double cc_mgarch::ols_mle_simplex( 	SGMatrix<double> &resid_mgarch, /*T by 2 matrix of residuals for CC-MGARCH*/
									SGVector<double> &theta_var_mgarch, /*11 by 1 vector of parameter estimates*/
									const SGMatrix<double> &Y, /*T by 2 matrix of observations*/
									SGVector<double> theta_var_mgarch0 /*11 by 1 vector of initial parameters*/ ) {

	int T = Y.num_rows;
	ASSERT_(resid_mgarch.num_rows == T && theta_var_mgarch.vlen == theta_var_mgarch0.vlen);
	/* 1. Estimate the VAR part by the OLS */
	SGMatrix<double> Phi(2, 3);
	SGMatrix<double> resid_var = NL_Dgp::resid_VAR(Phi, Y, 1);
	theta_var_mgarch[0] = Phi(0,1);
	theta_var_mgarch[1] = Phi(0,2);
	theta_var_mgarch[2] = Phi(1,1);
	theta_var_mgarch[3] = Phi(1,2);

	/* 2. Fit 'resid_var' to the CC-MGARCH model */
	SGVector<double> theta_mgarch0 = get_subvector(theta_var_mgarch0, 4, 10);
	SGVector<double> theta_mgarch(7);
	double fmin = cc_mgarch::mle_simplex(resid_mgarch, theta_mgarch, resid_var, theta_mgarch0);
	for (int i = 0; i < 7; ++i)
		theta_var_mgarch[4+i] = theta_mgarch[i];

	return fmin;
}
//#endif




#if 0
Matrix cc_mgarch::covariance_proc (Matrix& A, Matrix& B, Matrix& xi, Matrix& S, Matrix& Q0)
{
       int N = 0;
       N = A.nRow();
       Matrix O(N,1), A2(N,N), B2(N,N), res (N,N);
       A2 = A*Tr(A);
       B2 = B*Tr(B);
       O = ones(N);
       res = S->*(O*Tr(O) - A2 - B2) + A2->*(xi*Tr(xi)) + B2->*Q0;
       return res;
}

double cc_mgarch::log_likelihood_var (Matrix& parameters, Matrix& data)//parameters is a N by 4 matrix
{
       int N, T;
       N = data.nCol();
       T = data.nRow();
       Matrix  kappa(N,1),  beta(N,1), zeta(N,1);
       bool key = true;
       for (int i = 1; i <= N; i++)
       {
           beta(i,1) = parameters(i,2);
           kappa(i,1) = parameters(i,3);
           zeta(i,1) = parameters(i,4);
           if ((beta(i,1) <= -1) || (beta(i,1) >= 1) || (kappa(i,1) < 0) || (zeta(i,1) < 0) || (kappa(i,1) + zeta(i,1) >=1))
           {
                           key = false;
                           break;
           }
       }
       if (key == true)
       {
               Matrix lambda(N,1), X(N,1), X1(N,1), Sigma(N,N);
               Sigma = covariance (data);
               for (int i = 1; i <= N; i++)
               {
                   lambda(i,1) = parameters(i,1);
               }
               Matrix mu(N,1), D(N,N), D0(N,N);
               D0.set(0.);
               double res = 0.;
               for (int t = 2; t <= T; t++)
               {
                   for (int i = 1; i <= N; i++)
                   {
                       X(i,1) = data(t-1,i);
                       X1(i,1) = data(t,i);
                   }
                   mu = mean_proc (lambda, beta, X);
                   D = variance_proc (kappa, zeta, Sigma, X, D0);
                   res -= 0.5*(N*log(2*M_PI)+ log(fabs(determot(D))) + scalar((Tr(X1-mu))*((D^(-1.))*(X1-mu))));
                   D0 = D;
               }
               //cout << "res..." << res << endl;
               return res;
        }
        else
        {
            return (GSL_NEGINF);
        }

}

double cc_mgarch::log_likelihood_cor (Matrix& parameters, Matrix& data, Matrix& coeff)
{
       int N, T;
       N = data.nCol();
       T = data.nRow();
       Matrix A(N,1), B(N,1);
       bool key = true;
       for (int i = 1; i <= N; i++)
       {
           A(i,1) = parameters(i,1);
           B(i,1) = parameters(i,2);
           if ((A(i,1) < 0) || (B(i,1) < 0) || (A(i,1)+B(i,1) >= 1))
           {
                       key = false;
                       break;
           }
       }
       if (key == true)
       {
               Matrix Sigma(N,N), lambda(N,1), beta(N,1), kappa(N,1), zeta(N,1);
               Sigma = covariance (data);
               for (int i = 1; i <= N; i++)
               {
                   lambda(i,1) = coeff(i,1);
                   beta(i,1) = coeff(i,2);
                   kappa(i,1) = coeff(i,3);
                   zeta(i,1) = coeff(i,4);
               }
               Matrix D0(N,N), X0(N,1), X1(N,1), mu(N,1), D(N,N), xi(N,1), S(N,N), xi1(N,T-1);
               Matrix mu0(N,1);
               mu0 = mean (data);
               D0.set(0.);
               S.set(0.);
               for (int t = 2; t <= T; t++)
               {
                   for (int i = 1; i <= N; i++)
                   {
                       X0(i,1) = data(t-1,i);
                       X1(i,1) = data(t,i);
                   }
                   mu = mean_proc (lambda, beta, X0);
                   D = variance_proc (kappa, zeta, Sigma, X0, D0);
                   xi = (D0^(-0.5))*(X0 - mu0);
                   for (int i = 1; i <= N; i++)
                   {
                       xi1(i,t-1) = xi(i,1);
                   }
                   S = S + ((double) 1/(T-1))*(xi*Tr(xi));
                   mu0 = mu;
                   D0 = D;
               }
               Matrix tmp(N,1), Q(N,N), Q0(N,N), R(N,N), Qd(N,N);
               Q0.set(0.);
               double res = 0.;
               for (int t = 2; t <= T; t++)
               {
                   for (int i = 1; i <= N; i++)
                   {
                       xi(i,1) = xi1(i, t-1);
                   }
                   Q = covariance_proc (A, B, xi, S, Q0);
                   for (int i = 1; i <= N; i++)
                   {
                       tmp(i,1) = Q(i,i);
                   }
                   Qd = diag(tmp);
                   R = (Qd^(-0.5))*(Q*(Qd^(-0.5)));
                   res -= 0.5*(log(fabs(determ(R))) + scalar(Tr(xi)*(inv(R)*xi)) - scalar(Tr(xi)*xi));
                   Q0 = Q;
               }
               return res;
        }
        else
        {
            return (GSL_NEGINF);
        }
}
#endif

















#endif
