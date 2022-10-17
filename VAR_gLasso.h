#ifndef VAR_GLASSO_H_
#define VAR_GLASSO_H_

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_blas.h>
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
#include <gsl/gsl_bspline.h>
#include <nmsimplex.h>
#include <asserts.h>

using namespace std;

class VAR_gLASSO {
	public:
		VAR_gLASSO () {   }; //default constructor
		~VAR_gLASSO () {   }; //default destructor
		//calculate the least-squares estimates as an initial estimate used for the Block Coordinate Descent Algorithm. INPUT:  INPUT: a T0 by 1 matrix of data on the response (Y),
		//a N*L*m by T0 matrix of all B-spline basis functions evaluated at all past data points on the X's (BS). OUTPUT: a N*L*m by 1 matrix of  least-squares estimates (beta).
		static void calcul_LS (Matrix &beta, const Matrix &Y, const Matrix &BS);

		//calculate the gradient vector of the least-squares criterion function for each additive component (j,ell). INPUT: a T0 by 1 matrix of centered data on the response (Y),
		//a N*L*m by T0 matrix of all centered B-spline basis functions evaluated at all past data points on the X's (BS), a N*L*m by 1 matrix of B-spline coefficients (beta),
		//the number of covariates (N), and a specific additive component j = 1,...,N; ell = 1,...L. OUTPUT; a m by 1 matrix of gradients.
		static void calcul_Gradients (Matrix &rho_jl, const Matrix &Y, const Matrix &BS, const Matrix &beta, const int N, const int j, const int ell);

		//calculate the sum of squared residuals (SSR) for the additive autoregression model. INPUT: a T0 by 1 matrix of centered data on the response (Y), a N*L*m by T0 matrix of all
		//centered B-spline basis functions, evaluated at all past data points on the X's (BS), a N*L*m by 1 matrix of B-spline coefficients (beta), the number of covariates (N), and
		//the maximum number of lags (L). OUTPUT: the SSR.
		static double calcul_SSR (const Matrix &Y, const Matrix &BS, const Matrix &beta, const int N, const int L_T);

		//calculate the group LASSO objective function at a specific block, say the (j,ell) block while keeping all the other blocks fixed.
		//INPUT: a m by 1 matrix of slope coefficients for the (j,ell)-th block (beta_jl), a T0 by 1 matrix of centered data on the response (Y), a N*L*m by T0 matrix of all centered B-spline
		//basis functions evaluated at all past data points on the X's (BS), a N*L*m by 1 matrix of B-spline coefficients (beta), the number of covariates (N), a tuning coefficient (lambda_T),
		//and a specific block j = 1,...,N; ell = 1,...L. OUTPUT: a function of specific block coordinates, say beta_jl.
		static double obj_Func (const Matrix &beta_jl, const Matrix &Y, const Matrix &BS, const Matrix &beta, const int N, const double lambda_T, const int j, const int ell);
		//rewrite VAR_gLASSO::obj_Func (defined above) in the GSL functional format, which can then be used as an argument to the GSL minimization routine
		//employing the Simplex algorithm of Nelder and Mead.
		static double obj_Func_gsl (const gsl_vector *beta_jl0, void *parms);

		//calculate the group LASSO estimates for the B-spline coefficients by using the least-squares estimates as initial estimates. This routine employs the Block Coordinate Descent Algorithm
		//(see p. 69 in Buhlmann & van de Geer (2011).
		//INPUT:  an N*L*m by 1 matrix of initial parameter vetor (beta_init), a T0 by 1 matrix of centered data on the response (Y), a N*L*m by T0 matrix of all centered B-spline basis
		//functions evaluated at all past data points on the X's (BS), the number of covariates (N), the number of time lags (L_T), a tuning parameter (lambda_T), a maximum number of
		//iterations (max_iter), a minimum tolerance level for numerical convergence (min_tol), and an output stream (glasso_out).
		//OUTPUT: a N*L*m by 1 matrix of the group LASSO estimates for the B-spline coefficients (beta_init).
		static void calcul_gLasso (Matrix &beta_init, const Matrix &Y, const Matrix &BS, const int N, const int L_T, const double lambda_T, const int max_iter,const double min_tol,
								   ofstream &glasso_out);

		//select the optimal tuning parameters for the group LASSO by employing the BIC and the EBIC. INPUT:  an N*L*m by 1 matrix of initial parameters (beta_init0), a T0 by 1 matrix of
		//centered data on the response (Y), a N*L*m by T0 matrix of all centered B-spline basis functions evaluated at all past data points on the X's (BS), a divergence rate for the tuning
		//parameter (lambda_glasso_rate), lower and upper bounds (lb and ub) of the scaling factor C0 in the tuning parameter (lambda = C0*lambda_glasso_rate), a number of grid points
		//(ngrid) in the interval [lb, ub], a number of the number of covariates (N), the number of time lags (L_T), a maximum number of iterations (max_iter), a minimum tolerance level
		//for numerical convergence (min_tol), output streams (ic_out and glasso_out).
		//OUTPUT: optimal tuning parameters using the BIC and EBIC; optimal values of the BIC and the EBIC (opt_BIC and opt_EBIC).
		static Matrix calcul_IC (double & opt_BIC, double & opt_EBIC, const Matrix &beta_init0, const Matrix &Y, const Matrix &BS, const double lambda_glasso_rate,
										 const double lb, const double ub, const int ngrid,  const int N, const int L_T, const int max_iter, const double min_tol, ofstream &ic_out,
										 ofstream &glasso_out);

		//calculate the adaptive group LASSO estimates from datasets for nonparametric additive ARDL models with each component being approximated by B-splines. INPUT: a T by 1 matrix of data
		//on the response Y (Y0), a T by N matrix of data on covariates (X0), a maximum number of autoregressive lags (L_T), an order of B-spline polynomials (poly_degree, set it to 4 to yield cubic B-splines),
		//a number of break points in [0,1] (nbreak), tuning parameters for group LASSO and adaptive group LASSO (lambda1 and lambda2), a maximum number of iterations (max_iter) for the Block
		//Coordinate Descent Algorithm, a minimum level of tolerance to evaluate numerical convergence of this algorithm (min_tol), a output file stream (glasso_out), and an initial estimates (beta_init).
		//OUTPUT: a N*L*m by 1 matrix of the adaptive group LASSO estimates for the B-spline coefficients (beta_init).
		static void calcul_adap_gLasso (Matrix &beta_init, const Matrix &Y0, const Matrix &X0, const int L_T, const int poly_degree, const int nbreak, const double lambda1,
										const double lambda2, const int max_iter, const double min_tol, ofstream &glasso_out);

};

//calculate the least-squares estimates as an initial estimate used for the Block Coordinate Descent Algorithm. INPUT:  INPUT: a T0 by 1 matrix of data on the response (Y),
//a N*L*m by T0 matrix of all B-spline basis functions evaluated at all past data points on the X's (BS). OUTPUT: a N*L*m by 1 matrix of  least-squares estimates (beta).
void VAR_gLASSO::calcul_LS (Matrix &beta, const Matrix &Y, const Matrix &BS) {
	auto T0 = 0;
	T0 = Y.nRow();
	auto M = BS.nRow();
	ASSERT ((T0 == BS.nCol()) && (M == beta.nRow()));
	Matrix denom(M,M),  denom1(M,M), num(M,1);
	denom = BS * Tr(BS);
	num = BS * Y;
	inverse (denom1, denom);
	beta = denom1 * num;
}

//calculate the gradient vector of the least-squares criterion function for each additive component (j,ell). INPUT: a T0 by 1 matrix of centered data on the response (Y),
//a N*L*m by T0 matrix of all centered B-spline basis functions evaluated at all past data points on the X's (BS), a N*L*m by 1 matrix of B-spline coefficients (beta),
//the number of covariates (N), and a specific additive component j = 1,...,N; ell = 1,...L. OUTPUT; a m by 1 matrix of gradients.
void VAR_gLASSO::calcul_Gradients (Matrix &rho_jl, const Matrix &Y, const Matrix &BS, const Matrix &beta, const int N, const int j, const int ell) {
	auto T0 = 0, L_T = 0, m_T = 0;
	T0 = BS.nCol();
	ASSERT (T0 == Y.nRow() && BS.nRow() == beta.nRow());
	m_T = rho_jl.nRow();
	L_T = BS.nRow() / (N*m_T);
	Matrix sum0(T0,1);
	sum0.set(0.);
	for (auto t = 1; t <= T0; ++t) {
		for (auto j0 = 1; j0 <= N; ++j0) {
			for (auto ell0 = 1; ell0 <= L_T; ++ell0) {
				for (auto k = 1; k <= m_T; ++k) {
					sum0(t) += beta((j0-1)*m_T*L_T+(ell0-1)*m_T+k) * BS((j0-1)*m_T*L_T+(ell0-1)*m_T+k, t);
				}
			}
		}
	}
	rho_jl.set(0.);
	for (auto k = 1; k <= m_T; ++k) {
		for (auto t = 1; t <= T0; ++t) {
			rho_jl(k) += 2*BS((j-1)*m_T*L_T+(ell-1)*m_T+k, t) * (Y(t) - sum0(t));
		}
	}
}

//calculate the sum of squared residuals (SSR) for the additive autoregression model. INPUT: a T0 by 1 matrix of centered data on the response (Y), a N*L*m by T0 matrix of all centered
//B-spline basis functions, evaluated at all past data points on the X's (BS), a N*L*m by 1 matrix of B-spline coefficients (beta), the number of covariates (N), and the maximum number
//of lags (L). OUTPUT: the SSR.
double VAR_gLASSO::calcul_SSR (const Matrix &Y, const Matrix &BS, const Matrix &beta, const int N, const int L_T) {
	auto T0 = 0, m_T = 0;
	T0 = Y.nRow();
	m_T = BS.nRow() / (N*L_T);
	ASSERT(T0 == BS.nCol());
	double SSR  = 0., sum0 = 0.;
	for (auto t = 1; t <= T0; ++t) {
		sum0 = 0.;
		for (auto j0 = 1; j0 <= N; ++j0) {
			for (auto ell0 = 1; ell0 <= L_T; ++ell0) {
				for (auto k = 1; k <= m_T; ++k) {
					sum0 += beta((j0-1)*m_T*L_T+(ell0-1)*m_T+k) * BS((j0-1)*m_T*L_T+(ell0-1)*m_T+k, t);
				}
			}
		}
		SSR += pow(Y(t) - sum0, 2.);
	}
	return SSR;
}

//calculate the group LASSO objective function at a specific block, say the (j,ell) block while keeping all the other blocks fixed.
//INPUT: a m by 1 matrix of slope coefficients for the (j,ell)-th block (beta_jl), a T0 by 1 matrix of centered data on the response (Y), a N*L*m by T0 matrix of all centered B-spline
//basis functions evaluated at all past data points on the X's (BS), a N*L*m by 1 matrix of B-spline coefficients (beta), the number of covariates (N), a tuning coefficient (lambda_T),
//and a specific block j = 1,...,N; ell = 1,...L. OUTPUT: a function of specific block coordinates, say beta_jl.
double VAR_gLASSO::obj_Func (const Matrix &beta_jl, const Matrix &Y, const Matrix &BS, const Matrix &beta_, const int N, const double lambda_T, const int j, const int ell) {
	auto T0 = 0, m_T = 0, L_T = 0;
	T0 = Y.nRow();
	m_T = beta_jl.nRow();
	L_T = BS.nRow() / (N*m_T);
	ASSERT(T0 == BS.nCol());
	Matrix beta = beta_; //copy matrices beta_ to beta
	for (auto k = 1; k <= m_T; ++k) {
		beta((j-1)*m_T*L_T+(ell-1)*m_T+k) = beta_jl(k); //copy beta_jl to beta at the (j,ell)-th position
	}
	auto res = 0., enorm_sq = 0.;
	res = VAR_gLASSO::calcul_SSR (Y, BS, beta, N, L_T); //calculate the SSR
	for (auto j0 = 1; j0 <= N; ++j0) {
		for (auto ell0 = 1; ell0 <= L_T; ++ell0) {
			enorm_sq = 0.; //re-set the initial value to zero
			for (auto k = 1; k <= m_T; ++k) {
				enorm_sq += pow(beta((j0-1)*m_T*L_T+(ell0-1)*m_T+k), 2.);
			}
			res += lambda_T * std::sqrt(enorm_sq);
		}
	}
	return res;
}

//rewrite VAR_gLASSO::obj_Func (defined above) in the GSL functional format, which can then be used as an argument to the GSL minimization routine
//employing the Simplex algorithm of Nelder and Mead.
double VAR_gLASSO::obj_Func_gsl (const gsl_vector *beta_jl0, void *parms) {
	double *p = (double *) parms;
	int m_T = beta_jl0->size, N = p[0], L_T = p[1], T0 = p[2], j = p[4], ell = p[5], BS_nr = N*L_T*m_T;
	auto lambda_T = p[3]; //tuning parameter
	Matrix Y(T0,1), BS(BS_nr,T0), beta(BS_nr,1), beta_jl(m_T,1);

	for (auto t = 1; t <= T0; ++t) {
          Y(t) = p[5+t]; //reconstruct the data matrix on the response
	}

	for (auto j0 = 1; j0 <= N; ++j0) {
		for (auto ell0 = 1; ell0 <= L_T; ++ell0) {
			for (auto k = 1; k <= m_T; ++k ) {
				for (auto t = 1; t <= T0; ++t) {
					BS((j0-1)*m_T*L_T+(ell0-1)*m_T+k, t) = p[5+T0+T0*((j0-1)*m_T*L_T+(ell0-1)*m_T+k-1)+t]; //reconstruct the N*L*m by T0 matrix of all B-spline basis functions evaluated at all past data points on the X's (BS)
				}
				beta((j0-1)*m_T*L_T+(ell0-1)*m_T+k) = p[5+T0+T0*N*L_T*m_T+(j0-1)*m_T*L_T+(ell0-1)*m_T+k]; //reconstruct the N*L*m by 1 matrix of B-spline coefficients
			}
		}
	}

	for (auto k = 1; k <= m_T; ++k) {
		beta_jl(k) = gsl_vector_get(beta_jl0, k-1); //copy beta_jl0 to beta_jl
	}
     return VAR_gLASSO::obj_Func (beta_jl, Y, BS, beta, N, lambda_T, j, ell);
}

//calculate the group LASSO estimates for the B-spline coefficients by using the least-squares estimates as initial estimates. This routine employs the Block Coordinate Descent Algorithm
//(see p. 69 in Buhlmann & van de Geer (2011).
//INPUT: an N*L*m by 1 matrix of initial parameters (beta_init), a T0 by 1 matrix of centered data on the response (Y), a N*L*m by T0 matrix of all centered B-spline basis functions
//evaluated at all past data points on the X's (BS), the number of covariates (N), the number of time lags (L_T), a tuning parameter (lambda_T), a maximum number of iterations (max_iter),
//a minimum tolerance level for numerical convergence (min_tol), and an output stream (glasso_out). OUTPUT: a N*L*m by 1 matrix of the group LASSO estimates for the B-spline
//coefficients (beta_init).
void VAR_gLASSO::calcul_gLasso (Matrix &beta_init, const Matrix &Y, const Matrix &BS, const int N, const int L_T, const double lambda_T, const int max_iter,
												   const double min_tol, ofstream &glasso_out) {
	ASSERT (Y.nRow() == BS.nCol() && beta_init.nRow() == BS.nRow());
	auto iter = 1, nr_beta = 0, m_T = 0, T0 = 0;
	nr_beta = beta_init.nRow();
	m_T = nr_beta / (N*L_T);
	T0 = Y.nRow();
	Matrix beta0(nr_beta,1), beta1(nr_beta,1), rho_jl(m_T,1), beta_jl1(m_T,1);
	gsl_vector *beta_jl0 = gsl_vector_alloc(m_T); //define a vector of starting points used in the the Simplex minimization algorithm of Nelder and Mead

	//VAR_gLASSO::calcul_LS (beta_init, Y, BS); //use the LS estimates as an initial estimate
	glasso_out << "The initial estimates for the B-spline coefficients are: " << endl;
	for (auto j = 1; j <= N; ++j) {
		for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= m_T; ++k) {
				glasso_out << "j = " << j << ", ell = " << ell << ", k = " << k << ": " << beta_init((j-1)*m_T*L_T+(ell-1)*m_T+k) << endl;
			}
		}
	}

	std::vector<int> set_j;
	std::vector<int> set_ell;
	for (auto j = 1; j <= N; ++j) {
		for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= m_T; ++k) {
				beta_jl1(k) = beta_init((j-1)*m_T*L_T+(ell-1)*m_T+k);
			}
			if (ENorm(beta_jl1) != 0.) { //include only non-zero groups
				set_j.push_back(j); //copy group indices to vectors of type integer
				set_ell.push_back(ell);
			}
		}
	}
	std::sort(set_j.begin(), set_j.end()); // sort set_j
	std::sort(set_ell.begin(), set_ell.end()); // sort set_ell
	auto last_j = std::unique(set_j.begin(), set_j.end()); //override the duplicate elements of set_j
	auto last_ell = std::unique(set_ell.begin(), set_ell.end()); //override the duplicate elements of set_ell
	set_j.erase(last_j, set_j.end()); //remove extra elements at the end from the range of set_j
	set_ell.erase(last_ell, set_ell.end()); //remove extra elements at the end from the range of set_ell
	//cout << "size of set_j = " << set_j.size() << " and size of set_ell = " << set_ell.size() << endl;

	//main loop
	do {
		//cycling through all the non-zero block coordinates {1, ... , m_T}
		for (auto j : set_j) {
			for (auto ell : set_ell) {
				//cout << "(j, ell) = " << "(" << j << ", " << ell << ")" << endl;
				beta0 = beta_init; //re-set beta0 for each (j,ell)-th group
				for (auto k = 1; k <= m_T; ++k) {
					//copy the (j,ell)-th block to beta_jl0, which is then used as the starting values to call the Simplex minimization algorithm right below
					gsl_vector_set (beta_jl0, k-1, beta0((j-1)*m_T*L_T+(ell-1)*m_T+k));
					beta0((j-1)*m_T*L_T+(ell-1)*m_T+k) = 0.; //assign zeros to the (j,ell)-th block in the vector beta0
				}

				VAR_gLASSO::calcul_Gradients (rho_jl, Y, BS, beta0, N, j, ell); //calculate the gradients of the SSE for the (j,ell)-th block at beta0

				//cout << "ENorm(rho_jl) = " << ENorm(rho_jl)  << endl;
				if (ENorm(rho_jl) <= lambda_T) {
					for (auto k = 1; k <= m_T; ++k) {
						beta1((j-1)*m_T*L_T+(ell-1)*m_T+k) = 0.;
					}
				}
				else {
					double *par = new double[6+T0+(T0+1)*N*L_T*m_T]; //define an array
					//assign the parameters for the GSL objective function to be minimized
					par[0] = N;
					par[1] = L_T;
					par[2] = T0;
					par[3] = lambda_T;
					par[4] = j;
					par[5] = ell;
					for (auto t = 1; t <= T0; ++t) {
						par[5+t] = Y(t);
					}
					for (auto j0 = 1; j0 <= N; ++j0) {
						for (auto ell0 = 1; ell0 <= L_T; ++ell0) {
							for (auto k = 1; k <= m_T; ++k ) {
								for (auto t = 1; t <= T0; ++t) {
									par[5+T0+T0*((j0-1)*m_T*L_T+(ell0-1)*m_T+k-1)+t] = BS((j0-1)*m_T*L_T+(ell0-1)*m_T+k, t);
								}
								par[5+T0+T0*N*L_T*m_T+(j0-1)*m_T*L_T+(ell0-1)*m_T+k] = beta0((j0-1)*m_T*L_T+(ell0-1)*m_T+k);
							}
						}
					}
					//cerr << "Starting the NM Simplex algorithm. . ." << endl;
					minimization<VAR_gLASSO::obj_Func_gsl> (beta_jl1, beta_jl0, par); //call the Simplex minimization algorithm of Nelder and Mead initiazed at beta_jl0
					for (auto k = 1; k <= m_T; ++k) {
						beta1((j-1)*m_T*L_T+(ell-1)*m_T+k) = beta_jl1(k); //copy the minimum points from beta_jl1 to beta1
					}
					delete [] par; //delete par
				}
			}
		}

		//cout << "stopcdn(beta1, beta_init) = " << stopcdn(beta1, beta_init) << endl;
		if (stopcdn(beta1, beta_init) <= min_tol) {
			beta_init = beta1;
			break;
		}
		else {
			beta_init = beta1;
			iter += 1;
		}
		//cout << "VAR_gLASSO::calcul_gLasso: Implementing the BCDA -- loop # " << iter  << ".  " << endl;
	} while (iter <= max_iter);

	if (iter <= max_iter) {
		glasso_out << "VAR_gLASSO::calcul_gLasso: The block coordinate descent algorithm converges at the " << iter << "-th iteration. . ."<< endl;
		cout << "VAR_gLASSO::calcul_gLasso: The block coordinate descent algorithm converges at the " << iter << "-th iteration. . ."<< endl;
	}
	else {
		cerr << "VAR_gLASSO::calcul_gLasso: The block coordinate descent algorithm does not converge when the maximum number of iterations is set to " << max_iter << ". . ." << endl;
		glasso_out << "VAR_gLASSO::calcul_gLasso: The block coordinate descent algorithm does not converge when the maximum number of iterations is set to " << max_iter << ". . ." << endl;
	}
	glasso_out << "The group LASSO estimates for the B-spline coefficients are: " << endl;
	for (auto j = 1; j <= N; ++j) {
		for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= m_T; ++k) {
				glasso_out << "j = " << j << ", ell = " << ell << ", k = " << k << ": " << beta_init((j-1)*m_T*L_T+(ell-1)*m_T+k) << endl;
			}
		}
	}
	gsl_vector_free (beta_jl0); //free memory
}

//calculate the adaptive group LASSO estimates from datasets for nonparametric additive ARDL models with each component being approximated by B-splines. INPUT: a T by 1 matrix of data
//on the response Y (Y0), a T by N matrix of data on covariates (X0), a maximum number of autoregressive lags (L_T), an order of B-spline polynomials (poly_degree, set it to 4 to yield cubic B-splines),
//a number of break points in [0,1] (nbreak), tuning parameters for group LASSO and adaptive group LASSO (lambda1 and lambda2), a maximum number of iterations (max_iter) for the Block
//Coordinate Descent Algorithm, a minimum level of tolerance to evaluate numerical convergence of this algorithm (min_tol), a output file stream (glasso_out), and an initial estimates (beta_init).
//OUTPUT: a N*L*m by 1 matrix of the adaptive group LASSO estimates for the B-spline coefficients (beta_init).
void VAR_gLASSO::calcul_adap_gLasso (Matrix &beta_init, const Matrix &Y0, const Matrix &X0, const int L_T, const int poly_degree, const int nbreak, const double lambda1,
									 const double lambda2, const int max_iter, const double min_tol, ofstream &glasso_out) {
	auto T = Y0.nRow(), N = X0.nCol(), beta_nR = beta_init.nRow();
	ASSERT (T == X0.nRow());
	int ncoeffs = nbreak + poly_degree - 2;
	int T0 = T- L_T;

	Matrix Y(T0,1), X(T,N);
	do_Norm(X, X0); //normalize data to the 0-1 range

	for (auto t = L_T+1; t <= T; ++t ) {
		Y(t-L_T) = Y0(t);
     }
     double mean_Y = mean_u(Y);
     for (auto t = 1; t <= T0; ++t) {
		Y(t) = Y(t) - mean_Y; //re-center data on the response
     }

     gsl_bspline_workspace *bw; //define workspace for B-splines
     bw = gsl_bspline_alloc(poly_degree, nbreak);
     //use uniform breakpoints on [0, 1]
     gsl_bspline_knots_uniform(0.0, 1.0, bw);
    //calculate the knots
	//for (auto k = 1; k <= ncoeffs+poly_degree; ++k )
        //cout << "knot = " << k <<  ": " << gsl_vector_get(bw->knots, k-1) << endl;
     Matrix BS(beta_nR,T0), ave_BS(beta_nR,1), beta_init_jl(ncoeffs,1);
     gsl_vector *B = gsl_vector_alloc (ncoeffs); //define a GSL array pointer
     for (auto  j= 1; j <= N; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = L_T+1; t <= T; ++t) {
                         gsl_bspline_eval(X(t-ell,j), B, bw); //evaluate all B-spline basis functions at X(t-ell,j)
                         for (auto k = 1; k <= ncoeffs; ++k) {
						BS((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t-L_T) = poly_degree * gsl_vector_get(B, k-1) / (nbreak-1); //normalize all the B-splines basis functions
						//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
                         }
               }
          }
     }
     gsl_bspline_free (bw); //free the B-spline worksapce
     gsl_vector_free (B); //free memory

     ave_BS.set(0.);
     for (auto  j= 1; j <= N; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= ncoeffs; ++k) {
				for (auto t = 1; t <= T0; ++t) {
					ave_BS((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) += BS((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) / T0; //calculate the temporal averages for all B-spline basis functions
				}
			}
          }
     }
     for (auto  j= 1; j <= N; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
               for (auto t = 1; t <= T0; ++t) {
				for (auto k = 1; k <= ncoeffs; ++k) {
					//calculate the centered B-spline basis functions
					BS((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) = BS((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) - ave_BS((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k);
					//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
				}
               }
          }
     }

	glasso_out << "Do the group LASSO: " << endl;
	cout << "Doing the group LASSO. . ." << endl;
	VAR_gLASSO::calcul_gLasso (beta_init, Y, BS, N, L_T, lambda1, max_iter, min_tol, glasso_out); //do the group LASSO
	cout << "The group LASSO estimates for the B-spline coefficients are: " << endl;
	glasso_out << "The group LASSO estimates for the B-spline coefficients are: " << endl;
	for (auto j = 1; j <= N; ++j) {
		for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= ncoeffs; ++k) {
				cout << "j = " << j << ", ell = " << ell << ", k = " << k << ": " << beta_init((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) << endl;
				glasso_out << "j = " << j << ", ell = " << ell << ", k = " << k << ": " << beta_init((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) << endl;
			}
		}
	}

	Matrix norm_beta(N,L_T);
	for (auto  j= 1; j <= N; ++j) {
          for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= ncoeffs; ++k)
				beta_init_jl(k) = beta_init((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k);
			norm_beta(j,ell) = ENorm(beta_init_jl);
			for (auto t = 1; t <= T0; ++t) {
				for (auto k = 1; k <= ncoeffs; ++k) {
					//multiply B-spline basis functions with the Euclidean norms of the group LASSO estimates in each (j,ell) block
					BS((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) = BS((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k, t) * norm_beta(j,ell);
					//cout << "j = " << j << ", ell = " << ell << ", k = " << k << ", t = " << t << ": " << BS((j-1)*ncoeffs*L+(ell-1)*ncoeffs+k, t-L) << endl;
				}
			}
          }
     }

	glasso_out << "Do the adaptive group LASSO: " << endl;
	cout << "Doing the adaptive group LASSO. . ." << endl;
	VAR_gLASSO::calcul_gLasso (beta_init, Y, BS, N, L_T, lambda2, max_iter, min_tol, glasso_out); //do the adaptive group LASSO
	cout << "The adaptive group LASSO estimates for the B-spline coefficients are: " << endl;
	glasso_out << "The adaptive group LASSO estimates for the B-spline coefficients are: " << endl;
	for (auto j = 1; j <= N; ++j) {
		for (auto ell = 1; ell <= L_T; ++ell) {
			for (auto k = 1; k <= ncoeffs; ++k) {
				//multiply the obtained B-spline coefficients with the Euclidean norms of the group LASSO estimates in each (j,ell) block to yield
				//the estimated slope coefficients of the original B-spline basis functions
				beta_init((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) = beta_init((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) * norm_beta(j,ell);
				cout << "j = " << j << ", ell = " << ell << ", k = " << k << ": " << beta_init((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) << endl;
				glasso_out << "j = " << j << ", ell = " << ell << ", k = " << k << ": " << beta_init((j-1)*ncoeffs*L_T+(ell-1)*ncoeffs+k) << endl;
			}
		}
	}
}


//select the optimal tuning parameters for the group LASSO by employing the BIC and the EBIC as in Huang, Horowitz & Fei (2009). INPUT:  an N*L*m by 1 matrix of initial parameters (beta_init0),
//a T0 by 1 matrix of centered data on the response (Y), a N*L*m by T0 matrix of all centered B-spline basis functions evaluated at all past data points on the X's (BS), a divergence rate
//for the tuning parameter (lambda_glasso_rate), lower and upper bounds (lb and ub) of the scaling factor C0 in the tuning parameter (lambda = C0*lambda_glasso_rate), a number of grid
//points (ngrid) in the interval [lb, ub], a number of the number of covariates (N), the number of time lags (L_T), a maximum number of iterations (max_iter), a minimum tolerance level
//for numerical convergence (min_tol), and output streams (ic_out and glasso_out).
//OUTPUT: optimal tuning parameters using the BIC and EBIC; optimal values of the BIC and the EBIC (opt_BIC and opt_EBIC).
Matrix VAR_gLASSO::calcul_IC (double & opt_BIC, double & opt_EBIC, const Matrix &beta_init0, const Matrix &Y, const Matrix &BS, const double lambda_glasso_rate,
											    const double lb, const double ub, const int ngrid,  const int N, const int L_T, const int max_iter, const double min_tol, ofstream &ic_out,
											    ofstream &glasso_out) {
	auto i = 0, T0 = Y.nRow(), beta_nR = beta_init0.nRow(), df = 0;
	ASSERT ((lb < ub) && (Y.nRow() == BS.nCol()) && (beta_init0.nRow() == BS.nRow()));
	Matrix beta_init(beta_nR,1), BIC(ngrid,1), EBIC(ngrid,1), lambda(ngrid,1), min0(2,1), res(2,1);
	double ssr = 0.;
	ic_out << "Calculating BIC and EBIC. . ." << endl;
	cout << "Calculating BIC and EBIC. . ." << endl;
	ic_out << "lambda" << " , " << "BIC" << " , " << "EBIC" << endl;
	cout << "lambda" << " , " << "BIC" << " , " << "EBIC" << endl;
	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(i) firstprivate(beta_init,df,ssr)
	for (i = 0; i < ngrid; ++i) {
		lambda(i+1) = (lb + i*(ub-lb)/(ngrid-1)) * lambda_glasso_rate;
		//cout << "lambda(i+1) = " << lambda(i+1) << endl;
		beta_init = beta_init0; //copy beta_init0 to beta_init to initialize the group LASSO
		VAR_gLASSO::calcul_gLasso (beta_init, Y, BS, N, L_T, lambda(i+1), max_iter, min_tol, glasso_out); //call the group LASSO routine with this lambda
		df = nonzero(beta_init); //the number of degrees of freedom for this lambda
		if (df == 0) {
			ic_out << "All the B-spline coefficients are shrunk to zeros when the penalty parameter (lambda_T) = " << lambda(i+1) << endl;
			cout << "All the B-spline coefficients are shrunk to zeros when the penalty parameter (lambda_T) = " << lambda(i+1) << endl;
			BIC(i+1) = 1000;
			EBIC(i+1) = 1000;
		}
		else {
			ssr = VAR_gLASSO::calcul_SSR (Y, BS, beta_init, N, L_T); //calculate SSR for this lambda
			BIC(i+1) = std::log(ssr) + df*std::log(T0)/T0;
			EBIC(i+1) = std::log(ssr) + df*std::log(T0)/T0 + 0.5*df*std::log(N*L_T)/T0;
			ic_out << lambda(i+1) << " , " << BIC(i+1) << " , " << EBIC(i+1) << endl;
			cout << lambda(i+1) << " , " << BIC(i+1) << " , " << EBIC(i+1) << endl;
		}
	}
	min0 = minn(BIC); //find the minimum BIC
	opt_BIC = min0(2);
	int opt_BIC_ind = ((int) min0(1));
	res(1) = lambda(opt_BIC_ind);
	ic_out << "Optimal value for the tuning parameter using BIC is " << res(1) << " with the optimal BIC = " << opt_BIC << endl;
	cout << "Optimal value for the tuning parameter using BIC is " << res(1) << " with the optimal BIC = " << opt_BIC << endl;

	min0 = minn(EBIC); //find the minimum EBIC
	opt_EBIC = min0(2);
	int opt_EBIC_ind = ((int) min0(1));
	res(2) = lambda(opt_EBIC_ind);
	ic_out << "Optimal value for the tuning parameter using EBIC is " << res(2) << " with the optimal BIC = " << opt_EBIC << endl;
	cout << "Optimal value for the tuning parameter using EBIC is " << res(2) << " with the optimal EBIC = " << opt_EBIC << endl;
	return res;
}
















































#endif
