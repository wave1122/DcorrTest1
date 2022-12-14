#ifndef NONGAUSSIAN_REG_H_
#define NONGAUSSIAN_REG_H_

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <nl_dgp.h>

#define CHUNK 1

using namespace std;

class NGReg {
	public:
		NGReg () {   };//default constructor
		~NGReg () {   };//default destructor
		//calculate conditional expectations of |x_t - x_s|^\alpha and |y_t - y_s|^\alpha using the lagged values of x_t and y_t respectively.
        //INPUT: a 2x1 vector of the first lags of x_t and y_t (lag_t1), a 2x1 vector of the second lags of x_t and y_t (lag_t2), a 2x1 vector of X_s and Y_s (xy_s),
        //6x1 vectors of AR coefficients (alpha_X and alpha_Y), Nx1 vectors of random errors (epsilon and eta) generated by gen_RAN (epsilon, eta, delta, 0., 0, rseed)
        //using a random seed (rseed), a delta in (-1,1), and an exponent for the Euclidean distance (expn in (1,2)).
        //OUTPUT: conditional expectations E_{X_t|X_{t-1},X_{t-2}}[|X_t - X_s|^expn] and E_{Y_t|Y_{t-1},Y_{t-2}}[|Y_t - Y_s|^expn] (res_Xt and res_Yt).
        template <void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
        static void reg (double &res_Xt, double &res_Yt, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &xy_s, const Matrix &alpha_X,
		                 const Matrix &alpha_Y, const Matrix &epsilon, const Matrix &eta, const double expn);
		//calculate conditional expectations of |x_t - x_s|^\alpha and ||y_t - y_s||^\alpha using the lagged values of x_t and y_t respectively.
        //INPUT: a 3x1 vector of the first lags of x_t and y_t (lag_t1), a 3x1 vector of the second lags of x_t and y_t (lag_t2), a 3x1 vector of X_s and Y_s (xy_s),
        //a 6x1 vectors of AR coefficients (alpha_X) and a 6x2 matrix of AR coefficients (alpha_Y), Nx1 vectors of random errors (epsilon, eta1 and eta2) generated by
        //gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed) with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, and an exponent for the Euclidean
        //distance (expn in (1,2)). OUTPUT: conditional expectations E_{X_t|X_{t-1},X_{t-2}}[|X_t - X_s|^expn] and E_{Y_t|Y_{t-1},Y_{t-2}}[||Y_t - Y_s||^expn] (res_Xt and res_Yt).
        template <void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
        static void reg (double &res_Xt, double &res_Yt, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &xy_s, const Matrix &alpha_X, const Matrix &alpha_Y,
                         const Matrix &epsilon, const Matrix &eta1, const Matrix &eta2, const double expn);
		//calculate the conditional expectation of  ||x_t - x_s||^\alpha (conditioning on the lags of x_t) for t = 1, ..., T0 and s = 1, ..., T0 by using the sample averaging.
		//INPUT: a N*L*m by N matrix of B-spline coefficients (beta) for all individual ARDL B-spline regression models, a T0 by N matrix of data (X),
		//a N*L*m by T0 matrix of all B-spline basis functions evaluated at all past data points on the X's (BS), and an exponent of the distance correlation (expn).
		//OUTPUT: a T0 by T0 matrix (reg0).
		static void reg (Matrix &reg0, const Matrix &X, const Matrix &BS, const Matrix &beta, const double expn);
		//calculate the conditional expectation of  ||x_t - x_s||^\alpha (conditioning on the lags of x_t) for t = 1, ..., T0 and s = 1, ..., T0, where x_t follows a multivariate DCC model,
		// by using the sample averaging. INPUT: a T by N matrix of data (X), a T by N matrix of conditional means (cmean), a T by N(N+1)/2 matrix of the square roots of
		//conditional variances/covariances (csigma), a T by N matrix of MGARCH residuals, a distance-correlation exponent (expn).
		//OUTPUT: a T by T matrix (reg0)
		static void reg (Matrix &reg0, const Matrix &X, const Matrix &cmean, const Matrix &csigma, const Matrix &resid, const double expn);
        //calculate conditional expectations of |x_t - x_s|^\alpha and |y_t - y_s|^\alpha using the lagged values of (x_t, x_s) and (y_t, y_s) respectively.
        //INPUT: a 2x1 vector of the first lags of x_t and y_t (lag_t1), a 2x1 vector of the second lags of x_t and y_t (lag_t2), a 2x1 vector of the first lags
        //of x_s and y_s (lag_s1), a 2x1 vector of the second lags of x_s and y_s (lag_s2), a 6x1 vectors of AR coefficients (alpha_X and alpha_Y), Nx1 vectors of
        //random errors (epsilon_t, epsilon_s, eta_t and eta_s) generated by gen_RAN (epsilon, eta, delta, 0., 0, rseed) with random seeds (rseed), a delta in (-1,1),
        //and an exponent for the Euclidean distance (expn in (1,2)).
        //OUTPUT: conditional expectations E_{X_t|X_{t-1},X_{t-2}}E_{X_s|X_{s-1},X_{s-2}}[|X_t - X_s|^expn] and
        //E_{Y_t|Y_{t-1},Y_{t-2}}E_{Y_s|Y_{s-1},Y_{s-2}}[|Y_t - Y_s|^expn] (res_X and res_Y).
        template <void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
        static void breg (double &res_X, double &res_Y, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &lag_s1, const Matrix &lag_s2,
                          const Matrix &alpha_X, const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta_t,
				          const Matrix &eta_s, const double expn);
		//calculate conditional expectations of |x_t - x_s|^\alpha and ||y_t - y_s||^\alpha using the lagged values of (x_t, x_s) and (y_t, y_s) respectively.
        //INPUT: a 3x1 vector of the first lags of x_t and y_t (lag_t1), a 3x1 vector of the second lags of x_t and y_t (lag_t2), a 3x1 vector of the first lags
        //of x_s and y_s (lag_s1), a 3x1 vector of the second lags of x_s and y_s (lag_s2), a 6x1 vectors of AR coefficients (alpha_X) and a 6x2 matrix AR
        //coefficients (alpha_Y), Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by
        //gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed) with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, and an exponent for the Euclidean
        //distance (expn in (1,2)). OUTPUT: conditional expectations E_{X_t|X_{t-1},X_{t-2}}E_{X_s|X_{s-1},X_{s-2}}[|X_t - X_s|^expn] and
        //E_{Y_t|Y_{t-1},Y_{t-2}}E_{Y_s|Y_{s-1},Y_{s-2}}[||Y_t - Y_s||^expn] (res_X and res_Y).
        template <void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
        static void breg (double &res_X, double &res_Y, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &lag_s1, const Matrix &lag_s2,
		                  const Matrix &alpha_X, const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t,
						  const Matrix &eta1_s, const Matrix &eta2_t, const Matrix &eta2_s, const double expn);
		//calculate the conditional expectation of  ||x_t - x_s||^\alpha (conditioning on the lags of  both x_t and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by using the sample averaging.
		//calculate the conditional expectation of  ||x_t - x_s||^\alpha (conditioning on the lags of x_t) for t = 1, ..., T0 and s = 1, ..., T0 by using the sample averaging.
		//INPUT: a N*L*m by N matrix of B-spline coefficients (beta) for all individual ARDL B-spline regression models, a T0 by N matrix of data (X),
		//a N*L*m by T0 matrix of all B-spline basis functions evaluated at all past data points on the X's (BS), and an exponent of the distance correlation (expn).
		//OUTPUT: a T0 by T0 matrix (breg0).
		static void breg (Matrix &breg0, const Matrix &X, const Matrix &BS, const Matrix &beta, const double expn);
		//calculate the conditional expectation of  ||x_t - x_s||^\alpha (conditioning on the lags of  both x_t and x_s) for t = 1, ..., T0 and s = 1, ..., T0, where x_t follows a multivariate
		//GARCH process, by using the sample averaging. INPUT: a T by N matrix of data (X), a T by N matrix of conditional means (cmean), a T by N(N+1)/2 matrix of the square roots of
		//conditional variances/covariances (csigma), a T by N matrix of MGARCH residuals, a distance-correlation exponent (expn).
		//OUTPUT: a T0 by T0 matrix (breg0).
		static void breg (Matrix &breg0, const Matrix &X, const Matrix &cmean, const Matrix &csigma, const Matrix &resid, const double expn);
		//calculate re-centered Euclidean distances, U_{t,s}, with X_t and Y_t are univariate. INPUT: a 2x1 vector of X_t and Y_t (xy_t),
        //a 2x1 vector of X_s and Y_s (xy_s), a 2x1 vector of the first lags of X_t and Y_t (lag_t1), a 2x1 vector of the second lags of X_t and Y_t (lag_t2),
        //a 2x1 vector of the first lags of X_s and Y_s (lag_s1), a 2x1 vector of the second lags of X_s and Y_s (lag_s2), a 6x1 vectors of AR coefficients
        //(alpha_X and alpha_Y), vectors of random errors (epsilon_t, epsilon_s, eta_t and eta_s) generated by gen_RAN (epsilon, eta, delta, 0., 0, rseed)
        //with random seeds (rseed), a delta in (-1,1), and an exponent for the Euclidean distance (expn in (1,2)).
        //OUTPUT: U_{t,s}^{(x)} and U_{t,s}^{(y)}.
        template <void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
        static void var_U_ts (double &u_x, double &u_y, const Matrix &xy_t, const Matrix &xy_s, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &lag_s1,
                              const Matrix &lag_s2, const Matrix &alpha_X, const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s,
							  const Matrix &eta_t, const Matrix &eta_s, const double expn);
		//calculate re-centered Euclidean distances, U_{t,s}, with X_t univariate and Y_t BIVARIATE. INPUT: a 3x1 vector of X_t and Y_t (xy_t),
        //a 3x1 vector of X_s and Y_s (xy_s), a 3x1 vector of the first lags of X_t and Y_t (lag_t1), a 3x1 vector of the second lags of X_t and Y_t (lag_t2),
        //a 3x1 vector of the first lags of X_s and Y_s (lag_s1), a 3x1 vector of the second lags of X_s and Y_s (lag_s2), a 6x1 vectors of AR coefficients (alpha_X),
        //a 6x2 matrix of AR coefficients (alpha_Y), Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by
        //gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed) with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, and an exponent for the Euclidean distance
        //(expn in (1,2)). OUTPUT: U_{t,s}^{(x)} and U_{t,s}^{(y)}.
        template <void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
        static void var_U_ts (double &u_x, double &u_y, const Matrix &xy_t, const Matrix &xy_s, const Matrix &lag_t1, const Matrix &lag_t2,
		                      const Matrix &lag_s1, const Matrix &lag_s2, const Matrix &alpha_X, const Matrix &alpha_Y, const Matrix &epsilon_t,
							  const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t, const Matrix &eta2_s,
							  const double expn);
		//calculate re-centered Euclidean distances, U_{t,s}, with multivariate X_t modeled through a flexible VAR model with B-splines (which is estimated using the adaptive group LASSO).
		//INPUT: a N*L*m by N matrix of B-spline coefficients (beta) for all individual ARDL B-spline regression models, a T0 by N matrix of data (X), a N*L*m by T0 matrix of all B-spline
		//basis functions evaluated at all past data points on the X's (BS), and an exponent of the distance correlation (expn). OUTPUT: a T0 by T0 matrix (U).
		static void var_U (Matrix &U, const Matrix &X, const Matrix &BS, const Matrix &beta, const double expn);
		//calculate re-centered Euclidean distances, U_{t,s}, with multivariate X_t modeled through a multivariate DCC model.
		//INPUT: a T by N matrix of data (X), a T by N matrix of conditional means (cmean), a T by N(N+1)/2 matrix of the square roots of conditional variances/covariances (csigma),
		//a T by N matrix of MGARCH residuals, a distance-correlation exponent (expn). OUTPUT: a T0 by T0 matrix (U).
		static void var_U (Matrix &U, const Matrix &X, const Matrix &cmean, const Matrix &csigma, const Matrix &resid, const double expn);

        //calculate the conditional means for the bivariate Bilinear model
        static void cmean_BL (double &mu_X, double &mu_Y, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &alpha_X, const Matrix &alpha_Y);
        //calculate the conditional means for the bivariate TAR model
        static void cmean_TAR (double &mu_X, double &mu_Y, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &alpha_X, const Matrix &alpha_Y);
        //calculate the conditional means for the trivariate Bilinear model
        static void cmean_BL (double &mu_X, double &mu_Y1, double &mu_Y2, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &alpha_X, const Matrix &alpha_Y);
        //calculate the conditional means for the trivariate TAR model
        static void cmean_TAR (double &mu_X, double &mu_Y1, double &mu_Y2, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &alpha_X, const Matrix &alpha_Y);
	private:
	    //convert the indices of the 'vech' vector to those of the matrix
	    static int index (int i, int j, const int N);

};

//convert the indices of the 'vech' vector to those of the matrix
int NGReg::index (int i, int j, const int N) {
	double res = 0.;
	ASSERT(i >= 2);
	for (int k = 0; k <= i - 2; k++)
		res += (N - k);
	return res + j - i + 1;
}

//calculate the conditional means for the bivariate Bilinear model
void NGReg::cmean_BL (double &mu_X, double &mu_Y, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &alpha_X, const Matrix &alpha_Y) {
    mu_X = alpha_X(1) + alpha_X(2)*lag_t1(1) + alpha_X(3)*pow(lag_t1(1), 2.) + alpha_X(4)*lag_t2(1) + alpha_X(5)*pow(lag_t2(1), 2.)
	            + alpha_X(6)*lag_t1(1)*lag_t2(1); //calculate conditional means
	mu_Y = alpha_Y(1) + alpha_Y(2)*lag_t1(2) + alpha_Y(3)*pow(lag_t1(2), 2.) + alpha_Y(4)*lag_t2(2) + alpha_Y(5)*pow(lag_t2(2), 2.)
	            + alpha_Y(6)*lag_t1(2)*lag_t2(2);
}

//calculate the conditional means for the bivariate TAR model
void NGReg::cmean_TAR (double &mu_X, double &mu_Y, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &alpha_X, const Matrix &alpha_Y) {
    auto pos = [](double x) { //get the positive part of a real number (x)
    	if (x > 0.)
    		return x;
    	else
    		return 0.;
	};
    mu_X = alpha_X(1) + alpha_X(2)*lag_t1(1) + alpha_X(3)*pos(lag_t1(1)) + alpha_X(4)*lag_t2(1) + alpha_X(5)*pos(lag_t2(1));
	mu_Y = alpha_Y(1) + alpha_Y(2)*lag_t1(2) + alpha_Y(3)*pos(lag_t1(2)) + alpha_Y(4)*lag_t2(2) + alpha_Y(5)*pos(lag_t2(2));
}

//calculate the conditional means for the trivariate Bilinear model
void NGReg::cmean_BL (double &mu_X, double &mu_Y1, double &mu_Y2, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &alpha_X, const Matrix &alpha_Y) {
    mu_X = alpha_X(1) + alpha_X(2)*lag_t1(1) + alpha_X(3)*pow(lag_t1(1), 2.) + alpha_X(4)*lag_t2(1) + alpha_X(5)*pow(lag_t2(1), 2.)
	            + alpha_X(6)*lag_t1(1)*lag_t2(1); //calculate conditional means
	mu_Y1 = alpha_Y(1,1) + alpha_Y(2,1)*lag_t1(2) + alpha_Y(3,1)*pow(lag_t1(2), 2.) + alpha_Y(4,1)*lag_t2(2) + alpha_Y(5,1)*pow(lag_t2(2), 2.)
	            + alpha_Y(6,1)*lag_t1(2)*lag_t2(2);
	mu_Y2 = alpha_Y(1,2) + alpha_Y(2,2)*lag_t1(3) + alpha_Y(3,2)*pow(lag_t1(3), 2.) + alpha_Y(4,2)*lag_t2(3) + alpha_Y(5,2)*pow(lag_t2(3), 2.)
	            + alpha_Y(6,2)*lag_t1(3)*lag_t2(3);
}

//calculate the conditional means for the trivariate TAR model
void NGReg::cmean_TAR (double &mu_X, double &mu_Y1, double &mu_Y2, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &alpha_X, const Matrix &alpha_Y) {
    auto pos = [](double x) { //get the positive part of a real number (x)
    	if (x > 0.)
    		return x;
    	else
    		return 0.;
	};
    mu_X =  alpha_X(1) + alpha_X(2)*lag_t1(1) + alpha_X(3)*pos(lag_t1(1)) + alpha_X(4)*lag_t2(1) + alpha_X(5)*pos(lag_t2(1));
    mu_Y1 = alpha_Y(1,1) + alpha_Y(2,1)*lag_t1(2) + alpha_Y(3,1)*pos(lag_t1(2)) + alpha_Y(4,1)*lag_t2(2) + alpha_Y(5,1)*pos(lag_t2(2));
    mu_Y2 = alpha_Y(1,2) + alpha_Y(2,2)*lag_t1(3) + alpha_Y(3,2)*pos(lag_t1(3)) + alpha_Y(4,2)*lag_t2(3) + alpha_Y(5,2)*pos(lag_t2(3));
}

//calculate conditional expectations of |x_t - x_s|^\alpha and |y_t - y_s|^\alpha using the lagged values of x_t and y_t respectively.
//INPUT: a 2x1 vector of the first lags of x_t and y_t (lag_t1), a 2x1 vector of the second lags of x_t and y_t (lag_t2), a 2x1 vector of X_s and Y_s (xy_s),
//6x1 vectors of AR coefficients (alpha_X and alpha_Y), Nx1 vectors of random errors (epsilon and eta) generated by gen_RAN (epsilon, eta, delta, 0., 0, rseed)
//using a random seed (rseed), a delta in (-1,1), and an exponent for the Euclidean distance (expn in (1,2)).
//OUTPUT: conditional expectations E_{X_t|X_{t-1},X_{t-2}}[|X_t - X_s|^expn] and E_{Y_t|Y_{t-1},Y_{t-2}}[|Y_t - Y_s|^expn] (res_Xt and res_Yt).
template <void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
void NGReg::reg (double &res_Xt, double &res_Yt, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &xy_s, const Matrix &alpha_X, const Matrix &alpha_Y,
                 const Matrix &epsilon, const Matrix &eta, const double expn) {
    double mu_X = 0., mu_Y = 0.;
    cmean (mu_X, mu_Y, lag_t1, lag_t2, alpha_X, alpha_Y); //calculate conditional means
	auto x_t = 0., y_t = 0.;
	auto N = epsilon.nRow(), i = 1;
	res_Xt = 0.;
	res_Yt = 0.;
    //#pragma omp parallel for default(shared) reduction(+:sum) schedule(static) private(i) firstprivate(eta1,eta2,Y_t)
    for (i = 1; i <= N; ++i) {
    	//rseed = gsl_rng_get (r); //a random seed
    	//gen_RAN (epsilon, eta, delta, 0., 0, rseed); //generate independent random error terms using a random seed
    	//draw x_t and y_t given its past values, lag_t1 and lag_t2
    	x_t = mu_X + epsilon(i);
    	y_t = mu_Y + eta(i);
    	res_Xt += pow(fabs(x_t - xy_s(1)), expn) / N;
    	res_Yt += pow(fabs(y_t - xy_s(2)), expn) / N;
	}
}

//calculate conditional expectations of |x_t - x_s|^\alpha and ||y_t - y_s||^\alpha using the lagged values of x_t and y_t respectively.
//INPUT: a 3x1 vector of the first lags of x_t and y_t (lag_t1), a 3x1 vector of the second lags of x_t and y_t (lag_t2), a 3x1 vector of X_s and Y_s (xy_s),
//a 6x1 vectors of AR coefficients (alpha_X) and a 6x2 matrix of AR coefficients (alpha_Y), Nx1 vectors of random errors (epsilon, eta1 and eta2) generated by
//gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed) with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, and an exponent for the Euclidean
//distance (expn in (1,2)). OUTPUT: conditional expectations E_{X_t|X_{t-1},X_{t-2}}[|X_t - X_s|^expn] and E_{Y_t|Y_{t-1},Y_{t-2}}[||Y_t - Y_s||^expn] (res_Xt and res_Yt).
template <void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
void NGReg::reg (double &res_Xt, double &res_Yt, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &xy_s, const Matrix &alpha_X, const Matrix &alpha_Y,
                 const Matrix &epsilon, const Matrix &eta1, const Matrix &eta2, const double expn) {
    auto mu_X = 0., mu_Y1 = 0., mu_Y2 = 0.;
    cmean (mu_X, mu_Y1, mu_Y2, lag_t1, lag_t2, alpha_X, alpha_Y); //calculate conditional means
	auto N = epsilon.nRow(), i = 1;
	double x_t = 0., y1_t = 0., y2_t = 0.;
	res_Xt = 0.;
	res_Yt = 0.;
    //#pragma omp parallel for default(shared) reduction(+:res_Xt,res_Yt) schedule(dynamic,CHUNK) private(i) firstprivate(rseed,epsilon,eta1,eta2,x_t,y1_t,y2_t)
    for (i = 1; i <= N; i++) {
    	//rseed = gsl_rng_get (r); //a random seed
    	//generate random error terms (epsilon is independent of eta1 and eta2) using a random seed
    	//gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed);
    	//draw x_t and y_t given its past values, lag_t1 and lag_t2
    	x_t = mu_X + epsilon(i);
    	y1_t = mu_Y1 + eta1(i);
    	y2_t = mu_Y2 + eta2(i);
    	res_Xt += pow(fabs(x_t - xy_s(1)), expn) / N;
    	res_Yt += pow(pow(y1_t - xy_s(2), 2.) + pow(y2_t - xy_s(3), 2.), expn/2) / N;
	}
}

//calculate the conditional expectation of  ||x_t - x_s||^\alpha (conditioning on the lags of x_t) for t = 1, ..., T0 and s = 1, ..., T0 by using the sample averaging.
//INPUT: a N*L*m by N matrix of B-spline coefficients (beta) for all individual ARDL B-spline regression models, a T0 by N matrix of data (X),
//a N*L*m by T0 matrix of all B-spline basis functions evaluated at all past data points on the X's (BS), and an exponent of the distance correlation (expn).
//OUTPUT: a T0 by T0 matrix (reg0).
void NGReg::reg (Matrix &reg0, const Matrix &X, const Matrix &BS, const Matrix &beta, const double expn) {
	auto T0 = X.nRow(), N = X.nCol(), BS_nR = BS.nRow();
	ASSERT (T0 == BS.nCol() && N == beta.nCol() && BS_nR == beta.nRow());
	Matrix resid(T0,N), cmean(T0,N);
	cmean = Tr(BS) * beta;
	resid = X - cmean;
	reg0.set(0.);
	auto t = 0, s = 0, tau = 0, i = 0;
	auto enorm_sq = 0.;
	for (t = 1; t <= T0; ++t) {
		for (s = 1; s <= T0; ++s) {
			for (tau = 1; tau <= T0; ++tau) {
				enorm_sq = 0.; //re-set the initial value to zero
				for (i = 1; i <= N; ++i) {
					enorm_sq += pow(cmean(t,i) + resid(tau,i) - X(s,i), 2.);
				}
				reg0(t,s) += pow(enorm_sq, expn/2.) / T0;
			}
		}
	}
}

//calculate the conditional expectation of  ||x_t - x_s||^\alpha (conditioning on the lags of x_t) for t = 1, ..., T0 and s = 1, ..., T0,
//where x_t follows a multivariate DCC model,
// by using the sample averaging. INPUT: a T by N matrix of data (X), a T by N matrix of conditional means (cmean), a T by N(N+1)/2 matrix of
//the square roots of conditional variances/covariances (csigma), a T by N matrix of MGARCH residuals, a distance-correlation exponent (expn).
//OUTPUT: a T by T matrix (reg0)
void NGReg::reg (Matrix &reg0, const Matrix &X, const Matrix &cmean, const Matrix &csigma, const Matrix &resid, const double expn) {
	auto T0 = X.nRow(), N = X.nCol(), M = csigma.nCol();
	ASSERT (T0 == cmean.nRow() && T0 == csigma.nRow() && T0 == resid.nRow() && N == cmean.nCol() && N == resid.nCol() && M == N*(N+1)/2);

	auto t = 0, s = 0, tau = 0, i = 0, j = 0;
	auto enorm_sq = 0.;
	Matrix sigma(N, N), rresid(T0, N);
	reg0.set(0.);
	for (t = 1; t <= T0; t++) {
		//copy the row csigma(t, ) to a square matrix
		for (j = 1; j <= N; j++) {
			sigma(j, 1) = csigma(t,  j);
			sigma(1, j) = sigma(j, 1);
		}
		if (N > 1) {
			for (i = 2; i <= N; i++) {
				for (j = i ; j <= N; j++) {
					sigma(j, i) = csigma(t, NGReg::index (i, j, N));
					sigma(i, j) = sigma(j, i);
				}
			}
		}
		rresid.set(0.); //reset rresid to zeros for every t
		for (tau = 1; tau <= T0; tau++) {
			for (i = 1; i <= N; i++) {
				for (j = 1; j <= N; j++) {
					rresid(tau,i) += sigma(i,j) * resid(tau, j); //calculate residuals using csigma(t, ) * Tr(resid(tau, ))
				}
			}
		}
		for (s = 1; s <= T0; ++s) {
			for (tau = 1; tau <= T0; ++tau) {
				enorm_sq = 0.; //re-set the initial value to zero
				for (i = 1; i <= N; i++) {
					enorm_sq += pow(cmean(t, i) + rresid(tau, i) - X(s, i), 2.);
				}
				reg0(t,s) += pow(enorm_sq, expn/2.)  /  T0;
			}
		}
	}
}

//calculate conditional expectations of |x_t - x_s|^\alpha and |y_t - y_s|^\alpha using the lagged values of (x_t, x_s) and (y_t, y_s) respectively.
//INPUT: a 2x1 vector of the first lags of x_t and y_t (lag_t1), a 2x1 vector of the second lags of x_t and y_t (lag_t2), a 2x1 vector of the first lags
//of x_s and y_s (lag_s1), a 2x1 vector of the second lags of x_s and y_s (lag_s2), a 6x1 vectors of AR coefficients (alpha_X and alpha_Y), Nx1 vectors of
//random errors (epsilon_t, epsilon_s, eta_t and eta_s) generated by gen_RAN (epsilon, eta, delta, 0., 0, rseed) with random seeds (rseed), a delta in (-1,1),
//and an exponent for the Euclidean distance (expn in (1,2)).
//OUTPUT: conditional expectations E_{X_t|X_{t-1},X_{t-2}}E_{X_s|X_{s-1},X_{s-2}}[|X_t - X_s|^expn] and
//E_{Y_t|Y_{t-1},Y_{t-2}}E_{Y_s|Y_{s-1},Y_{s-2}}[|Y_t - Y_s|^expn] (res_X and res_Y).
template <void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
void NGReg::breg (double &res_X, double &res_Y, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &lag_s1, const Matrix &lag_s2,
                  const Matrix &alpha_X, const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta_t,
				  const Matrix &eta_s, const double expn) {
    auto mu_X = 0., mu_Y = 0.;
    cmean (mu_X, mu_Y, lag_s1, lag_s2, alpha_X, alpha_Y); //calculate conditional means
	auto res_Xt = 0., res_Yt = 0.;
	auto N = epsilon_t.nRow(), i = 1;
	Matrix xy_s(2, 1);
	res_X = 0.;
	res_Y = 0.;
	//#pragma omp parallel for default(shared) reduction(+:res_X,res_Y) schedule(dynamic,CHUNK) private(i) firstprivate(rseed,epsilon,eta,xy_s,res_Xt,res_Yt)
    for (i = 1; i <= N; i++) {
    	//rseed = gsl_rng_get (r); //a random seed
    	//gen_RAN (epsilon, eta, delta, 0., 0, rseed); //generate independent random error terms using a random seed
    	//draw x_s and y_s given its past values, lag_s1 and lag_s2
		xy_s(1) = mu_X + epsilon_s(i);
    	xy_s(2) = mu_Y + eta_s(i);
    	NGReg::reg <cmean> (res_Xt, res_Yt, lag_t1, lag_t2, xy_s, alpha_X, alpha_Y, epsilon_t, eta_t, expn);
    	res_X += res_Xt / N;
    	res_Y += res_Yt / N;
	}
}

//calculate conditional expectations of |x_t - x_s|^\alpha and ||y_t - y_s||^\alpha using the lagged values of (x_t, x_s) and (y_t, y_s) respectively.
//INPUT: a 3x1 vector of the first lags of x_t and y_t (lag_t1), a 3x1 vector of the second lags of x_t and y_t (lag_t2), a 3x1 vector of the first lags
//of x_s and y_s (lag_s1), a 3x1 vector of the second lags of x_s and y_s (lag_s2), a 6x1 vectors of AR coefficients (alpha_X) and a 6x2 matrix AR
//coefficients (alpha_Y), Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by
//gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed) with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, and an exponent for the Euclidean
//distance (expn in (1,2)). OUTPUT: conditional expectations E_{X_t|X_{t-1},X_{t-2}}E_{X_s|X_{s-1},X_{s-2}}[|X_t - X_s|^expn] and
//E_{Y_t|Y_{t-1},Y_{t-2}}E_{Y_s|Y_{s-1},Y_{s-2}}[||Y_t - Y_s||^expn] (res_X and res_Y).
template <void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
void NGReg::breg (double &res_X, double &res_Y, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &lag_s1, const Matrix &lag_s2, const Matrix &alpha_X,
                  const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t, const Matrix &eta1_s, const Matrix &eta2_t,
				  const Matrix &eta2_s, const double expn) {
    double mu_X = 0., mu_Y1 = 0., mu_Y2 = 0.;
    cmean (mu_X, mu_Y1, mu_Y2, lag_s1, lag_s2, alpha_X, alpha_Y);
	auto N = epsilon_t.nRow(), i = 1;
    Matrix xy_s(3, 1);
	res_X = 0.;
	res_Y = 0.;
	double res_Xt = 0., res_Yt = 0.;
	//#pragma omp parallel for default(shared) reduction(+:res_X,res_Y) schedule(dynamic,CHUNK) private(i) firstprivate(rseed,epsilon,eta1,eta2,xy_s,res_Xt,res_Yt)
    for (i = 1; i <= N; i++) {
    	//rseed = gsl_rng_get (r); //set a random seed
    	//cout << rseed << endl;
    	//generate random error terms (epsilon is independent of eta1 and eta2) using a random seed
    	//gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed);
    	//draw x_s and y_s given its past values, lag_s1 and lag_s2
    	xy_s(1) = mu_X + epsilon_s(i);
    	xy_s(2) = mu_Y1 + eta1_s(i);
    	xy_s(3) = mu_Y2 + eta2_s(i);
    	NGReg::reg <cmean> (res_Xt, res_Yt, lag_t1, lag_t2, xy_s, alpha_X, alpha_Y, epsilon_t, eta1_t, eta2_t, expn);
    	res_X += res_Xt / N;
    	res_Y += res_Yt / N;
	}
}

//calculate the conditional expectation of  ||x_t - x_s||^\alpha (conditioning on the lags of  both x_t and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by using the sample averaging.
//INPUT: a N*L*m by N matrix of B-spline coefficients (beta) for all individual ARDL B-spline regression models, a T0 by N matrix of data (X),
//a N*L*m by T0 matrix of all B-spline basis functions evaluated at all past data points on the X's (BS), and an exponent of the distance correlation (expn).
//OUTPUT: a T0 by T0 matrix (breg0).
void NGReg::breg (Matrix &breg0, const Matrix &X, const Matrix &BS, const Matrix &beta, const double expn) {
	auto T0 = X.nRow(), N = X.nCol(), BS_nR = BS.nRow();
	ASSERT (T0 == BS.nCol() && N == beta.nCol() && BS_nR == beta.nRow());
	Matrix resid(T0,N), cmean(T0,N);
	cmean = Tr(BS) * beta;
	resid = X - cmean;
	breg0.set(0.);
	auto t = 0, s = 0, tau1 = 0, tau2 = 0, i = 0;
	auto enorm_sq = 0.;
	//#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,tau1,tau2,i) firstprivate(cmean_t,cmean_s,resid_tau1,resid_tau2)
	for (t = 1; t <= T0; ++t) {
		for (s = t+1; s <= T0; ++s) {
			for (tau1 = 1; tau1 <= T0; ++tau1){
				for (tau2 = 1; tau2 <= T0; ++tau2) {
					enorm_sq = 0.; //re-set the initial value to zero
					for (i = 1; i <= N; ++i) {
						enorm_sq += pow(cmean(t,i) - cmean(s,i) + resid(tau1,i) - resid(tau2,i), 2.);
					}
					breg0(t,s) += pow(enorm_sq, expn/2.) / pow(T0, 2.);
				}
			}
			breg0(s,t) = breg0(t,s);
		}
	}
}

//calculate the conditional expectation of  ||x_t - x_s||^\alpha (conditioning on the lags of  both x_t and x_s) for t = 1, ..., T0 and s = 1, ..., T0, where x_t follows a multivariate
//GARCH process, by using the sample averaging. INPUT: a T by N matrix of data (X), a T by N matrix of conditional means (cmean), a T by N(N+1)/2 matrix of the square roots of
//conditional variances/covariances (csigma), a T by N matrix of MGARCH residuals, a distance-correlation exponent (expn).
//OUTPUT: a T0 by T0 matrix (breg0).
void NGReg::breg (Matrix &breg0, const Matrix &X, const Matrix &cmean, const Matrix &csigma, const Matrix &resid, const double expn) {
	auto T0 = X.nRow(), N = X.nCol(), M = csigma.nCol();
	ASSERT (T0 == cmean.nRow() && T0 == csigma.nRow() && T0 == resid.nRow() && N == cmean.nCol() && N == resid.nCol() && M == N*(N+1)/2);

	float sigma[T0][N][N], rresid[T0][T0][N];
	auto t = 0, s = 0, tau = 0, tau1 = 0, tau2 = 0, i = 0, j = 0;
	auto enorm_sq = 0.;
	breg0.set(0.);

	//copy the vector of sqrts of variances-covariances to a 3-m matrix
	for (t = 1; t <= T0; ++t) {
		//copy the row csigma(t, ) to a square matrix
		for (j = 1; j <= N; j++) {
			sigma[t-1][j-1][0] = csigma(t, j);
			sigma[t-1][0][j-1] = sigma[t-1][j-1][0];
		}
		if  (N > 1) {
			for (i = 2; i <= N; i++) {
				for (j = i ; j <= N; j++) {
					sigma[t-1][j-1][i-1] = csigma(t, NGReg::index (i, j, N));
					sigma[t-1][i-1][j-1] = sigma[t-1][j-1][i-1];
				}
			}
		}
		// calculate the residuals sigma[t][][] * resid(tau, )
		for (tau = 1; tau <= T0; tau++) {
			for (i = 1; i <= N; i++) {
				rresid[t-1][tau-1][i-1] = 0.; //set each element of rresid to zero
				for (j = 1; j <= N; j++) {
					rresid[t-1][tau-1][i-1] += sigma[t-1][i-1][j-1] * resid(tau, j); //calculate residuals using csigma(t, ) * Tr(resid(tau, ))
				}
			}
		}
	}

	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,tau1,tau2,i) firstprivate(enorm_sq)
	for (t = 1; t <= T0; ++t) {
		for (s = t+1; s <= T0; ++s) {
			for (tau1 = 1; tau1 <= T0; ++tau1){
				for (tau2 = 1; tau2 <= T0; ++tau2) {
					enorm_sq = 0.; //re-set the initial value to zero
					for (i = 1; i <= N; i++) {
						enorm_sq += pow(cmean(t, i) - cmean(s, i) + rresid[t-1][tau1-1][i-1] - rresid[s-1][tau2-1][i-1], 2.);
					}
					breg0(t,s) += pow(enorm_sq, expn/2.) / pow(T0, 2.);
				}
			}
			breg0(s,t) = breg0(t,s);
		}
	}
}

//calculate re-centered Euclidean distances, U_{t,s}, with X_t and Y_t are univariate. INPUT: a 2x1 vector of X_t and Y_t (xy_t),
//a 2x1 vector of X_s and Y_s (xy_s), a 2x1 vector of the first lags of X_t and Y_t (lag_t1), a 2x1 vector of the second lags of X_t and Y_t (lag_t2),
//a 2x1 vector of the first lags of X_s and Y_s (lag_s1), a 2x1 vector of the second lags of X_s and Y_s (lag_s2), a 6x1 vectors of AR coefficients
//(alpha_X and alpha_Y), vectors of random errors (epsilon_t, epsilon_s, eta_t and eta_s) generated by gen_RAN (epsilon, eta, delta, 0., 0, rseed)
//with random seeds (rseed), a delta in (-1,1), and an exponent for the Euclidean distance (expn in (1,2)).
//OUTPUT: U_{t,s}^{(x)} and U_{t,s}^{(y)}.
template <void cmean (double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
void NGReg::var_U_ts (double &u_x, double &u_y, const Matrix &xy_t, const Matrix &xy_s, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &lag_s1,
                      const Matrix &lag_s2, const Matrix &alpha_X, const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta_t,
					  const Matrix &eta_s, const double expn) {
	auto mean_Xt = 0., mean_Yt = 0., mean_Xs = 0., mean_Ys = 0., mean_X = 0., mean_Y = 0.;
	#pragma omp parallel sections num_threads(3)
	{
		#pragma omp section
	    //calculate E_{X_t|X_{t-1},X_{t-2}}[|X_t - X_s|^expn] and E_{Y_t|Y_{t-1},Y_{t-2}}[|Y_t - Y_s|^expn]
	    NGReg::reg<cmean> (mean_Xt, mean_Yt, lag_t1, lag_t2, xy_s, alpha_X, alpha_Y, epsilon_t, eta_t, expn);
	    #pragma omp section
	    //calculate E_{X_s|X_{s-1},X_{s-2}}[|X_t - X_s|^expn] and E_{Y_s|Y_{s-1},Y_{s-2}}[|Y_t - Y_s|^expn]
	    NGReg::reg<cmean> (mean_Xs, mean_Ys, lag_s1, lag_s2, xy_t, alpha_X, alpha_Y, epsilon_s, eta_s, expn);
	    #pragma omp section
	    //calculate E_{X_t|X_{t-1},X_{t-2}}E_{X_s|X_{s-1},X_{s-2}}[|X_t - X_s|^expn] and E_{Y_t|Y_{t-1},Y_{t-2}}E_{Y_s|Y_{s-1},Y_{s-2}}[|Y_t - Y_s|^expn]
	    NGReg::breg<cmean> (mean_X, mean_Y, lag_t1, lag_t2, lag_s1, lag_s2, alpha_X, alpha_Y, epsilon_t, epsilon_s, eta_t, eta_s, expn);
    }
	u_x = mean_Xt + mean_Xs - pow(fabs(xy_t(1) - xy_s(1)), expn) - mean_X;
	u_y = mean_Yt + mean_Ys - pow(fabs(xy_t(2) - xy_s(2)), expn) - mean_Y;
}

//calculate re-centered Euclidean distances, U_{t,s}, with X_t univariate and Y_t BIVARIATE. INPUT: a 3x1 vector of X_t and Y_t (xy_t),
//a 3x1 vector of X_s and Y_s (xy_s), a 3x1 vector of the first lags of X_t and Y_t (lag_t1), a 3x1 vector of the second lags of X_t and Y_t (lag_t2),
//a 3x1 vector of the first lags of X_s and Y_s (lag_s1), a 3x1 vector of the second lags of X_s and Y_s (lag_s2), a 6x1 vectors of AR coefficients (alpha_X),
//a 6x2 matrix of AR coefficients (alpha_Y), Nx1 vectors of random errors (epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t and eta2_s) generated by
//gen_RAN (epsilon, eta1, eta2, delta, 0., 0., 0, rseed) with random seeds (rseed), a 3x1 vector (delta) in (-1,1)^3, and an exponent for the Euclidean distance
//(expn in (1,2)). OUTPUT: U_{t,s}^{(x)} and U_{t,s}^{(y)}.
template <void cmean (double &, double &, double &, const Matrix &, const Matrix &, const Matrix &, const Matrix &)>
void NGReg::var_U_ts (double &u_x, double &u_y, const Matrix &xy_t, const Matrix &xy_s, const Matrix &lag_t1, const Matrix &lag_t2, const Matrix &lag_s1,
                      const Matrix &lag_s2, const Matrix &alpha_X, const Matrix &alpha_Y, const Matrix &epsilon_t, const Matrix &epsilon_s, const Matrix &eta1_t,
					  const Matrix &eta1_s, const Matrix &eta2_t, const Matrix &eta2_s, const double expn) {
	auto mean_Xt = 0., mean_Yt = 0., mean_Xs = 0., mean_Ys = 0., mean_X = 0., mean_Y = 0.;
	#pragma omp parallel sections num_threads(3)
	{
	    #pragma omp section
        //calculate E_{X_t|X_{t-1},X_{t-2}}[|X_t - X_s|^expn] and E_{Y_t|Y_{t-1},Y_{t-2}}[||Y_t - Y_s||^expn]
        NGReg::reg<cmean> (mean_Xt, mean_Yt, lag_t1, lag_t2, xy_s, alpha_X, alpha_Y, epsilon_t, eta1_t, eta2_t, expn);
        #pragma omp section
        //calculate E_{X_s|X_{s-1},X_{s-2}}[|X_t - X_s|^expn] and E_{Y_s|Y_{s-1},Y_{s-2}}[||Y_t - Y_s||^expn]
        NGReg::reg<cmean> (mean_Xs, mean_Ys, lag_s1, lag_s2, xy_t, alpha_X, alpha_Y, epsilon_s, eta1_s, eta2_s, expn);
        #pragma omp section
        //calculate E_{X_t|X_{t-1},X_{t-2}}E_{X_s|X_{s-1},X_{s-2}}[|X_t - X_s|^expn] and E_{Y_t|Y_{t-1},Y_{t-2}}E_{Y_s|Y_{s-1},Y_{s-2}}[||Y_t - Y_s||^expn]
        NGReg::breg<cmean> (mean_X, mean_Y, lag_t1, lag_t2, lag_s1, lag_s2, alpha_X, alpha_Y, epsilon_t, epsilon_s, eta1_t, eta1_s, eta2_t, eta2_s, expn);
    }
	u_x = mean_Xt + mean_Xs - pow(fabs(xy_t(1) - xy_s(1)), expn) - mean_X;
	u_y = mean_Yt + mean_Ys - pow(pow(xy_t(2) - xy_s(2), 2.) + pow(xy_t(3) - xy_s(3), 2.), expn/2) - mean_Y;
}

//calculate re-centered Euclidean distances, U_{t,s}, with multivariate X_t modeled through a flexible VAR model with B-splines (which is estimated using the adaptive group LASSO).
////INPUT: a N*L*m by N matrix of B-spline coefficients (beta) for all individual ARDL B-spline regression models, a T0 by N matrix of data (X), a N*L*m by T0 matrix of all B-spline
//basis functions evaluated at all past data points on the X's (BS), and an exponent of the distance correlation (expn). OUTPUT: a T0 by T0 matrix (U).
void NGReg::var_U (Matrix &U, const Matrix &X, const Matrix &BS, const Matrix &beta, const double expn) {
	auto T0 = X.nRow(), N = X.nCol(), BS_nR = BS.nRow();
	ASSERT (T0 == BS.nCol() && T0 == U.nRow() && T0 == U.nCol() && N == beta.nCol() && BS_nR == beta.nRow());
	Matrix cmean_reg(T0,T0), cmean_breg(T0,T0), X_t(N,1), X_s(N,1);
	#pragma omp parallel sections num_threads(2)
	{
	    #pragma omp section
		NGReg::reg (cmean_reg, X, BS, beta, expn); //calculate conditional means of ||X_t - X_s|| (conditioning on the lags of X_t or X_s)
		#pragma omp section
		NGReg::breg (cmean_breg, X, BS, beta, expn); //calculate conditional means of ||X_t - X_s|| (conditioning on the lags of both X_t and X_s)
	}
	auto t = 0, s = 0, i = 0;
	auto enorm_sq = 0.;

	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,i) firstprivate(enorm_sq)
	for (t = 1; t <= T0; ++t) {
		for (s = 1; s <= T0; ++s) {
			enorm_sq = 0.; //re-set the initial value to zero
			for (i = 1; i <= N; ++i) {
				enorm_sq += pow(X(t,i) - X(s,i), 2.);
			}
			U(t,s) = cmean_reg(t,s) + cmean_reg(s,t) - pow(enorm_sq, expn/2.) - cmean_breg(t,s);
		}
	}
}

//calculate re-centered Euclidean distances, U_{t,s}, with multivariate X_t modeled through a multivariate DCC model.
//INPUT: a T by N matrix of data (X), a T by N matrix of conditional means (cmean), a T by N(N+1)/2 matrix of the square roots of conditional variances/covariances (csigma),
//a T by N matrix of MGARCH residuals, a distance-correlation exponent (expn). OUTPUT: a T0 by T0 matrix (U).
void NGReg::var_U (Matrix &U, const Matrix &X, const Matrix &cmean, const Matrix &csigma, const Matrix &resid, const double expn) {
	auto T0 = X.nRow(), N = X.nCol(), M = csigma.nCol();
	ASSERT (T0 == cmean.nRow() && T0 == csigma.nRow() && T0 == resid.nRow() && N == cmean.nCol() && N == resid.nCol() && M == N*(N+1)/2);
	Matrix cmean_reg(T0,T0), cmean_breg(T0,T0);

	NGReg::reg (cmean_reg, X,  cmean,  csigma,  resid, expn); //calculate conditional means of ||X_t - X_s|| (conditioning on the lags of X_t or X_s)
	NGReg::breg (cmean_breg, X, cmean, csigma, resid, expn); //calculate conditional means of ||X_t - X_s|| (conditioning on the lags of both X_t and X_s)

	auto t = 0, s = 0, i = 0;
	auto enorm_sq = 0.;

	//#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,i) firstprivate(enorm_sq)
	for (t = 1; t <= T0; ++t) {
		for (s = t+1; s <= T0; ++s) {
			enorm_sq = 0.; //re-set the initial value to zero
			for (i = 1; i <= N; i++) {
				enorm_sq += pow(X(t,i) - X(s,i), 2.);
			}
			U(t,s) = cmean_reg(t,s) + cmean_reg(s,t) - pow(enorm_sq, expn/2.) - cmean_breg(t,s);
			U(s,t) = U(t,s);
		}
	}
}





















#endif
