#ifndef DGP_H
#define DGP_H

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <asserts.h>

using namespace std;

class Dgp {
	public:
		Dgp () {   };//default constructor
		//constructor for linear d.g.p.'s: alpha_x and alpha_y are 2 by 1 vectors of intercepts, beta_x and beta_y are 2 by 1 vectors of slopes
		Dgp (Matrix alpha_x, Matrix alpha_y, Matrix beta_x, Matrix beta_y);
		~Dgp () {   };//default destructor
		//generate a univariate process and a bivariate process that may have some dependency.
		//INPUT: a 3x1 vector of intercepts (alpha(1) for X, and alpha(2-3) for Y), 3x1 vectors of AR slopes (beta and lambda), a 3x1 vector of std. deviations
        //of error terms (sigma(1) for X, and sigma(2) for Y), a correlation between \eta_1 and \eta_2 (rho) for the Y d.g.p, a threshold value (threshold)
        //used to generate some dependence between X and Y, a seed for the random generator (seed). OUTPUT: a Tx1 matrix (X) and Tx2 matrices (Y1 and Y2).
        void gen_AR1 (Matrix &X, Matrix &Y1, Matrix &Y2, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma, double rho,
                      double threshold, unsigned long seed);
        //generate a univarite time series and a bivariate time series that may have some correlation
        //INPUT: a 3x1 vector of intercepts (alpha(1) for X, and alpha(2-3) for Y), 3x1 vectors of AR slopes (beta and lambda), a 3x1 vector of std. deviations
        //of error terms (sigma(1) for epsilon and eta1; and sigma(2) for xi), a correlation between \eta_1 and \eta_2 (rho) for the Y d.g.p,
        //a value for correlation (corr) between X and Y1, a seed for the random generator (seed). OUTPUT: a Tx1 matrix (X) and Tx2 matrices (Y1 and Y2).
        void gen_CAR1 (Matrix &X, Matrix &Y1, Matrix &Y2, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma, double rho,
                       double corr, unsigned long seed);
		//generate random variables from bivariate nonlinear AR(2) processes
		//INPUT: correlation (rho); OUTPUT: Matrices X(T,2) and Y(T,2)
		void gen_NAR2 (Matrix &X, Matrix &Y, double rho, unsigned long seed);
		//generate two uncorrelated, but dependent, AR processes with Student's t innovations.
        //INPUT: alpha is a 2 by 1 vector of intercepts, beta and lambda are 2 by 1 vectors of slopes, nu is the number of degrees of freedom,
        //sigma is a 2 by 1 vector of scaling paramters, set rho = 0. to generate independent Student's t random variables. OUTPUT: two T by 1 vectors of data (X and Y)
        void gen_TMixedAR (Matrix &X, Matrix &Y, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double nu, const Matrix &sigma,
		                   double rho, unsigned long seed);
		//generate two possibly uncorrelated, but dependent, AR processes with Gaussian innovation.
        //INPUT: alpha is a 2 by 1 vector of intercepts, beta and lambda are 2 by 1 vectors of slopes, rho is a coefficient of dependence
        //OUTPUT: two T by 1 vectors of data (X and Y)
        void gen_MixedAR (Matrix &X, Matrix &Y, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double rho, unsigned long seed);
        //generate two correlated Gaussian AR processes.
        //INPUT: alpha is a 2 by 1 vector of intercepts, beta and lambda are 2 by 1 vectors of slopes, rho is a correlation taking
        //a value in [0,1]. OUTPUT: two T by 1 vectors of data (X and Y)
        void gen_CMixedAR (Matrix &X, Matrix &Y, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double rho, unsigned long seed);
        //calculate residuals for the d.g.p's used by gen_TMixedAR, gen_MixedAR, and gen_CMixedAR. INPUT: a Tx1 vector of data (X).
        //OUTPUT: a (T-2)x1 vector of residuals (resid) and a 3x1 vector of the OLS estimates (slope)
        void gen_Resid (Matrix &resid, Matrix &slope, const Matrix &X);
        //do a simple linear regression
        //INPUT: Tx1 vectors of data (X and Y). OUTPUT: a Tx1 vector of residuals (resid), a slope (slope), and a standard deviation of the residuals (std_dev).
        void gen_Resid (Matrix &resid, double &slope, double &std_dev, const Matrix &X, const Matrix &Y);
        //select an optimal lag length to fit data to an AR(p) process. INPUT: a column matrix of data (Y), a maximum lag (max_lag), a IC to be used (select_IC = "BIC"
        //or "HQIC", or "AIC" by default). OUTPUT: an optimal lag length (opt_lag) and optimal IC (opt_IC)
        void opt_AR_lag (int &opt_lag, double &opt_IC, string select_IC, const Matrix &Y, int max_lag);
        //estimate an AR(p) process. INPUT: data (Y) and an AR lag length (lag_len).
        //OUTPUT: a T-lag_len by 1 vector of residuals, a lag_len + 1 by 1 vector of coefficients, a sum of squared residuals (SSR)
        void est_AR (Matrix &resid, Matrix &coeff, double &SSR, const Matrix &Y, int lag_len);
	private:
		//const double sigma_u = 0.0006, sigma_w = 0.001, rho_epsilon = 0.5, rho_eta = 0.6;
		const double sigma_u = sqrt(0.5), sigma_w = sqrt(0.1), rho_epsilon = 0.5, rho_eta = 0.6;
		Matrix alpha_x, alpha_y, beta_x, beta_y;

};

Dgp::Dgp (Matrix _alpha_x, Matrix _alpha_y, Matrix _beta_x, Matrix _beta_y) {
	alpha_x = _alpha_x;
	alpha_y = _alpha_y;
	beta_x  = _beta_x;
	beta_y  = _beta_y;
}


//generate two uncorrelated, but dependent, AR processes with Student's t innovations.
//INPUT: alpha is a 2 by 1 vector of intercepts, beta and lambda are 2 by 1 vectors of slopes, nu is the number of degrees of freedom,
//sigma is a 2 by 1 vector of scaling paramters, set rho = 0. to generate independent Student's t random variables. OUTPUT: two T by 1 vectors of data (X and Y)
void Dgp::gen_TMixedAR (Matrix &X, Matrix &Y, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double nu, const Matrix &sigma,
                        double rho, unsigned long seed) {
	int t, T, B;
	T = X.nRow();
	B = 50; //burning the first 500 observations
	gsl_rng * r;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    ASSERT_ (X.nRow() == Y.nRow());
    double epsilon {0.}, eta {0.}, sig {0.}, z1 {0.}, z2 {0.};
    Matrix X_tmp(T+B, 1), Y_tmp(T+B, 1);
    X_tmp(1) = gsl_ran_gaussian (r, 0.8);
    X_tmp(2) = gsl_ran_gaussian (r, 0.5);
    Y_tmp(1) = gsl_ran_gaussian (r, 1.);
    Y_tmp(2) = gsl_ran_gaussian (r, 0.7);
	for (t = 3; t <= T + B; t++) {
		if (rho == 0) {
			epsilon = sigma(1) * gsl_ran_tdist (r, nu);//Var(epsilon) = sigma(1)^2 * nu/(nu-2)
			eta = sigma(2) * gsl_ran_tdist (r, nu);//Var(eta) = sigma(2)^2 * nu/(nu-2)
		}
		else {
	        gsl_ran_bivariate_gaussian (r, sigma(1), sigma(2), 0., &z1, &z2);
	        sig = gsl_ran_chisq (r, nu);
	        epsilon = sqrt(nu/sig) * z1;//Var(epsilon) = sigma(1)^2 * nu/(nu-2)
	        eta = sqrt(nu/sig) * z2;//Var(eta) = sigma(2)^2 * nu/(nu-2)
	    }
	    X_tmp(t) = alpha(1) + beta(1) * X_tmp(t-1) + lambda(1) * sin(2 * X_tmp(t-2)) + epsilon;
	    Y_tmp(t) = alpha(2) + beta(2) * Y_tmp(t-1) + lambda(2) * sin(2 * Y_tmp(t-2)) + eta;
	    if (t > B) {
	    	X(t-B) = X_tmp(t);
	    	Y(t-B) = Y_tmp(t);
		}
    }
	gsl_rng_free (r);
}

//generate two possibly uncorrelated, but dependent, AR processes with Gaussian innovation.
//INPUT: alpha is a 2 by 1 vector of intercepts, beta and lambda are 2 by 1 vectors of slopes, rho is a coefficient of dependence
//OUTPUT: two T by 1 vectors of data (X and Y)
void Dgp::gen_MixedAR (Matrix &X, Matrix &Y, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double rho, unsigned long seed) {
	int t, T, B;
	T = X.nRow();
	B = 50; //burning the first 500 observations
	gsl_rng * r;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    ASSERT_ (X.nRow() == Y.nRow());
    double epsilon, eta;
    //int sigma = 1;
    Matrix X_tmp(T+B, 1), Y_tmp(T+B, 1);
    X_tmp(1) = gsl_ran_gaussian (r, 1.8);
    X_tmp(2) = gsl_ran_gaussian (r, 1.5);
    Y_tmp(1) = gsl_ran_gaussian (r, 1.5);
    Y_tmp(2) = gsl_ran_gaussian (r, 0.7);
	for (t = 3; t <= T + B; t++) {
	    epsilon = gsl_ran_gaussian (r, 1.5); //sigma(1) = 1.5
	    X_tmp(t) = alpha(1) + beta(1) * X_tmp(t-1) + lambda(1) * sin(2 * X_tmp(t-2)) + epsilon;
        //sigma = 2 * gsl_ran_bernoulli (r, 0.5) - 1; //generate a Rademacher random variable
        if (fabs(epsilon) <= rho) {
            eta  = epsilon;
        }
        else {
            eta = -epsilon;
        }                                    //sigma(2) = 1.5
        //eta = rho * (pow (epsilon, 2.) - 1); //sigma(2) = 2 * rho^2 + 1
	    Y_tmp(t) = alpha(2) + beta(2) * Y_tmp(t-1) + lambda(2) * sin(2 * Y_tmp(t-2)) + eta;
	    if (t > B) {
	    	X(t-B) = X_tmp(t);
	    	Y(t-B) = Y_tmp(t);
		}
    }
	gsl_rng_free (r);
}

//generate two correlated Gaussian AR processes.
//INPUT: alpha is a 2 by 1 vector of intercepts, beta and lambda are 2 by 1 vectors of slopes, rho is a correlation taking
//a value in [0,1]. OUTPUT: two T by 1 vectors of data (X and Y)
void Dgp::gen_CMixedAR (Matrix &X, Matrix &Y, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double rho, unsigned long seed) {
	int t, T, B;
	T = X.nRow();
	B = 50; //burning the first 500 observations
	gsl_rng * r;
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    ASSERT_ (X.nRow() == Y.nRow());
    double epsilon, eta;
    Matrix X_tmp(T+B, 1), Y_tmp(T+B, 1);
    X_tmp(1) = gsl_ran_gaussian (r, 1.8);
    X_tmp(2) = gsl_ran_gaussian (r, 1.5);
    Y_tmp(1) = gsl_ran_gaussian (r, 1.4);
    Y_tmp(2) = gsl_ran_gaussian (r, 0.7);
	for (t = 3; t <= T + B; t++) {
	    gsl_ran_bivariate_gaussian (r, 1.5, 1.5, rho, &epsilon, &eta);
	    //cout << epsilon << " , " << eta << endl;
	    X_tmp(t) = alpha(1) + beta(1) * X_tmp(t-1) + lambda(1) * sin(2 * X_tmp(t-2)) + epsilon;
	    Y_tmp(t) = alpha(2) + beta(2) * Y_tmp(t-1) + lambda(2) * sin(2 * Y_tmp(t-2)) + eta;
	    if (t > B) {
	    	X(t-B) = X_tmp(t);
	    	Y(t-B) = Y_tmp(t);
		}
    }
	gsl_rng_free (r);
}

//calculate residuals for the d.g.p's used by gen_TMixedAR, gen_MixedAR, gen_CMixedAR, and gen_AR1. INPUT: a Tx1 vector of data (X).
//OUTPUT: a (T-2)x1 vector of residuals (resid) and a 3x1 vector of the OLS estimates (slope)
void Dgp::gen_Resid (Matrix &resid, Matrix &slope, const Matrix &X) {
	int t = 1, T = 1;
	T = X.nRow();
	Matrix X_t(3, 1), XX_t(3, 3), denom(3, 3), num(3, 1), denom_Inv(3, 3);
	X_t(1) = 1.;
	denom.set(0.);
	num.set(0.);
	for (t = 3; t <= T; ++t) {
	    X_t(2) = X(t-1);
	    X_t(3) = sin (2 * X(t-2));
	    XX_t = X_t * Tr(X_t);
	    denom = denom + XX_t;
	    num = num + (X(t) * X_t);
	}
	denom_Inv = inv(denom);
	slope = denom_Inv * num;
	for (t = 3; t <= T; ++t) {
		X_t(2) = X(t-1);
	    X_t(3) = sin(2 * X(t-2));
	    resid(t-2) = X(t) - (Tr(X_t) * slope);
	}
}

//do a simple linear regression
//INPUT: Tx1 vectors of data (X and Y). OUTPUT: a Tx1 vector of residuals (resid), a slope (slope), and a standard deviation of the residuals (std_dev).
void Dgp::gen_Resid (Matrix &resid, double &slope, double &std_dev, const Matrix &X, const Matrix &Y) {
	int t = 1, T = 1;
	T = X.nRow();
	ASSERT_ (T == Y.nRow());
	double mean_X = 0., mean_Y = 0., num = 0., denom = 0., interc = 0.;
	mean_X = mean_u(X);
	mean_Y = mean_u(Y);
	for (t = 1; t <= T; ++t) {
		num += (X(t) - mean_X) * (Y(t) - mean_Y);
		denom += pow(X(t) - mean_X, 2.);
	}
	slope = num / denom;
	interc = mean_Y - slope * mean_X;
	for (t = 1; t <= T; ++t) {
		resid(t) = Y(t) - interc - slope * X(t);
	}
	std_dev = sqrt(variance (resid));
}

//select an optimal lag length to fit data to an AR(p) process. INPUT: a column matrix of data (Y), a maximum lag (max_lag), a IC to be used (select_IC = "BIC"
//or "HQIC", or "AIC" by default). OUTPUT: an optimal lag length (opt_lag) and optimal IC (opt_IC)
void Dgp::opt_AR_lag (int &opt_lag, double &opt_IC, string select_IC, const Matrix &Y, int max_lag) {
	int lag = 1, t = 1, i = 1, T = 1;
	T = Y.nRow();
	double SSR = 0.;
	Matrix AIC(max_lag, 1), BIC(max_lag, 1), HQIC(max_lag, 1);
	for (lag = 1; lag <= max_lag; ++lag) {
		Matrix Z(T-lag, lag+1), Yd(T-lag, 1), ZZ(lag+1, lag+1), ZZ_inv(lag+1, lag+1), ZY(lag+1, 1), slope(lag+1, 1);
		for (t = 1; t <= T-lag; ++t) {
			Z(t,1) = 1.;
			for (i = 2; i <= lag+1; ++i) {
				Z(t,i) = Y(lag+t-i+1);
			}
			Yd(t) = Y(lag+t);
		}
		ZZ = Tr(Z) * Z;
		ZZ_inv = inv(ZZ);
		ZY = Tr(Z) * Yd;
		slope = ZZ_inv * ZY;
		Matrix err(T-lag, 1);
		err = Yd - (Z * slope);
		SSR = Tr(err) * err;
		AIC(lag) = log (((double) SSR/T)) + 2 * (lag + 1) * ((double) 1 / T);
		BIC(lag) = log (((double) SSR/T)) + (lag + 1) * ((double) log(T) / T);
		//cout << BIC(lag) << endl;
		HQIC(lag) = log (((double) SSR/T)) + 2 * (lag + 1) * ((double) log(log(T)) / T);
		//cout << AIC(lag) << endl;
	}
	Matrix min_IC(2, 1);
	if (select_IC == "BIC") {
		min_IC = minn (BIC);
		opt_lag = min_IC(1);
		opt_IC = min_IC(2);
	}
	else if (select_IC == "HQIC") {
		min_IC = minn (HQIC);
		opt_lag = min_IC(1);
		opt_IC = min_IC(2);
	}
	else {
		min_IC = minn (AIC);
		opt_lag = min_IC(1);
		opt_IC = min_IC(2);
	}
}

//estimate an AR(p) process. INPUT: data (Y) and an AR lag length (lag_len).
//OUTPUT: a T-lag_len by 1 vector of residuals, a lag_len + 1 by 1 vector of coefficients, a sum of squared residuals (SSR)
void Dgp::est_AR (Matrix &resid, Matrix &coeff, double &SSR, const Matrix &Y, int lag_len) {
	int t = 1, i = 1, T = 1;
	T = Y.nRow();
	Matrix Z(T-lag_len, lag_len+1), Yd(T-lag_len, 1), ZZ(lag_len+1, lag_len+1), ZZ_inv(lag_len+1, lag_len+1), ZY(lag_len+1, 1);
	for (t = 1; t <= T-lag_len; ++t) {
		Z(t,1) = 1.;
		for (i = 2; i <= lag_len+1; ++i) {
			Z(t,i) = Y(lag_len+t-i+1);
		}
			Yd(t) = Y(lag_len+t);
	}
	ZZ = Tr(Z) * Z;
	ZZ_inv = inv(ZZ);
	ZY = Tr(Z) * Yd;
	coeff = ZZ_inv * ZY;
	resid = Yd - (Z * coeff);
	SSR = Tr(resid) * resid;
}

//generate a univariate process and a bivariate process that may have some dependency.
//INPUT: a 3x1 vector of intercepts (alpha(1) for X, and alpha(2-3) for Y), 3x1 vectors of AR slopes (beta and lambda), a 3x1 vector of std. deviations
//of error terms (sigma(1) for X, and sigma(2) for Y), a correlation between \eta_1 and \eta_2 (rho) for the Y d.g.p, a threshold value (threshold)
//used to generate some dependence between X and Y, a seed for the random generator (seed). OUTPUT: a Tx1 matrix (X) and Tx1 matrices (Y1 and Y2).
void Dgp::gen_AR1 (Matrix &X, Matrix &Y1, Matrix &Y2, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma, double rho,
                   double threshold, unsigned long seed) {
	int t = 1, T = 1, B = 1;
	T = X.nRow();
	ASSERT_ (T == Y1.nRow() && T == Y2.nRow());
	B = 50;//burning the first 500 observations
	gsl_rng * r;
    const gsl_rng_type *gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double epsilon = 0., eta1 = 0., eta2 = 0.;
    Matrix X_tmp(T+B, 1), Y_tmp(T+B, 2);
    X_tmp(1) = gsl_ran_gaussian (r, 0.5);
    Y_tmp(1, 1) = gsl_ran_gaussian (r, 0.5);
    Y_tmp(1, 2) = gsl_ran_gaussian (r, 0.5);
    for (t = 3; t <= T + B; ++t) {
	    epsilon = gsl_ran_gaussian(r, sigma(1));
	    X_tmp(t) = alpha(1) + beta(1) * X_tmp(t-1) + lambda(1) * sin(2 * X_tmp(t-2)) + epsilon;
	    if (threshold == 0.) {
	    	eta1 = gsl_ran_gaussian(r, sigma(1));//X and Y are independent of each other
		}
		else {
	        if (fabs(epsilon) <= threshold * sigma(1)) {//set threshold equal to 1.54 for zero correlation
	            eta1 = epsilon;
		    }
		    else {
			    eta1 = -epsilon;
		    }
		}
	    Y_tmp(t,1) = alpha(2) + beta(2) * Y_tmp(t-1,1) + lambda(2) * sin(2 * Y_tmp(t-2,1)) + eta1;
	    eta2 = rho * eta1 + gsl_ran_gaussian (r, sigma(2));
	    Y_tmp(t,2) = alpha(3) + beta(3) * Y_tmp(t-1,2) + lambda(3) * sin(2 * Y_tmp(t-2,2)) + eta2;
	    if (t > B) {
	    	X(t-B) = X_tmp(t);
	    	Y1(t-B) = Y_tmp(t, 1);
	    	Y2(t-B) = Y_tmp(t, 2);
		}
    }
	gsl_rng_free (r);
}

//generate a univarite time series and a bivariate time series that may have some correlation
//INPUT: a 3x1 vector of intercepts (alpha(1) for X, and alpha(2-3) for Y), 3x1 vectors of AR slopes (beta and lambda), a 3x1 vector of std. deviations
//of error terms (sigma(1) for epsilon and eta1; and sigma(2) for xi), a correlation between \eta_1 and \eta_2 (rho) for the Y d.g.p,
//a value for correlation (corr) between X and Y, a seed for the random generator (seed). OUTPUT: a Tx1 matrix (X) and Tx1 matrices (Y1 and Y2).
void Dgp::gen_CAR1 (Matrix &X, Matrix &Y1, Matrix &Y2, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma, double rho,
                    double corr, unsigned long seed) {
	int t = 1, T = 1, B = 1;
	T = X.nRow();
	ASSERT_ (T == Y1.nRow() && T == Y2.nRow());
	B = 50;//burning the first 500 observations
	gsl_rng * r;
    const gsl_rng_type *gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double epsilon = 0., eta1 = 0., eta2 = 0.;
    Matrix X_tmp(T+B, 1), Y_tmp(T+B, 2);
    X_tmp(1) = gsl_ran_gaussian (r, 0.5);
    Y_tmp(1, 1) = gsl_ran_gaussian (r, 0.5);
    Y_tmp(1, 2) = gsl_ran_gaussian (r, 0.5);
    for (t = 3; t <= T + B; ++t) {
	    gsl_ran_bivariate_gaussian (r, sigma(1), sigma(1), corr, &epsilon, &eta1);
	    X_tmp(t) = alpha(1) + beta(1) * X_tmp(t-1) + lambda(1) * sin(2 * X_tmp(t-2)) + epsilon;
	    Y_tmp(t,1) = alpha(2) + beta(2) * Y_tmp(t-1,1) + lambda(2) * sin(2 * Y_tmp(t-2,1)) + eta1;
	    eta2 = rho * eta1 + gsl_ran_gaussian (r, sigma(2));
	    Y_tmp(t,2) = alpha(3) + beta(3) * Y_tmp(t-1,2) + lambda(3) * sin(2 * Y_tmp(t-2,2)) + eta2;
	    if (t > B) {
	    	X(t-B) = X_tmp(t);
	    	Y1(t-B) = Y_tmp(t, 1);
	    	Y2(t-B) = Y_tmp(t, 2);
		}
    }
	gsl_rng_free (r);
}

void Dgp::gen_NAR2 (Matrix &X, Matrix &Y, double rho, unsigned long seed) {
	int t, T, B;
	T = X.nRow();
	B = 50;//burning the first 500 observations
	gsl_rng * r;
    const gsl_rng_type *gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    ASSERT_ (X.nCol() == Y.nCol());
    double w, epsilon_1, epsilon_2, eta_1, eta_2;
    int sigma = 1;
    Matrix X_tmp(T+B, 2), Y_tmp(T+B, 2);
    X_tmp(1, 1) = gsl_ran_gaussian (r, 0.05);
    X_tmp(2, 1) = gsl_ran_gaussian (r, 0.05);
    X_tmp(1, 2) = gsl_ran_gaussian (r, 0.01);
    X_tmp(2, 2) = gsl_ran_gaussian (r, 0.01);
    Y_tmp(1, 1) = gsl_ran_gaussian (r, 0.01);
    Y_tmp(2, 1) = gsl_ran_gaussian (r, 0.01);
    Y_tmp(1, 2) = gsl_ran_gaussian (r, 0.01);
    Y_tmp(2, 2) = gsl_ran_gaussian (r, 0.01);
    for (t = 3; t <= T + B; t++) {
    	w = gsl_ran_gaussian (r, sigma_w);
	    epsilon_1 = gsl_ran_gaussian (r, sigma_u) + w;
	    epsilon_2 = rho_epsilon * epsilon_1 + gsl_ran_gaussian (r, sigma_u);
	    X_tmp(t,1) = alpha_x(1) + beta_x(1) * X_tmp(t-1,1) + 10 * X_tmp(t-1,1) * exp(-pow(X_tmp(t-1,1), 2.)) - sin(2 * X_tmp(t-2,1)) + epsilon_1;
	    X_tmp(t,2) = alpha_x(2) + beta_x(2) * X_tmp(t-1,2) + 10 * X_tmp(t-1,2) * exp(-pow(X_tmp(t-1,2), 2.)) - sin(2 * X_tmp(t-2,2)) + epsilon_2;
	    eta_1 = gsl_ran_gaussian (r, sigma_u) + rho * w *  sigma;
	    eta_2 = rho_eta * eta_1 + gsl_ran_gaussian (r, sigma_u);
	    Y_tmp(t,1) = alpha_y(1) + beta_y(1) * Y_tmp(t-1,1) - sin(-2 * Y_tmp(t-2,1)) + eta_1;
	    Y_tmp(t,2) = alpha_y(2) + beta_y(2) * Y_tmp(t-1,2) - sin(-2 * Y_tmp(t-2,2)) + eta_2;
	    if (t > B) {
	    	X(t-B, 1) = X_tmp(t, 1);
	    	X(t-B, 2) = X_tmp(t, 2);
	    	Y(t-B, 1) = Y_tmp(t, 1);
	    	Y(t-B, 2) = Y_tmp(t, 2);
		}
	}
	gsl_rng_free (r);
}



#endif
