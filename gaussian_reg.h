#ifndef GAUSSIAN_REG_H
#define GAUSSIAN_REG_H

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


using namespace std;

class GReg {
	public:
		GReg () {   };//default constructor
		~GReg () {   };//default destructor
		//calculate U_{t,s}
        double var_U_ts (double x_t, double x_s, double x_t_lag1, double x_t_lag2, double x_s_lag1, double x_s_lag2, double alpha, double beta, double lambda, 
		                 double sigma, unsigned long seed);
		//calculate U_{t,s} for bivariate time series. INPUT: 2x1 matrices of y_t and y_s and their first- and second- order lagsintercepts (alpha), 
        //slopes (beta and lambda), a standard deviation of eta_1 (sigma(1)), a standard deviation of xi (sigma(2)), a value for the correlation between eta_1 
        //and eta_2 (rho), and a seed for the random generator. OUTPUT: a double number
        double var_U_ts (const Matrix &y_t, const Matrix &y_s, const Matrix &y_t_lag1, const Matrix &y_t_lag2, const Matrix &y_s_lag1, const Matrix &y_s_lag2, 
                         const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma, double rho, unsigned long seed);
		//calculate conditional expectations of |x_t-x_s| using lagged values of x_t
        double reg_F (double x_t_lag1, double x_t_lag2, double x_s, double alpha, double beta, double lambda, double sigma);
        //calculate conditional expectations of ||y_t-y_s|| using lagged values of the 2x1 vector y_t. INPUT: 2x1 matrices of lags of y_t (y_t_lag_1 and y_t_lag2),
        //a 2x1 matrix of scale variables (y_s), intercepts (alpha), slopes (beta and lambda), a standard deviation of eta_1 (sigma(1)), 
        //a standard deviation of xi (sigma(2)), a value for the correlation between eta_1 and eta_2 (rho), and a seed for the random generator. OUTPUT: a double number
        double reg_F (const Matrix &y_t_lag1, const Matrix &y_t_lag2, const Matrix &y_s, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, 
                      const Matrix &sigma, double rho, unsigned long seed);
        //calculate conditional expectations of |x_t-x_s| using lagged values of x_t and x_s
        double reg_BF (double x_t_lag1, double x_t_lag2, double x_s_lag1, double x_s_lag2, double alpha, double beta, double lambda, double sigma, unsigned long seed);
        //calculate conditional expectations of ||y_t-y_s|| using lagged values of the 2x1 vectors y_t and y_s. 
        //INPUT: 2x1 matrices of lags of y_t (y_t_lag_1 and y_t_lag2), 2x1 matrices of lags of y_s (y_s_lag1 and y_s_lag2), intercepts (alpha), 
		//slopes (beta and lambda), a standard deviation of eta_1 (sigma(1)), a standard deviation of xi (sigma(2)), a value for the correlation 
		//between eta_1 and eta_2 (rho), and a seed for the random generator. OUTPUT: a double number
        double regBF (const Matrix &y_t_lag1, const Matrix &y_t_lag2, const Matrix &y_s_lag1, const Matrix &y_s_lag2, const Matrix &alpha, const Matrix &beta, 
                      const Matrix &lambda, const Matrix &sigma, double rho, unsigned long seed);
};

//calculate conditional expectations of |x_t-x_s| using lagged values of x_t
double GReg::reg_F (double x_t_lag1, double x_t_lag2, double x_s, double alpha, double beta, double lambda, double sigma) {
	double mu = alpha + beta*x_t_lag1 + lambda*sin(2*x_t_lag2) - x_s;
	return sigma * sqrt((double) 2/M_PI) * exp(-pow(mu, 2.)/(2*pow(sigma, 2.))) + mu * (1 - 2 * gsl_cdf_ugaussian_P(-mu/sigma));
} 

//calculate conditional expectations of ||y_t-y_s|| using lagged values of the 2x1 vector y_t. INPUT: 2x1 matrices of lags of y_t (y_t_lag_1 and y_t_lag2),
//a 2x1 matrix of scale variables (y_s), intercepts (alpha), slopes (beta and lambda), a standard deviation of eta_1 (sigma(1)), 
//a standard deviation of xi (sigma(2)), a value for the correlation between eta_1 and eta_2 (rho), and a seed for the random generator. OUTPUT: a double number
double GReg::reg_F (const Matrix &y_t_lag1, const Matrix &y_t_lag2, const Matrix &y_s, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, 
                    const Matrix &sigma, double rho, unsigned long seed) {
	gsl_rng * r;
    const gsl_rng_type *gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double eta1 = 0., eta2 = 0., sum = 0.;
    int i = 1, N = 500;
    Matrix Y_t(2, 1), mu_Y(2, 1);
    mu_Y(1) = alpha(1) + beta(1) * y_t_lag1(1) + lambda(1) * sin(2 * y_t_lag2(1));
    mu_Y(2) = alpha(2) + beta(2) * y_t_lag1(2) + lambda(2) * sin(2 * y_t_lag2(2));
    //#pragma omp parallel for default(shared) reduction(+:sum) schedule(static) private(i) firstprivate(eta1,eta2,Y_t)
    for (i = 1; i <= N; ++i) {
    	eta1 = gsl_ran_gaussian (r, sigma(1));
    	//draw Y_t from a conditional bivariate Gaussian pdf given its past values, y_t_lag1 and y_t_lag2
    	Y_t(1) =  mu_Y(1) + eta1;
    	eta2 = rho * eta1 + gsl_ran_gaussian (r, sigma(2));
    	Y_t(2) =  mu_Y(2) + eta2;
    	sum += ((double) 1/N) * sqrt(pow(Y_t(1) - y_s(1), 2.) + pow(Y_t(2) - y_s(2), 2.));
	}
	gsl_rng_free (r);
    return sum;
}


//calculate conditional expectations of |x_t-x_s| using lagged values of x_t and x_s
double GReg::reg_BF (double x_t_lag1, double x_t_lag2, double x_s_lag1, double x_s_lag2, double alpha, double beta, double lambda, double sigma, 
                     unsigned long seed) {
	gsl_rng * r;
    const gsl_rng_type *gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double x_s = 0., sum = 0.;
    int i = 1, N = 1000;
    //#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) private(i) firstprivate(x_s)
    for (i = 1; i <= N; i++) {
    	//draw x_s from a conditional pdf conditioning on its past values, x_s_lag1 and x_s_lag2
    	x_s = gsl_ran_gaussian (r, sigma) + (alpha + beta * x_s_lag1 + lambda * sin(2 * x_s_lag2));
    	sum += ((double) 1/N) * GReg::reg_F (x_t_lag1, x_t_lag2, x_s, alpha, beta, lambda, sigma);
	}
    gsl_rng_free (r);
    return sum;
}

//calculate conditional expectations of ||y_t-y_s|| using lagged values of the 2x1 vectors y_t and y_s. 
//INPUT: 2x1 matrices of lags of y_t (y_t_lag_1 and y_t_lag2), 2x1 matrices of lags of y_s (y_s_lag1 and y_s_lag2), intercepts (alpha), slopes (beta and lambda), 
//a standard deviation of eta_1 (sigma(1)), a standard deviation of xi (sigma(2)), a value for the correlation between eta_1 and eta_2 (rho), 
//and a seed for the random generator. OUTPUT: a double number
double GReg::regBF (const Matrix &y_t_lag1, const Matrix &y_t_lag2, const Matrix &y_s_lag1, const Matrix &y_s_lag2, const Matrix &alpha, const Matrix &beta, 
                    const Matrix &lambda, const Matrix &sigma, double rho, unsigned long seed) {
    gsl_rng * r;
    const gsl_rng_type *gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double eta1 = 0., eta2 = 0., sum = 0.;
    int i = 1, N = 500;
    Matrix Y_s(2, 1), mu_Y(2, 1);
    mu_Y(1) = alpha(1) + beta(1) * y_s_lag1(1) + lambda(1) * sin(2 * y_s_lag2(1));
    mu_Y(2) = alpha(2) + beta(2) * y_s_lag1(2) + lambda(2) * sin(2 * y_s_lag2(2));
    //#pragma omp parallel for default(shared) reduction(+:sum) schedule(dynamic) private(i) firstprivate(eta1,eta2,Y_s)
    for (i = 1; i <= N; ++i) {
    	eta1 = gsl_ran_gaussian (r, sigma(1));
    	//draw Y_t from a conditional bivariate Gaussian pdf given its past values, y_t_lag1 and y_t_lag2
    	Y_s(1) =  mu_Y(1) + eta1;
    	eta2 = rho * eta1 + gsl_ran_gaussian (r, sigma(2));
    	Y_s(2) =  mu_Y(2) + eta2;
    	sum += ((double) 1/N) * GReg::reg_F (y_t_lag1, y_t_lag2, Y_s, alpha, beta, lambda, sigma, rho, seed);
	}
	gsl_rng_free (r);
    return sum;
}



//calculate U_{t,s} for univariate time series
double GReg::var_U_ts (double x_t, double x_s, double x_t_lag1, double x_t_lag2, double x_s_lag1, double x_s_lag2, 
                       double alpha, double beta, double lambda, double sigma, unsigned long seed) {
    return GReg::reg_F (x_t_lag1, x_t_lag2, x_s, alpha, beta, lambda, sigma) + GReg::reg_F (x_s_lag1, x_s_lag2, x_t, alpha, beta, lambda, sigma)
           - fabs(x_t - x_s) - GReg::reg_BF (x_t_lag1, x_t_lag2, x_s_lag1, x_s_lag2, alpha, beta, lambda, sigma, seed);
                       	
}

//calculate U_{t,s} for bivariate time series. INPUT: 2x1 matrices of y_t and y_s and their first- and second- order lags, two intercepts (alpha), 
//slopes (beta and lambda), a standard deviation of eta_1 (sigma(1)), a standard deviation of xi (sigma(2)), a value for the correlation between eta_1 
//and eta_2 (rho), and a seed for the random generator. OUTPUT: a double number
double GReg::var_U_ts (const Matrix &y_t, const Matrix &y_s, const Matrix &y_t_lag1, const Matrix &y_t_lag2, const Matrix &y_s_lag1, const Matrix &y_s_lag2, 
                       const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma, double rho, unsigned long seed) {
    return GReg::reg_F (y_t_lag1, y_t_lag2, y_s, alpha, beta, lambda, sigma, rho, seed) 
	       + GReg::reg_F (y_s_lag1, y_s_lag2, y_t, alpha, beta, lambda, sigma, rho, seed) 
		   - std::sqrt (pow(y_t(1) - y_s(1), 2.) + pow(y_t(2) - y_s(2), 2.)) 
		   - GReg::regBF (y_t_lag1, y_t_lag2, y_s_lag1, y_s_lag2, alpha, beta, lambda, sigma, rho, seed);                     	
}






#endif
