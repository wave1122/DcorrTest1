#ifndef STUDENT_REG_H
#define STUDENT_REG_H

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>


using namespace std;

class TReg {
	public:
		TReg () {   };//default constructor
		~TReg () {   };//default destructor
		//calculate U_{t,s}
		double var_U_ts (double x_t, double x_s, double x_t_lag1, double x_t_lag2, double x_s_lag1, double x_s_lag2, double alpha, double beta, double lambda, 
		                 double nu, double sigma, unsigned long seed);
		//calculate conditional expectations of |x_t-x_s| using lagged values of x_t. INPUT: nu > 1 is the number of degrees of freedom, sigma is a standard deviation
        //OUTPUT: a double number
        double reg_F (double x_t_lag1, double x_t_lag2, double x_s, double alpha, double beta, double lambda, double nu, double sigma);
		//calculate conditional expectations of |x_t-x_s| using lagged values of x_t and x_s. INPUT: nu > 1 is the number of degrees of freedom, 
        //sigma is a scaling parameter and a seed for the random number generator (seed). OUTPUT: a double number
        double reg_BF (double x_t_lag1, double x_t_lag2, double x_s_lag1, double x_s_lag2, double alpha, double beta, double lambda, double nu, double sigma, 
		               unsigned long seed);
};

//calculate conditional expectations of |x_t-x_s| using lagged values of x_t. INPUT: nu > 1 is the number of degrees of freedom, sigma is a standard deviation
//OUTPUT: a double number
double TReg::reg_F (double x_t_lag1, double x_t_lag2, double x_s, double alpha, double beta, double lambda, double nu, double sigma) {
	double mu = alpha + beta*x_t_lag1 + lambda*sin(2*x_t_lag2) - x_s;
	double cC = gsl_sf_gamma((nu+1)/2) / (gsl_sf_gamma(nu/2) * sqrt(M_PI*nu));
	return mu * (2 * gsl_cdf_tdist_Q(-mu/sigma, nu) - 1) + 2 * (sigma * cC * nu) / ((nu-1) * pow(1 + pow(mu/sigma, 2.)/nu, (nu-1)/2));
} 

//calculate conditional expectations of |x_t-x_s| using lagged values of x_t and x_s. INPUT: nu > 1 is the number of degrees of freedom, 
//sigma is a scaling parameter and a seed for the random number generator (seed). OUTPUT: a double number
double TReg::reg_BF (double x_t_lag1, double x_t_lag2, double x_s_lag1, double x_s_lag2, double alpha, double beta, double lambda, double nu, 
                     double sigma, unsigned long seed) {
	gsl_rng * r;
    const gsl_rng_type *gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double x_s = 0., sum = 0.;
    auto i = 1, N = 1000;
    //#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) private(i) firstprivate(x_s)
    for (i = 1; i <= N; i++) {
    	//draw x_s from a conditional Student's t pdf conditioning on its past values, x_s_lag1 and x_s_lag2
    	x_s = sigma * gsl_ran_tdist  (r, nu) + (alpha + beta * x_s_lag1 + lambda * sin(2 * x_s_lag2));
    	sum += ((double) 1/N) * TReg::reg_F (x_t_lag1, x_t_lag2, x_s, alpha, beta, lambda, nu, sigma);
	}
    gsl_rng_free (r);
    return sum;
}

//calculate U_{t,s}. INPUT: nu is the number of degrees of freedom, sigma is the standard deviation. OUTPUT: a double number
double TReg::var_U_ts (double x_t, double x_s, double x_t_lag1, double x_t_lag2, double x_s_lag1, double x_s_lag2, double alpha, double beta, 
                       double lambda, double nu, double sigma, unsigned long seed) {
    return TReg::reg_F (x_t_lag1, x_t_lag2, x_s, alpha, beta, lambda, nu, sigma) + TReg::reg_F (x_s_lag1, x_s_lag2, x_t, alpha, beta, lambda, nu, sigma)
           - fabs(x_t - x_s) - TReg::reg_BF (x_t_lag1, x_t_lag2, x_s_lag1, x_s_lag2, alpha, beta, lambda, nu, sigma, seed);                   	
}




#endif
