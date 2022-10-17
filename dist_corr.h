#ifndef DIST_CORR_H
#define DIST_CORR_H

//#include <boost/numeric/conversion/cast.hpp>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <asserts.h>
#include <kernel.h>
#include <nreg.h>
#include <gaussian_reg.h>
#include <student_reg.h>

#define CHUNK 1

using namespace std;

class Dist_corr : public NReg, public GReg, public TReg  {
	public:
		Dist_corr(){  }; //default constructor
		Dist_corr(const Matrix &, const Matrix &);
		~Dist_corr () { };//default destructor
		//calculate the distance covariance. INPUT: X and Y are T by d_x data matrices, a time lag (lag), a lag truncation (TL), and a bandwidth (bandw)
        //OUTPUT: the value of the distance covariance
        template <double kernel_f (double )>
        double cov_XY (int lag, int TL, double bandw);
        //calculate distance covariance for data generated from a bivarate Student's t AR(2) process. INPUT: a time lag (lag), a lag truncation (TL), intercepts (alpha),
        //first-order AR slopes (beta), second-order AR slopes (lambda), degrees of freedom (nu), standard deviations of error terms (sigma), a seed for
		//the random number generator. OUTPUT: a double number
        double cov_XY (int lag, int TL, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double nu, const Matrix &sigma, unsigned long seed);
        //calculate distance covariance for data generated from a bivarate Gaussian AR(2) process. INPUT: a time lag (lag), a lag truncation (TL), intercepts (alpha),
        //first-order AR slopes (beta), second-order AR slopes (lambda), standard deviations of error terms (sigma), a seed for random number generator.
		//OUTPUT: a double number
        double cov_XY (int lag, int TL, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma, unsigned long seed);
        //calculate the distance variance for data generated from a Student's t AR(2) process. INPUT: a T by 1 data matrix (X), a time lag (lag), a lag truncation (TL),
        //an intercept (alpha), a first-order AR slope (beta), a second-order AR slope (lambda), degrees of freedom (nu),
        //and a standard deviation of the error term (sigma). OUTPUT: a double number
        double var (const Matrix &X, int TL, double alpha, double beta, double lambda, double nu, double sigma, unsigned long seed);
        //calculate the distance variance for data generated from a Gaussian AR(2) process. INPUT: a T by 1 data matrix (X), a time lag (lag), a lag truncation (TL),
        //an intercept (alpha), a first-order AR slope (beta), a second-order AR slope (lambda), a standard deviation of the error term (sigma). OUTPUT: a double number
        double var (const Matrix &X, int TL, double alpha, double beta, double lambda, double sigma, unsigned long seed);
        //calculate the distance variance. INPUT: a T by d_x data matrix (X), a truncation lag (TL), and a bandwidth (bandw). OUTPUT: a double number
        template <double kernel_f (double )>
        double var (const Matrix &X, int TL, double bandw);
        //calculate the fully nonparametric test statistics. INPUT: a lag-smoothing kernel function (kernel_k), a kernel-weight function for
		//conditional moments (kernel_f), a truncation lag (TL), a lag-smoothing parameter (lag_smooth), an integral of the quartic function
		//of kernel_k (kernel_QRSum), and a kernel regression bandwidth (bandw). OUTPUT: a double number
        template <double kernel_k (double ), double kernel_f (double )>
        double do_Test (int TL, int lag_smooth, double kernel_QRSum, double bandw);
        //calculate the test statistic for data generated from a bivarate Gaussian AR(2) process. INPUT: a lag truncation (TL), a lag-smoothing parameter (lag_smooth),
        //value of the integral of the quartic function of a kernel (kernel_QRSum), intercepts (alpha), first-order AR slopes (beta), second-order AR slopes (lambda),
		//standard deviations of the error term (sigma). OUTPUT: a double number
        template <double kernel_k (double )>
        double do_Test (int TL, int lag_smooth, double kernel_QRSum, const Matrix &alpha, const Matrix &beta, const Matrix &lambda,
		                const Matrix &sigma, unsigned long seed);
		//calculate the test statistic for data generated from a bivarate Student's t AR(2) process. INPUT: a lag truncation (TL), a lag-smoothing parameter (lag_smooth),
        //value of the integral of the quartic function of a kernel (kernel_QRSum), intercepts (alpha), first-order AR slopes (beta), second-order AR slopes (lambda),
        //degrees of freedom (nu), standard deviations of the error term (sigma), a seed for the random generator (seed). OUTPUT: a double number
        template <double kernel_k (double )>
        double do_Test (int TL, int lag_smooth, double kernel_QRSum, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double nu,
                        const Matrix &sigma, unsigned long seed);
		//calculate the test statistic when X is generated by a Gaussian AR(2) process and Y is generated by a bivariate Gaussian AR(2) process.
        //INPUT: a truncation lag (TL), a lag-smoothing bandwidth (lag_smooth), the integral of the quartic polynomial of a kernel (kernel_QRSum),
        //a 3x1 vector of intercepts (alpha(1) for X, and alpha(2-3) for Y), 3x1 vectors of AR slopes (beta and lambda), a 3x1 vector of std. deviations of error terms
        //(sigma(1) for X, and sigma(2-3) for the error terms (eta_1 and xi) of Y), a correlation between \eta_1 and \eta_2 (rho) for the Y d.g.p,
        //a seed for the random generator (seed). OUTPUT: a double number
        template <double kernel_k (double )>
        double do_Test (int TL, int lag_smooth, double kernel_QRSum, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma,
                        double rho, unsigned long seed);
		double dcorr (int, double, int, int);//the distance correlation measure defined by eq. (2.3)
		double dcorr (Matrix, Matrix, int, double);
		template<double kernel(double)> //using template
        double bbootstrp_var (double &, int, int, int, double, unsigned long);//overlapping block bootstrap variance
        template<double kernel(double)> //using template
        double nbbootstrp_var (double &, int, int, int, double, unsigned long);//non-overlapping block bootstrap variance
        template<double kernel(double)> //using template
        double t_test (int, int, int, double, unsigned long);//t-test statistic
        double chisq_test (int, double);//Chi-squared statistic
        double bootstrp_chisq_test (int, double, int, unsigned long);//Bootstrap Chi-squared statistic
        double dcorr_mean_x (double);
        double dcorr_mean_x (double, Matrix, Matrix);
        double dcorr_mean_y (double);
        double dcorr_mean_y (double, Matrix, Matrix);
        double dcorr_mean (int, double, int, int);
		double A_1T (double alpha);
		double A_2T (double alpha);
		double A_3T (double alpha);
		//integrate quadratic and quartic functions of a kernel weight
        template <double kernel_k (double )>
        static void integrate_Kernel (double *kernel_QDSum, double *kernel_QRSum);
	protected:
	private:
		int T, d_x, d_y;
		Matrix X, Y;//X is a T by d_x data matrix and Y is a T by d_y data matrix
		double S_1j (int, double, int, int); //cross summations defined by eq. (2.3)
		double S_1j (Matrix, Matrix, int, double);//cross summations defined by eq. (2.3)
		double S_1j (int, double, Matrix, Matrix);//cross summations defined by eq. (2.3) using two different windows
		double S_2j (int, double, int, int); //cross summmations defined by eq. (2.3)
		double S_2j (Matrix, Matrix, int, double);
		double S_2j (int, double, Matrix, Matrix);//cross summations defined by eq. (2.3) using two different windows
		double S_3j (int, double, int, int); //cross summmations defined by eq. (2.3)
		double S_3j (Matrix, Matrix, int, double);
		double S_3j (int, double, Matrix, Matrix);//cross summations defined by eq. (2.3) using two different windows
};

Dist_corr::Dist_corr (const Matrix &_X, const Matrix &_Y) : T(_X.nRow()), d_x(_X.nCol()), d_y(_Y.nCol()) {
	X = _X;
	Y = _Y;
}

double Dist_corr::S_1j (int lag, double alpha, int start, int end)
{
	int t, s, i;
	double dev_x = 0, dev_y = 0, res = 0.;
	if (lag >= 0)
	{
        #pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = start+lag; t <= end; t++)
	    {
		    for (s = start+lag; s <= end; s++)
		    {
                dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
                    #pragma omp atomic
					dev_x += pow(X(t,i) - X(s,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
                    #pragma omp atomic
					dev_y += pow(Y(t-lag,i) - Y(s-lag,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    res += 1/pow(end-start-lag+1, 2.) * dev_x * dev_y;
			}
		}
	}
	else
	{
		#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = start-lag; t <= end; t++)
	    {
		    for (s = start-lag; s <= end; s++)
		    {
			    dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
				    #pragma omp atomic
					dev_x += pow(X(t+lag,i) - X(s+lag,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
			    	#pragma omp atomic
					dev_y += pow(Y(t,i) - Y(s,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    res += 1/pow(end-start+lag+1, 2.) * dev_x * dev_y;
			}
		}
	}
	return res;
}

double Dist_corr::S_1j (Matrix _X, Matrix _Y, int lag, double alpha)
{
	int t, s, i, start = 1, end;
	end = _X.nRow();
	double dev_x = 0, dev_y = 0, res = 0.;
	if (lag >= 0)
	{
        #pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = start+lag; t <= end; t++)
	    {
		    for (s = start+lag; s <= end; s++)
		    {
                dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
                    #pragma omp atomic
					dev_x += pow(_X(t,i) - _X(s,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
                    #pragma omp atomic
					dev_y += pow(_Y(t-lag,i) - _Y(s-lag,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    res += 1/pow(end-start-lag+1, 2.) * dev_x * dev_y;
			}
		}
	}
	else
	{
		#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = start-lag; t <= end; t++)
	    {
		    for (s = start-lag; s <= end; s++)
		    {
			    dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
				    #pragma omp atomic
					dev_x += pow(_X(t+lag,i) - _X(s+lag,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
			    	#pragma omp atomic
					dev_y += pow(_Y(t,i) - _Y(s,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    res += 1/pow(end-start+lag+1, 2.) * dev_x * dev_y;
			}
		}
	}
	return res;
}
double Dist_corr::S_1j (int lag, double alpha, Matrix start, Matrix end)
{
	int t, s, i;
	double dev_x = 0, dev_y = 0, res = 0.;
	if (lag >= 0)
	{
        #pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = (int) start(1)+lag; t <= (int) end(1); t++)
	    {
		    for (s = (int) start(2)+lag; s <= (int) end(2); s++)
		    {
                dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
                    #pragma omp atomic
					dev_x += pow(X(t,i) - X(s,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
                    #pragma omp atomic
					dev_y += pow(Y(t-lag,i) - Y(s-lag,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    res += ((double) 1/((end(1)-start(1)-lag+1) * (end(2)-start(2)-lag+1))) * dev_x * dev_y;
			}
		}
	}
	else
	{
		#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = (int) start(1)-lag; t <= (int) end(1); t++)
	    {
		    for (s = (int) start(2)-lag; s <= (int) end(2); s++)
		    {
			    dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
				    #pragma omp atomic
					dev_x += pow(X(t+lag,i) - X(s+lag,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
			    	#pragma omp atomic
					dev_y += pow(Y(t,i) - Y(s,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    res += ((double) 1/((end(1)-start(1)+lag+1) * (end(2)-start(2)+lag+1))) * dev_x * dev_y;
			}
		}
	}
	return res;
}

double Dist_corr::S_2j (int lag, double alpha, int start, int end)
{
    int t, s, i;
	double dev_x = 0, dev_y = 0, sum_x = 0., sum_y = 0., res = 0.;
	if (lag >= 0)
    {
    	#pragma omp parallel for default(shared) reduction (+:sum_x,sum_y) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = start+lag; t <= end; t++)
	    {
		    for (s = start+lag; s <= end; s++)
		    {
                dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
                    #pragma omp atomic
					dev_x += pow(X(t,i) - X(s,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
                    #pragma omp atomic
					dev_y += pow(Y(t-lag,i) - Y(s-lag,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    sum_x += 1/pow(end-start-lag+1, 2.) * dev_x;
			    sum_y += 1/pow(end-start-lag+1, 2.) * dev_y;
			}
		}
		res = sum_x * sum_y;
	}
	else
	{
		#pragma omp parallel for default(shared) reduction (+:sum_x,sum_y) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = start-lag; t <= end; t++)
	    {
		    for (s = start-lag; s <= end; s++)
		    {
			    dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
				    #pragma omp atomic
					dev_x += pow(X(t+lag,i) - X(s+lag,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
			    	#pragma omp atomic
					dev_y += pow(Y(t,i) - Y(s,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    sum_x += 1/pow(end-start+lag+1, 2.) * dev_x;
			    sum_y += 1/pow(end-start+lag+1, 2.) * dev_y;
			}
		}
		res = sum_x * sum_y;
	}
	return res;
}

double Dist_corr::S_2j (Matrix _X, Matrix _Y, int lag, double alpha)
{
    int t, s, i, start = 1, end;
    end = _X.nRow();
	double dev_x = 0, dev_y = 0, sum_x = 0., sum_y = 0., res = 0.;
	if (lag >= 0)
    {
    	#pragma omp parallel for default(shared) reduction (+:sum_x,sum_y) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = start+lag; t <= end; t++)
	    {
		    for (s = start+lag; s <= end; s++)
		    {
                dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
                    #pragma omp atomic
					dev_x += pow(_X(t,i) - _X(s,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
                    #pragma omp atomic
					dev_y += pow(_Y(t-lag,i) - _Y(s-lag,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    sum_x += 1/pow(end-start-lag+1, 2.) * dev_x;
			    sum_y += 1/pow(end-start-lag+1, 2.) * dev_y;
			}
		}
		res = sum_x * sum_y;
	}
	else
	{
		#pragma omp parallel for default(shared) reduction (+:sum_x,sum_y) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = start-lag; t <= end; t++)
	    {
		    for (s = start-lag; s <= end; s++)
		    {
			    dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
				    #pragma omp atomic
					dev_x += pow(_X(t+lag,i) - _X(s+lag,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
			    	#pragma omp atomic
					dev_y += pow(_Y(t,i) - _Y(s,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    sum_x += 1/pow(end-start+lag+1, 2.) * dev_x;
			    sum_y += 1/pow(end-start+lag+1, 2.) * dev_y;
			}
		}
		res = sum_x * sum_y;
	}
	return res;
}

double Dist_corr::S_2j (int lag, double alpha, Matrix start, Matrix end)
{
    int t, s, i;
	double dev_x = 0, dev_y = 0, sum_x = 0., sum_y = 0., res = 0.;
	if (lag >= 0)
    {
    	#pragma omp parallel for default(shared) reduction (+:sum_x,sum_y) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = (int) start(1)+lag; t <= (int) end(1); t++)
	    {
		    for (s = (int) start(2)+lag; s <= (int) end(2); s++)
		    {
                dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
                    #pragma omp atomic
					dev_x += pow(X(t,i) - X(s,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
                    #pragma omp atomic
					dev_y += pow(Y(t-lag,i) - Y(s-lag,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    sum_x += 1/((end(1)-start(1)-lag+1)*(end(2)-start(2)-lag+1)) * dev_x;
			    sum_y += 1/((end(1)-start(1)-lag+1)*(end(2)-start(2)-lag+1)) * dev_y;
			}
		}
		res = sum_x * sum_y;
	}
	else
	{
		#pragma omp parallel for default(shared) reduction (+:sum_x,sum_y) schedule(guided) private(t,s,i) firstprivate(dev_x,dev_y)
		for (t = (int) start(1)-lag; t <= (int) end(1); t++)
	    {
		    for (s = (int) start(2)-lag; s <= (int) end(2); s++)
		    {
			    dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
				    #pragma omp atomic
					dev_x += pow(X(t+lag,i) - X(s+lag,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
			    	#pragma omp atomic
					dev_y += pow(Y(t,i) - Y(s,i), 2.);
			    }
			    dev_x = pow(dev_x, alpha/2);
			    dev_y = pow(dev_y, alpha/2);
			    sum_x += 1/((end(1)-start(1)+lag+1)*(end(2)-start(2)+lag+1)) * dev_x;
			    sum_y += 1/((end(1)-start(1)+lag+1)*(end(2)-start(2)+lag+1)) * dev_y;
			}
		}
		res = sum_x * sum_y;
	}
	return res;
}

double Dist_corr::S_3j (int lag, double alpha, int start, int end)
{
    int t, s, tau, i;
	double dev_x = 0, dev_y = 0, res = 0.;
	if (lag >= 0)
    {
    	#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,tau,i) firstprivate(dev_x,dev_y)
    	for (t = start+lag; t <= end; t++)
	    {
		    for (s = start+lag; s <= end; s++)
		    {
		    	for (tau = start+lag; tau <= end; tau++)
		    	{
					dev_x = 0.;
			        dev_y = 0.;
			        for (i = 1; i <= d_x; i++)
			        {
						#pragma omp atomic
						dev_x += pow(X(t,i) - X(s,i), 2.);
			        }
			        for (i = 1; i <= d_y; i++)
			        {
						#pragma omp atomic
						dev_y += pow(Y(t-lag,i) - Y(tau-lag,i), 2.);
			        }
			        dev_x = pow(dev_x, alpha/2);
			        dev_y = pow(dev_y, alpha/2);
			        res += 1/pow(end-start-lag+1, 3.) * dev_x * dev_y;
				}
			}
		}
	}
	else
	{
		#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,tau,i) firstprivate(dev_x,dev_y)
		for (t = start-lag; t <= end; t++)
	    {
		    for (s = start-lag; s <= end; s++)
		    {
			    for (tau = start-lag; tau <= end; tau++)
			    {
			    	dev_x = 0.;
			        dev_y = 0.;
			        for (i = 1; i <= d_x; i++)
			        {
				        #pragma omp atomic
						dev_x += pow(X(t+lag,i) - X(s+lag,i), 2.);
			        }
			        for (i = 1; i <= d_y; i++)
			        {
			    	    #pragma omp atomic
						dev_y += pow(Y(t,i) - Y(tau,i), 2.);
			        }
			        dev_x = pow(dev_x, alpha/2);
			        dev_y = pow(dev_y, alpha/2);
			        res += 1/pow(end-start+lag+1, 3.) * dev_x * dev_y;
				}
			}
		}
	}
	return res;
}

double Dist_corr::S_3j (Matrix _X, Matrix _Y, int lag, double alpha)
{
    int t, s, tau, i, start = 1, end;
    end = _X.nRow();
	double dev_x = 0, dev_y = 0, res = 0.;
	if (lag >= 0)
    {
    	#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,tau,i) firstprivate(dev_x,dev_y)
    	for (t = start+lag; t <= end; t++)
	    {
		    for (s = start+lag; s <= end; s++)
		    {
		    	for (tau = start+lag; tau <= end; tau++)
		    	{
					dev_x = 0.;
			        dev_y = 0.;
			        for (i = 1; i <= d_x; i++)
			        {
						#pragma omp atomic
						dev_x += pow(_X(t,i) - _X(s,i), 2.);
			        }
			        for (i = 1; i <= d_y; i++)
			        {
						#pragma omp atomic
						dev_y += pow(_Y(t-lag,i) - _Y(tau-lag,i), 2.);
			        }
			        dev_x = pow(dev_x, alpha/2);
			        dev_y = pow(dev_y, alpha/2);
			        res += 1/pow(end-start-lag+1, 3.) * dev_x * dev_y;
				}
			}
		}
	}
	else
	{
		#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,tau,i) firstprivate(dev_x,dev_y)
		for (t = start-lag; t <= end; t++)
	    {
		    for (s = start-lag; s <= end; s++)
		    {
			    for (tau = start-lag; tau <= end; tau++)
			    {
			    	dev_x = 0.;
			        dev_y = 0.;
			        for (i = 1; i <= d_x; i++)
			        {
				        #pragma omp atomic
						dev_x += pow(_X(t+lag,i) - _X(s+lag,i), 2.);
			        }
			        for (i = 1; i <= d_y; i++)
			        {
			    	    #pragma omp atomic
						dev_y += pow(_Y(t,i) - _Y(tau,i), 2.);
			        }
			        dev_x = pow(dev_x, alpha/2);
			        dev_y = pow(dev_y, alpha/2);
			        res += 1/pow(end-start+lag+1, 3.) * dev_x * dev_y;
				}
			}
		}
	}
	return res;
}

double Dist_corr::S_3j (int lag, double alpha, Matrix start, Matrix end)
{
    int t, s, tau, i;
	double dev_x = 0, dev_y = 0, res = 0.;
	if (lag >= 0)
    {
    	#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,tau,i) firstprivate(dev_x,dev_y)
    	for (t = (int) start(1)+lag; t <= (int) end(1); t++)
	    {
		    for (s = (int) start(2)+lag; s <= (int) end(2); s++)
		    {
		    	for (tau = (int) start(3)+lag; tau <= (int) end(3); tau++)
		    	{
					dev_x = 0.;
			        dev_y = 0.;
			        for (i = 1; i <= d_x; i++)
			        {
						#pragma omp atomic
						dev_x += pow(X(t,i) - X(s,i), 2.);
			        }
			        for (i = 1; i <= d_y; i++)
			        {
						#pragma omp atomic
						dev_y += pow(Y(t-lag,i) - Y(tau-lag,i), 2.);
			        }
			        dev_x = pow(dev_x, alpha/2);
			        dev_y = pow(dev_y, alpha/2);
			        res += 1/((end(1)-start(1)-lag+1)*(end(2)-start(2)-lag+1)*(end(3)-start(3)-lag+1)) * dev_x * dev_y;
				}
			}
		}
	}
	else
	{
		#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(t,s,tau,i) firstprivate(dev_x,dev_y)
		for (t = (int) start(1)-lag; t <= (int) end(1); t++)
	    {
		    for (s = (int) start(2)-lag; s <= (int) end(2); s++)
		    {
			    for (tau = (int) start(3)-lag; tau <= (int) end(3); tau++)
			    {
			    	dev_x = 0.;
			        dev_y = 0.;
			        for (i = 1; i <= d_x; i++)
			        {
				        #pragma omp atomic
						dev_x += pow(X(t+lag,i) - X(s+lag,i), 2.);
			        }
			        for (i = 1; i <= d_y; i++)
			        {
			    	    #pragma omp atomic
						dev_y += pow(Y(t,i) - Y(tau,i), 2.);
			        }
			        dev_x = pow(dev_x, alpha/2);
			        dev_y = pow(dev_y, alpha/2);
			        res +=  1/((end(1)-start(1)+lag+1)*(end(2)-start(2)+lag+1)*(end(3)-start(3)+lag+1)) * dev_x * dev_y;
				}
			}
		}
	}
	return res;
}

double Dist_corr::A_1T (double alpha)
{
	int t, k, i;
	double dev_x = 0., dev_y = 0., sum_x = 0., sum_y = 0., res = 0.;
	for (k = 2; k <= T-1; k++)
	{
		sum_x = 0.;
		sum_y = 0.;
		for (t = 1; t <= T-k; t++)
		{
			dev_x = 0.;
			dev_y = 0.;
			for (i = 1; i <= d_x; i++)
			{
				dev_x += pow(X(t,i) - X(t+k,i), 2.);
			}
			for (i = 1; i <= d_y; i++)
			{
				dev_y += pow(Y(t,i) - Y(t+k,i), 2.);
			}
			sum_x += ((double) 1/(T-k)) * pow(dev_x, alpha/2);
			sum_y += ((double) 1/(T-k)) * pow(dev_y, alpha/2);
		}
		res += 2*(1-((double) (k-1)/T)) * sum_x * sum_y;
	}
	return res;
}

double Dist_corr::A_2T (double alpha)
{
	int t, k, i;
	double dev_x = 0., dev_y = 0., sum_x = 0., sum_y = 0., sum_xx = 0., sum_yy = 0.;
	for (k = 2; k <= T-1; k++)
	{
		sum_x = 0.;
		sum_y = 0.;
		for (t = 1; t <= T-k; t++)
		{
			dev_x = 0.;
			dev_y = 0.;
			for (i = 1; i <= d_x; i++)
			{
				dev_x += pow(X(t,i) - X(t+k,i), 2.);
			}
			for (i = 1; i <= d_y; i++)
			{
				dev_y += pow(Y(t,i) - Y(t+k,i), 2.);
			}
			sum_x += ((double) 1/(T-k)) * pow(dev_x, alpha/2);
			sum_y += ((double) 1/(T-k)) * pow(dev_y, alpha/2);
		}
		sum_xx += 2 * (1-((double) (k-1)/T)) * sum_x;
		sum_yy += 2 * (1-((double) (k-1)/T)) * sum_y;
	}
	return ((double) 1/T) * sum_xx * sum_yy;
}

double Dist_corr::A_3T (double alpha)
{
	int t, k, i, ell;
	double dev_x = 0., dev_y = 0., sum_x = 0., sum_y = 0., res = 0.;
	#pragma omp parallel for default(shared) reduction (+:res) schedule(guided) private(k,ell,t,i) firstprivate(dev_x,dev_y,sum_x,sum_y)
	for (k = 2; k <= T-1; k++)
	{
		for (ell = 2; ell <= T-1; ell++)
		{
		    sum_x = 0.;
		    sum_y = 0.;
			for (t = 1; t <= T-std::max(k,ell); t++)
		    {
			    dev_x = 0.;
			    dev_y = 0.;
			    for (i = 1; i <= d_x; i++)
			    {
				    #pragma omp atomic
					dev_x += pow(X(t,i) - X(t+k,i), 2.);
			    }
			    for (i = 1; i <= d_y; i++)
			    {
				    #pragma omp atomic
					dev_y += pow(Y(t,i) - Y(t+ell,i), 2.);
			    }
			    #pragma omp atomic
				sum_x += ((double) 1/(T-std::max(k,ell))) * pow(dev_x, alpha/2);
			    #pragma omp atomic
				sum_y += ((double) 1/(T-std::max(k,ell))) * pow(dev_y, alpha/2);
		    }
		    res += 4 *((double) 1/T) * (1-std::max((double) (k-1)/T, (double) (ell-1)/T))* sum_x * sum_y;
		}
	}
	return res;
}

template <double kernel (double)> //using template
double Dist_corr::bbootstrp_var (double &bootstrp_mean, int K, int bsize, int M, double alpha, unsigned long seed)
//K is number of random blocks; bsize is block size; M is bandwidth
{
	int i, j, index = 0;
	double ker_val = 0., tmp = 0.;
	Matrix B(K,1);
	gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    B.set(0.);
    #pragma omp parallel for default(shared) schedule(guided) private(i,j) firstprivate(index,ker_val,tmp)
	for (i = 1; i <= K; i++)
	{
		index = gsl_rng_uniform_int (r, T-bsize+1) + 1;//generate a Uniform random number in [1,T-bsize+1]
		for (j = 1-bsize; j <= bsize-1; j++)
		{
			ker_val = kernel ((double) j/M);
			if (ker_val != 0)
			{
				tmp = bsize * pow(ker_val, 2.) * dcorr(j, alpha, index, index+bsize-1);
				if (j >= 0)
		            tmp -= ((double) bsize/(bsize-j)) * pow(ker_val, 2.) * Dist_corr::dcorr_mean (j, alpha, index, index+bsize-1);
		        else
		            tmp -= ((double) bsize/(bsize+j)) * pow(ker_val, 2.) * Dist_corr::dcorr_mean (j, alpha, index, index+bsize-1);
		        #pragma omp atomic
		        B(i) += tmp;
			    //B(i) += bsize * pow(ker_val, 2.) * dcorr(j, alpha, index, index+bsize-1);
			}
		}
	}
	bootstrp_mean = mean_u (B);
	gsl_rng_free (r);
	return variance (B);
}

template <double kernel (double)> //using template
double Dist_corr::nbbootstrp_var (double &bootstrp_mean, int K, int bsize, int M, double alpha, unsigned long seed)
//K is number of draws; bsize is block size; M is bandwidth
{
	int nblocks, t, i = 1, j1, j2, jj, ell, index = 1, T_bt;
	nblocks = std::floor((double) T/bsize);
	T_bt = nblocks * bsize;
	gsl_rng *r;
    const gsl_rng_type *gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    double ker_val = 0.;
    Matrix B(K,1), X_bt(T_bt,d_x), Y_bt(T_bt,d_y);
    B.set(0.);
    //#pragma omp parallel for default(shared) schedule(guided) private(i,ell,t,j1,j2,jj) firstprivate(index,ker_val,X_bt,Y_bt)
	for (i = 1; i <= K; i++)
	{
		for (ell = 1; ell <= nblocks; ell++)
		{
		    index = gsl_rng_uniform_int (r, nblocks) + 1;//generate a Uniform random number in [1,nblocks]
		    //cout << "index = " << index << endl;
		    for (t = 1; t <= bsize; t++)
		    {
			    for (j1 = 1; j1 <= d_y; j1++)
			    {
				    Y_bt((ell-1)*bsize + t, j1) = Y((index-1)*bsize+t, j1);
			    }
			    for (j2 = 1; j2 <= d_x; j2++)
			    {
			    	X_bt((ell-1)*bsize + t, j2) = X((index-1)*bsize+t, j2);
				}
		    }
	    }
		for (jj = 1-T_bt; jj <= T_bt-1; jj++)
		{
			ker_val = kernel ((double) jj/M);
			if (ker_val != 0)
			{
			    //#pragma omp atomic
			    B(i) += T_bt * pow(ker_val, 2.) * dcorr(X_bt, Y_bt, jj, alpha);
			}
		}
	}
	bootstrp_mean = mean_u (B);
	gsl_rng_free (r);
	return variance(B);
}


double Dist_corr::dcorr (int lag, double alpha, int start, int end)
{
	return Dist_corr::S_1j(lag, alpha, start, end) + Dist_corr::S_2j(lag, alpha, start, end) - 2*Dist_corr::S_3j(lag, alpha, start, end);
}

double Dist_corr::dcorr (Matrix _X, Matrix _Y, int lag, double alpha)
{
	return Dist_corr::S_1j(_X, _Y, lag, alpha) + Dist_corr::S_2j(_X, _Y, lag, alpha) - 2*Dist_corr::S_3j(_X, _Y, lag, alpha);
}

double Dist_corr::dcorr_mean_x (double alpha)
{
    int t, s, i;
    double diff_x = 0., res = 0.;
    #pragma omp parallel for default(shared) reduction(+:res)schedule(guided) private(t,s,i) firstprivate(diff_x)
    for (t = 1; t <= T; t++)
    {
        for (s = 1; s <= T; s++)
        {
            diff_x = 0.;
            for (i = 1; i <= d_x; i++)
            {
                #pragma omp atomic
				diff_x += pow(X(t,i) - X(s,i), 2.);
            }
            res += ((double) 1/pow(T, 2.)) * pow(diff_x, alpha/2);
        }
    }
    return res;
}

double Dist_corr::dcorr_mean_x (double alpha, Matrix start, Matrix end)
{
    int t, s, i;
    double diff_x = 0., res = 0.;
    #pragma omp parallel for default(shared) reduction(+:res)schedule(guided) private(t,s,i) firstprivate(diff_x)
    for (t = (int) start(1) ; t <= (int) end(1); t++)
    {
        for (s = (int) start(2); s <= (int) end(2); s++)
        {
            diff_x = 0.;
            for (i = 1; i <= d_x; i++)
            {
                #pragma omp atomic
				diff_x += pow(X(t,i) - X(s,i), 2.);
            }
            res += 1/((end(1)-start(1)+1)*(end(2)-start(2)+1)) * pow(diff_x, alpha/2);
        }
    }
    return res;
}

double Dist_corr::dcorr_mean_y (double alpha)
{
    int t, s, i;
    double diff_y = 0., res = 0.;
    #pragma omp parallel for default(shared) reduction(+:res)schedule(guided) private(t,s,i) firstprivate(diff_y)
    for (t = 1; t <= T; t++)
    {
        for (s = 1; s <= T; s++)
        {
            diff_y = 0.;
            for (i = 1; i <= d_y; i++)
            {
                #pragma omp atomic
				diff_y += pow(Y(t,i) - Y(s,i), 2.);
            }
            res += ((double) 1/pow(T, 2.)) * pow(diff_y, alpha/2);
        }
    }
    return res;
}

double Dist_corr::dcorr_mean_y (double alpha, Matrix start, Matrix end)
{
    int t, s, i;
    double diff_y = 0., res = 0.;
    #pragma omp parallel for default(shared) reduction(+:res)schedule(guided) private(t,s,i) firstprivate(diff_y)
    for (t = (int) start(1); t <= (int) end(1); t++)
    {
        for (s = (int) start(2); s <= (int) end(2); s++)
        {
            diff_y = 0.;
            for (i = 1; i <= d_y; i++)
            {
                #pragma omp atomic
				diff_y += pow(Y(t,i) - Y(s,i), 2.);
            }
            res += 1/((end(1)-start(1)+1)*(end(2)-start(2)+1)) * pow(diff_y, alpha/2);
        }
    }
    return res;
}

double Dist_corr::dcorr_mean (int lag, double alpha, int start, int end)
{
    double diff_x = 0., diff_y = 0., av_x = 0., av_y = 0., mean_x, mean_y, res = 0.;
    int tau, t, i;
    mean_x = Dist_corr::dcorr_mean_x (alpha);
    mean_y = Dist_corr::dcorr_mean_y (alpha);
    res = mean_x * mean_y;
    if (lag >= 0)
    {
        #pragma omp parallel for default(shared) reduction(+:res)schedule(guided) private(tau,t,i) firstprivate(av_x,av_y,diff_x,diff_y)
		for (tau = 1; tau <= end-start-lag; tau++)
        {
		    av_x = 0.;
            av_y = 0.;
            for (t = start; t <= end-tau; t++)
            {
                diff_x = 0.;
                diff_y = 0.;
                for (i = 1; i <= d_x; i++)
                {
                    #pragma omp atomic
					diff_x += pow(X(t,i) - X(t+tau,i), 2.);
                }
                for (i = 1; i <= d_y; i++)
                {
                    #pragma omp atomic
					diff_y += pow(Y(t,i) - Y(t+tau,i), 2.);
                }
                av_x += ((double) 1/(end-start-tau+1)) * pow(diff_x, alpha/2);
                av_y += ((double) 1/(end-start-tau+1)) * pow(diff_y, alpha/2);
            }
            res += 2*(1 - (double) tau/(end-start-lag+1)) * (mean_x - av_x) * (mean_y - av_y);
        }
    }
    else
    {
    	#pragma omp parallel for default(shared) reduction(+:res)schedule(guided) private(tau,t,i) firstprivate(av_x,av_y,diff_x,diff_y)
		for (tau = 1; tau <= end-start+lag; tau++)
        {
		    av_x = 0.;
            av_y = 0.;
            for (t = start; t <= end-tau; t++)
            {
                diff_x = 0.;
                diff_y = 0.;
                for (i = 1; i <= d_x; i++)
                {
                    #pragma omp atomic
					diff_x += pow(X(t,i) - X(t+tau,i), 2.);
                }
                for (i = 1; i <= d_y; i++)
                {
                    #pragma omp atomic
					diff_y += pow(Y(t,i) - Y(t+tau,i), 2.);
                }
                av_x += ((double) 1/(end-start-tau+1)) * pow(diff_x, alpha/2);
                av_y += ((double) 1/(end-start-tau+1)) * pow(diff_y, alpha/2);
            }
            res += 2*(1 - (double) tau/(end-start+lag+1)) * (mean_x - av_x) * (mean_y - av_y);
        }
	}
    return res;
}

double Dist_corr::chisq_test (int M, double alpha)
//M is the maximum number of lags; alpha is the exponent of the distance correlation
{
	int lag;
	double res = 0.;
	for (lag = -M; lag <= M; lag++)
	{
		 res += T * Dist_corr::dcorr (lag, alpha, 1, T)/Dist_corr::dcorr_mean (lag, alpha, 1, T);
	}
	return res;
}

double Dist_corr::bootstrp_chisq_test (int M, double alpha, int bsize, unsigned long seed)
//M is the maximum number of lags; bsize is the block size; alpha is the exponent of the distance correlation
{
	int i, lag, K, start, end;
	double res = 0.;
	Matrix num(2*M+1,1), dem(2*M+1,1);
	gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
    K = floor ((double) T/bsize);
    num.set(0.);
    dem.set(0.);
    //generate K possibly over-lapping blocks of size b
    for (i = 1; i <= K; i++)
    {
		start = gsl_rng_uniform_int (r, T-bsize+1) + 1;//generate a Uniform random number in [1, T-bsize+1]
    	end = start + bsize - 1;
    	for (lag = -M; lag <= M; lag++)
    	{
    	    num(lag+M+1) += ((double) 1/K) * Dist_corr::dcorr (lag, alpha, start, end);
		    dem(lag+M+1) += ((double) 1/K) * Dist_corr::dcorr_mean (lag, alpha, start, end);
		}
	}
	for (i = 1; i <= 2*M+1; i++)
	{
		res += bsize * num(i)/dem(i);
	}
    gsl_rng_free (r);
    return res;
}

template <double kernel (double)> //using template
double Dist_corr::t_test (int K, int bsize, int M, double alpha, unsigned long seed)
//K is the number of bootstrap repetitions
//M is a kernel bandwidth (1 < M < bsize)
//alpha is the exponent of the correlation distance
{
	int j;
	double sum1 = 0., sum2 = 0., mean = 0., ker_val = 0., var = 1.;
	var = Dist_corr::bbootstrp_var <kernel> (mean, K, bsize, M, alpha, seed);
	//cout << var << endl;
	//#pragma omp parallel for default(shared) reduction(+:sum1,sum2) schedule(guided) private(j) firstprivate(ker_val)
	for (j = 1-T; j <= T-1; j++)
	{
		ker_val = kernel ((double) j/M);
		if (ker_val != 0)
		{
		    sum1 += T * pow(ker_val, 2.) * Dist_corr::dcorr (j, alpha, 1, T);
		    if (j >= 0)
		        sum2 += ((double) T/(T-j)) * pow(ker_val, 2.) * Dist_corr::dcorr_mean (j, alpha, 1, T);
		    else
		        sum2 += ((double) T/(T+j)) * pow(ker_val, 2.) * Dist_corr::dcorr_mean (j, alpha, 1, T);
		}
	}
	return (sum1-sum2)/pow(var, 0.5);
}

//calculate the distance covariance. INPUT: X and Y are T by d_x data matrices, a time lag (lag), a lag truncation (TL), and a bandwidth (bandw)
//OUTPUT: the value of the distance covariance
template <double kernel_f (double )>
double Dist_corr::cov_XY (int lag, int TL, double bandw) {
	ASSERT_ (d_x == d_y);
	Matrix xlag_t(TL, d_x), xlag_s(TL, d_x), x_t(1, d_x), x_s(1, d_x), ylag_t(TL, d_x), ylag_s(TL, d_x), y_t(1, d_x), y_s(1, d_x);
	Matrix ylag_tau(TL, d_x), y_tau(1, d_x);
	int t = 1, s = 1, i = 1, j = 1;
	double temp1 = 0., temp2 = 0., sum1 = 0., sum3 = 0., sum4 = 0.;
	if (lag >= 0) {
		//#pragma omp parallel for default(shared) reduction(+:sum1,sum2,sum3,sum4) schedule(dynamic,CHUNK) private(t,s,tau,i,j) firstprivate(xlag_t,ylag_t,x_t,y_t,xlag_s,ylag_s,x_s,y_s,ylag_tau,y_tau,temp1,temp2)
		#pragma omp parallel for default(shared) reduction(+:sum1,sum3,sum4) schedule(dynamic,CHUNK) private(t,s,i,j) firstprivate(xlag_t,ylag_t,x_t,y_t,xlag_s,ylag_s,x_s,y_s,temp1,temp2)
		for (t = lag+TL+1; t <= T; t++) {
			for (i = 1; i <= TL; i++) {
				for (j = 1; j <= d_x; j++) {
					xlag_t(i,j) = X(t-i, j);
					ylag_t(i,j) = Y(t-i-lag, j);
				}
			}
			for (j = 1; j <= d_x; j++) {
				x_t(1,j) = X(t, j);
				y_t(1,j) = Y(t-lag, j);
			}
			for (s = lag+TL+1; s <= T; s++) {
				for (i = 1; i <= TL; i++) {
					for (j = 1; j <= d_x; j++) {
						xlag_s(i,j) = X(s-i, j);
						ylag_s(i,j) = Y(s-i-lag, j);
					}
				}
				for (j = 1; j <= d_x; j++) {
					x_s(1,j) = X(s, j);
					y_s(1,j) = Y(s-lag, j);
				}
				temp1 = NReg::var_Ux_ts <kernel_f> (X, xlag_t, xlag_s, x_t, x_s, TL, bandw);
				temp2 = NReg::var_Ux_ts <kernel_f> (Y, ylag_t, ylag_s, y_t, y_s, TL, bandw);
				//cout << "(temp1, temp2) = " << temp1 << ", " << temp2 << endl;
				/*for (tau = lag+TL+1; tau <= T; tau++) {
					for (i = 1; i <= TL; i++) {
						for (j = 1; j <= d_x; j++) {
							ylag_tau(i,j) = Y(tau-i-lag, j);
						}
					}
					for (j = 1; j <= d_x; j++) {
						y_tau(1,j) = Y(tau-lag, j);
					}
					sum2 += temp1 * NReg::var_Ux_ts <kernel_f> (Y, ylag_tau, ylag_s, y_tau, y_s, TL, bandw);
					cout << "count = " << ++count << endl;
				}*/
				sum1 += temp1 * temp2;
				sum3 += temp1;
				sum4 += temp2;
			}
		}
		//return ((double) 1/pow(T-lag-TL, 2.)) * sum1 - ((double) 2/pow(T-lag-TL, 3.)) * sum2 + ((double) 1/pow(T-lag-TL, 4.)) * sum3 * sum4;
		return ((double) 1/pow(T-TL-lag, 2.)) * sum1 + ((double) 1/pow(T-TL-lag, 4.)) * sum3 * sum4;
	}
	else {
		//#pragma omp parallel for default(shared) reduction(+:sum1,sum2,sum3,sum4) schedule(dynamic,CHUNK) private(t,s,tau,i,j) firstprivate(xlag_t,ylag_t,x_t,y_t,xlag_s,ylag_s,x_s,y_s,ylag_tau,y_tau,temp1,temp2)
		#pragma omp parallel for default(shared) reduction(+:sum1,sum3,sum4) schedule(dynamic,CHUNK) private(t,s,i,j) firstprivate(xlag_t,ylag_t,x_t,y_t,xlag_s,ylag_s,x_s,y_s,temp1,temp2)
		for (t = TL+1-lag; t <= T; t++) {
			for (i = 1; i <= TL; i++) {
				for (j = 1; j <= d_x; j++) {
					xlag_t(i,j) = X(t-i+lag, j);
					ylag_t(i,j) = Y(t-i, j);
				}
			}
			for (j = 1; j <= d_x; j++) {
				x_t(1,j) = X(t+lag, j);
				y_t(1,j) = Y(t, j);
			}
			for (s = TL+1-lag; s <= T; s++) {
				for (i = 1; i <= TL; i++) {
					for (j = 1; j <= d_x; j++) {
						xlag_s(i,j) = X(s-i+lag, j);
						ylag_s(i,j) = Y(s-i, j);
					}
				}
				for (j = 1; j <= d_x; j++) {
					x_s(1,j) = X(s+lag, j);
					y_s(1,j) = Y(s, j);
				}
				temp1 = NReg::var_Ux_ts <kernel_f> (X, xlag_t, xlag_s, x_t, x_s, TL, bandw);
				temp2 = NReg::var_Ux_ts <kernel_f> (Y, ylag_t, ylag_s, y_t, y_s, TL, bandw);
				//cout << "(temp1, temp2) = " << temp1 << ", " << temp2 << endl;
				/*for (tau = TL+1-lag; tau <= T; tau++) {
					for (i = 1; i <= TL; i++) {
						for (j = 1; j <= d_x; j++) {
							ylag_tau(i,j) = Y(tau-i, j);
						}
					}
					for (j = 1; j <= d_x; j++) {
						y_tau(1,j) = Y(tau, j);
					}
					sum2 += temp1 * NReg::var_Ux_ts <kernel_f> (Y, ylag_t, ylag_tau, y_t, y_tau, TL, bandw);
					cout << "count = " << ++count << endl;
				}*/
				sum1 += temp1 * temp2;
				sum3 += temp1;
				sum4 += temp2;
			}
		}
		//return ((double) 1/pow(T+lag-TL, 2.)) * sum1 - ((double) 2/pow(T+lag-TL, 3.)) * sum2 + ((double) 1/pow(T+lag-TL, 4.)) * sum3 * sum4;
		return ((double) 1/pow(T-TL+lag, 2.)) * sum1 + ((double) 1/pow(T-TL+lag, 4.)) * sum3 * sum4;
	}
}

//calculate distance covariance for data generated from a bivarate Gaussian AR(2) process. INPUT: a time lag (lag), a lag truncation (TL), intercepts (alpha),
//first-order AR slopes (beta), second-order AR slopes (lambda), standard deviations of error terms (sigma). OUTPUT: a double number
double Dist_corr::cov_XY (int lag, int TL, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma, unsigned long seed) {
	ASSERT_ ((d_x == 1) && (d_y == 1));
	//int t = 1, s = 1, tau = 1;
	int t = 1, s = 1;
	double temp1 = 0., temp2 = 0., sum1 = 0., sum3 = 0., sum4 = 0.;
	if (lag >= 0) {
		//#pragma omp parallel for default(shared) reduction(+:sum1,sum2,sum3,sum4) schedule(dynamic,CHUNK) private(t,s,tau) firstprivate(temp1,temp2)
		#pragma omp parallel for default(shared) reduction(+:sum1,sum3,sum4) schedule(dynamic,CHUNK) private(t,s) firstprivate(temp1,temp2)
		for (t = lag+TL+1; t <= T; t++) {
			for (s = lag+TL+1; s <= T; s++) {
				temp1 = GReg::var_U_ts (X(t), X(s), X(t-1), X(t-2), X(s-1), X(s-2), alpha(1), beta(1), lambda(1), sigma(1), seed);
				temp2 = GReg::var_U_ts (Y(t-lag), Y(s-lag), Y(t-lag-1), Y(t-lag-2), Y(s-lag-1), Y(s-lag-2), alpha(2), beta(2), lambda(2), sigma(2), seed);
				sum1 += temp1 * temp2;
				sum3 += temp1;
				sum4 += temp2;
				/*for (tau = lag+TL+1; tau <= T; tau++) {
					sum2 += 2 * temp1 * GReg::var_U_ts (Y(tau-lag), Y(s-lag), Y(tau-lag-1), Y(tau-lag-2), Y(s-lag-1), Y(s-lag-2), alpha(2), beta(2), lambda(2), sigma(2));
				}*/
			}
		}
		//return ((double) 1/pow(T-TL-lag, 2.)) * sum1 - ((double) 1/pow(T-TL-lag, 3.)) * sum2 + ((double) 1/pow(T-TL-lag, 4.)) * sum3 * sum4;
		return ((double) 1/pow(T-TL-lag, 2.)) * sum1 + ((double) 1/pow(T-TL-lag, 4.)) * sum3 * sum4;
	}
	else {
		//#pragma omp parallel for default(shared) reduction(+:sum1,sum2,sum3,sum4) schedule(dynamic,CHUNK) private(t,s,tau) firstprivate(temp1,temp2)
		#pragma omp parallel for default(shared) reduction(+:sum1,sum3,sum4) schedule(dynamic,CHUNK) private(t,s) firstprivate(temp1,temp2)
		for (t = TL+1-lag; t <= T; t++) {
			for (s = TL+1-lag; s <= T; s++) {
				temp1 = GReg::var_U_ts (X(t+lag), X(s+lag), X(t+lag-1), X(t+lag-2), X(s+lag-1), X(s+lag-2), alpha(1), beta(1), lambda(1), sigma(1), seed);
				temp2 = GReg::var_U_ts (Y(t), Y(s), Y(t-1), Y(t-2), Y(s-1), Y(s-2), alpha(2), beta(2), lambda(2), sigma(2), seed);
				sum1 += temp1 * temp2;
				sum3 += temp1;
				sum4 += temp2;
				/*for (tau = TL+1-lag; tau <= T; tau++) {
					sum2 += 2 * temp1 * GReg::var_U_ts (Y(t), Y(tau), Y(t-1), Y(t-2), Y(tau-1), Y(tau-2), alpha(2), beta(2), lambda(2), sigma(2));
				}*/
			}
		}
		//return ((double) 1/pow(T-TL+lag, 2.)) * sum1 - ((double) 1/pow(T-TL+lag, 3.)) * sum2 + ((double) 1/pow(T-TL+lag, 4.)) * sum3 * sum4;
		return ((double) 1/pow(T-TL+lag, 2.)) * sum1 + ((double) 1/pow(T-TL+lag, 4.)) * sum3 * sum4;
	}
}

//calculate distance covariance for data generated from a bivarate Student's t AR(2) process. INPUT: a time lag (lag), a lag truncation (TL), intercepts (alpha),
//first-order AR slopes (beta), second-order AR slopes (lambda), degrees of freedom (nu), standard deviations of error terms (sigma). OUTPUT: a double number
double Dist_corr::cov_XY (int lag, int TL, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double nu, const Matrix &sigma, unsigned long seed) {
	ASSERT_ ((d_x == 1) && (d_y == 1));
	//int t = 1, s = 1, tau = 1;
	int t = 1, s = 1;
	double temp1 = 0., temp2 = 0., sum1 = 0., sum3 = 0., sum4 = 0.;
	if (lag >= 0) {
		//#pragma omp parallel for default(shared) reduction(+:sum1,sum2,sum3,sum4) schedule(dynamic,CHUNK) private(t,s,tau) firstprivate(temp1,temp2)
		#pragma omp parallel for default(shared) reduction(+:sum1,sum3,sum4) schedule(dynamic,CHUNK) private(t,s) firstprivate(temp1,temp2)
		for (t = lag+TL+1; t <= T; t++) {
			for (s = lag+TL+1; s <= T; s++) {
				temp1 = TReg::var_U_ts (X(t), X(s), X(t-1), X(t-2), X(s-1), X(s-2), alpha(1), beta(1), lambda(1), nu, sigma(1), seed);
				temp2 = TReg::var_U_ts (Y(t-lag), Y(s-lag), Y(t-lag-1), Y(t-lag-2), Y(s-lag-1), Y(s-lag-2), alpha(2), beta(2), lambda(2), nu, sigma(2), seed);
				sum1 += temp1 * temp2;
				sum3 += temp1;
				sum4 += temp2;
				/*for (tau = lag+TL+1; tau <= T; tau++) {
					sum2 += 2 * temp1 * GReg::var_U_ts (Y(tau-lag), Y(s-lag), Y(tau-lag-1), Y(tau-lag-2), Y(s-lag-1), Y(s-lag-2), alpha(2), beta(2), lambda(2), sigma(2));
				}*/
			}
		}
		//return ((double) 1/pow(T-TL-lag, 2.)) * sum1 - ((double) 1/pow(T-TL-lag, 3.)) * sum2 + ((double) 1/pow(T-TL-lag, 4.)) * sum3 * sum4;
		return ((double) 1/pow(T-TL-lag, 2.)) * sum1 + ((double) 1/pow(T-TL-lag, 4.)) * sum3 * sum4;
	}
	else {
		//#pragma omp parallel for default(shared) reduction(+:sum1,sum2,sum3,sum4) schedule(dynamic,CHUNK) private(t,s,tau) firstprivate(temp1,temp2)
		#pragma omp parallel for default(shared) reduction(+:sum1,sum3,sum4) schedule(dynamic,CHUNK) private(t,s) firstprivate(temp1,temp2)
		for (t = TL+1-lag; t <= T; t++) {
			for (s = TL+1-lag; s <= T; s++) {
				temp1 = TReg::var_U_ts (X(t+lag), X(s+lag), X(t+lag-1), X(t+lag-2), X(s+lag-1), X(s+lag-2), alpha(1), beta(1), lambda(1), nu, sigma(1), seed);
				temp2 = TReg::var_U_ts (Y(t), Y(s), Y(t-1), Y(t-2), Y(s-1), Y(s-2), alpha(2), beta(2), lambda(2), nu, sigma(2), seed);
				sum1 += temp1 * temp2;
				sum3 += temp1;
				sum4 += temp2;
				/*for (tau = TL+1-lag; tau <= T; tau++) {
					sum2 += 2 * temp1 * GReg::var_U_ts (Y(t), Y(tau), Y(t-1), Y(t-2), Y(tau-1), Y(tau-2), alpha(2), beta(2), lambda(2), sigma(2));
				}*/
			}
		}
		//return ((double) 1/pow(T-TL+lag, 2.)) * sum1 - ((double) 1/pow(T-TL+lag, 3.)) * sum2 + ((double) 1/pow(T-TL+lag, 4.)) * sum3 * sum4;
		return ((double) 1/pow(T-TL+lag, 2.)) * sum1 + ((double) 1/pow(T-TL+lag, 4.)) * sum3 * sum4;
	}
}

//calculate the distance variance. INPUT: a T by d_x data matrix (X), a truncation lag (TL), and a bandwidth (bandw). OUTPUT: a double number
template <double kernel_f (double )>
double Dist_corr::var (const Matrix &X, int TL, double bandw) {
	int T = 1, d_x = 1, t = 1, s = 1, i = 1, j = 1;
	T = X.nRow();
	d_x = X.nCol();
	double sum = 0.;
	Matrix x_t(1, d_x), xlag_t(TL, d_x), xlag_s(TL, d_x);
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) private(t,s,i,j) firstprivate(xlag_t,xlag_s,x_t)
	for (t = TL+1; t <= T; t++) {
		for (i = 1; i <= TL; i++) {
			for (j = 1; j <= d_x; j++) {
				xlag_t(i,j) = X(t-i, j);
			}
		}
		for (j = 1; j <= d_x; j++) {
			x_t(1,j) = X(t, j);
		}
		for (s = TL+1; s <= T; s++) {
			for (i = 1; i <= TL; i++) {
				for (j = 1; j <= d_x; j++) {
					xlag_s(i,j) = X(s-i, j);
				}
			}
			sum += 2 * NReg::reg_F <kernel_f> (X, xlag_s, x_t, TL, bandw) - NReg::reg_BF <kernel_f> (X, xlag_t, xlag_s, TL, bandw);
		}
	}
	return ((double) 1/pow(T-TL, 2.)) * sum;
}

//calculate the distance variance for data generated from a Student's t AR(2) process. INPUT: a T by 1 data matrix (X), a time lag (lag), a lag truncation (TL),
//an intercept (alpha), a first-order AR slope (beta), a second-order AR slope (lambda), degrees of freedom (nu),
//and a standard deviation of the error term (sigma). OUTPUT: a double number
double Dist_corr::var (const Matrix &X, int TL, double alpha, double beta, double lambda, double nu, double sigma, unsigned long seed) {
	int T = 1, d_x = 1;
	T = X.nRow();
	d_x = X.nCol();
	ASSERT_ (d_x == 1);
	int t = 1, s = 1;
	double temp1 = 0., temp2 = 0., sum = 0.;
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) private(t,s) firstprivate(temp1,temp2)
	for (t = TL+1; t <= T; t++) {
		for (s = TL+1; s <= T; s++) {
			temp1 = TReg::reg_F (X(t-1), X(t-2), X(s), alpha, beta, lambda, nu, sigma);
			temp2 = TReg::reg_BF (X(t-1), X(t-2), X(s-1), X(s-2), alpha, beta, lambda, nu, sigma, seed);
			sum += 2 * temp1 - temp2;
		}
	}
	return ((double) 1/pow(T-TL, 2.)) * sum;
}

//calculate the distance variance for data generated from a Gaussian AR(2) process. INPUT: a T by 1 data matrix (X), a time lag (lag), a lag truncation (TL),
//an intercept (alpha), a first-order AR slope (beta), a second-order AR slope (lambda), a standard deviation of the error term (sigma). OUTPUT: a double number
double Dist_corr::var (const Matrix &X, int TL, double alpha, double beta, double lambda, double sigma, unsigned long seed) {
	int T = 1, d_x = 1;
	T = X.nRow();
	d_x = X.nCol();
	ASSERT_ (d_x == 1);
	int t = 1, s = 1;
	double sum = 0.;
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) private(t,s)
	for (t = TL+1; t <= T; t++) {
		for (s = TL+1; s <= T; s++) {
			sum +=  GReg::reg_F (X(s-1), X(s-2), X(t), alpha, beta, lambda, sigma) + GReg::reg_F (X(t-1), X(t-2), X(s), alpha, beta, lambda, sigma)
				    - GReg::reg_BF (X(t-1), X(t-2), X(s-1), X(s-2), alpha, beta, lambda, sigma, seed);
		}
	}
	return ((double) 1/pow(T-TL, 2.)) * sum;
}

//integrate quadratic and quartic functions of a kernel weight
template <double kernel_k (double )>
void Dist_corr::integrate_Kernel (double *kernel_QDSum, double *kernel_QRSum) {
	double x = 0.;
	*kernel_QDSum = 0.;
	*kernel_QRSum = 0.;
	int t = 1, N = 500000;
	gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, 3525364362);
	while (t <= N) {
		x =  60 * gsl_rng_uniform(r) - 30; //integral over the range [-30, 30]
		*kernel_QDSum += 60 * ((double) 1/N) * pow(kernel_k (x), 2.);
		*kernel_QRSum += 60 * ((double) 1/N) * pow(kernel_k (x), 4.);
		++t;
	}
	gsl_rng_free (r);
}

//calculate the fully nonparametric test statistics. INPUT: a lag-smoothing kernel function (kernel_k), a kernel-weight function for
//conditional moments (kernel_f), a truncation lag (TL), a lag-smoothing parameter (lag_smooth), an integral of the quartic function of kernel_k (kernel_QRSum),
//and a kernel regression bandwidth (bandw). OUTPUT: a double number
template <double kernel_k (double ), double kernel_f (double )>
double Dist_corr::do_Test (int TL, int lag_smooth, double kernel_QRSum, double bandw) {
	int t = 1, s = 1, tau = 1, lag = 0, i = 1, j = 1;
	Matrix xlag_t(TL, d_x), xlag_s(TL, d_x), x_t(1, d_x), x_s(1, d_x), ylag_t(TL, d_y), ylag_s(TL, d_y), y_t(1, d_y), y_s(1, d_y);
	Matrix var_U_x(T-TL, T-TL), var_U_y(T-TL, T-TL);
	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,i,j) firstprivate(xlag_t,xlag_s,x_t,x_s,ylag_t,ylag_s,y_t,y_s)
	for (t = TL+1; t <= T; ++t) {
		for (i = 1; i <= TL; i++) {
			for (j = 1; j <= d_x; j++) {
					xlag_t(i,j) = X(t-i, j);
			}
			for (j = 1; j <= d_y; j++) {
					ylag_t(i,j) = Y(t-i, j);
			}
		}
		for (j = 1; j <= d_x; j++) {
			x_t(1,j) = X(t, j);
		}
		for (j = 1; j <= d_y; j++) {
			y_t(1,j) = Y(t, j);
		}
		for (s = TL+1; s <= T; ++s) {
			for (i = 1; i <= TL; i++) {
				for (j = 1; j <= d_x; j++) {
					xlag_s(i,j) = X(s-i, j);
				}
				for (j = 1; j <= d_y; j++) {
					ylag_s(i,j) = Y(s-i, j);
				}
			}
			for (j = 1; j <= d_x; j++) {
				x_s(1,j) = X(s, j);
			}
			for (j = 1; j <= d_y; j++) {
				y_s(1,j) = Y(s, j);
			}
			var_U_x(t-TL, s-TL) = NReg::var_Ux_ts <kernel_f> (X, xlag_t, xlag_s, x_t, x_s, TL, bandw);
			var_U_y(t-TL, s-TL) = NReg::var_Ux_ts <kernel_f> (Y, ylag_t, ylag_s, y_t, y_s, TL, bandw);
		}
	}
	double aVar = 0.;
	for (s = TL+1; s <= T; s++) {
		for (t = s+1; t <= T; t++) {
			aVar += ((double) 1/pow(T-TL, 2.)) * pow(var_U_x(t-TL, s-TL) * var_U_y(t-TL, s-TL), 2.);
		}
	}
	double weight = 0., sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0., cov = 0., sum = 0.;
	//prod_Var = var(X, TL, alpha(1), beta(1), lambda(1), sigma(1)) * var(Y, TL, alpha(2), beta(2), lambda(2), sigma(2));//calculate product of distance variances
	//cout << "product of variances = " << prod_Var << endl;
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) firstprivate(weight,sum1,sum2,sum3,sum4,cov) private(lag,t,s,tau)
	for (lag = 1-T+TL; lag <= T-TL-1; ++lag) {
        if (lag == 0) weight = 1.;
        else weight = kernel_k ((double) lag/lag_smooth);
        if ((weight > 0.0001) || (weight < -0.0001)) {
        	if (lag >= 0) {
        	    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
        	    for (t = lag+TL+1; t <= T; t++) {
			        for (s = lag+TL+1; s <= T; s++) {
			        	if (t != s) {
				            sum1 += var_U_x(t-TL, s-TL) * var_U_y(t-TL-lag, s-TL-lag);
				        }
				        sum3 += var_U_x(t-TL, s-TL);
				        sum4 += var_U_y(t-TL-lag, s-TL-lag);
				        for (tau = lag+TL+1; tau <= T; tau++) {
					        sum2 += 2 * var_U_x(t-TL, s-TL) * var_U_y(tau-TL-lag, s-TL-lag);
				        }
				    }
			    }
			    cov = ((double) 1/pow(T-TL-lag, 2.)) * sum1 - ((double) 1/pow(T-TL-lag, 3.)) * sum2 + ((double) 1/pow(T-TL-lag, 4.)) * sum3 * sum4;
		    }
		    else {
			    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
			    for (t = TL+1-lag; t <= T; t++) {
			        for (s = TL+1-lag; s <= T; s++) {
			        	if (t != s) {
				            sum1 += var_U_x(t-TL+lag, s-TL+lag) * var_U_y(t-TL, s-TL);
				        }
				        sum3 += var_U_x(t-TL+lag, s-TL+lag);
				        sum4 += var_U_y(t-TL, s-TL);
				        for (tau = TL+1-lag; tau <= T; tau++) {
					        sum2 += 2 * var_U_x(t-TL+lag, s-TL+lag) * var_U_y(t-TL, tau-TL);
				        }
			        }
		        }
			    cov = ((double) 1/pow(T-TL+lag, 2.)) * sum1 - ((double) 1/pow(T-TL+lag, 3.)) * sum2 + ((double) 1/pow(T-TL+lag, 4.)) * sum3 * sum4;
		    }
		    sum += (T-TL) * pow(weight, 2.) * cov;
            //cout << "counting lags: " << lag << endl;
        }
	}
	//cout << "sum = " << sum << endl;
	//return (sum - lag_smooth * kernel_QDSum * prod_Var) / (2 * sqrt(lag_smooth * kernel_QRSum * aVar));
	return sum / (2 * sqrt(lag_smooth * kernel_QRSum * aVar));
}

//calculate the test statistic when X is generated by a Gaussian AR(2) process and Y is generated by a bivariate Gaussian AR(2) process.
//INPUT: a truncation lag (TL), a lag-smoothing bandwidth (lag_smooth), the integral of the quartic polynomial of a kernel (kernel_QRSum),
//a 3x1 vector of intercepts (alpha(1) for X, and alpha(2-3) for Y), 3x1 vectors of AR slopes (beta and lambda), a 3x1 vector of std. deviations of error terms
//(sigma(1) for X, and sigma(2-3) for the error terms (eta_1 and xi) of Y), a correlation between \eta_1 and \eta_2 (rho) for the Y d.g.p,
//a seed for the random generator (seed). OUTPUT: a double number
template <double kernel_k (double )>
double Dist_corr::do_Test (int TL, int lag_smooth, double kernel_QRSum, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, const Matrix &sigma,
                           double rho, unsigned long seed) {
    int t = 1, s = 1, tau = 1, lag = 0;
	Matrix alpha_Y(2, 1), beta_Y(2, 1), lambda_Y(2, 1), sigma_Y(2, 1), var_U_x(T-TL, T-TL), var_U_y(T-TL, T-TL);
	alpha_Y(1) = alpha(2);
	alpha_Y(2) = alpha(3);
	beta_Y(1) = beta(2);
	beta_Y(2) = beta(3);
	lambda_Y(1) = lambda(2);
	lambda_Y(2) = lambda(3);
	sigma_Y(1) = sigma(2);
	sigma_Y(2) = sigma(3);
	Matrix y_t(2, 1), y_s(2, 1), y_t_lag1(2, 1), y_t_lag2(2, 1), y_s_lag1(2, 1), y_s_lag2(2, 1);
	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s) firstprivate(y_t,y_t_lag1,y_t_lag2,y_s,y_s_lag1,y_s_lag2)
	for (t = TL+1; t <= T; t++) {
		y_t(1) = Y(t, 1);
		y_t(2) = Y(t, 2);
		y_t_lag1(1) = Y(t-1, 1);
		y_t_lag1(2) = Y(t-1, 2);
		y_t_lag2(1) = Y(t-2, 1);
		y_t_lag2(2) = Y(t-2, 2);
		for (s = TL+1; s <= T; s++) {
			if (s >= t) {
			    var_U_x(t-TL, s-TL) = GReg::var_U_ts (X(t), X(s), X(t-1), X(t-2), X(s-1), X(s-2), alpha(1), beta(1), lambda(1), sigma(1), seed);
			    var_U_x(s-TL, t-TL) = var_U_x(t-TL, s-TL);
			    y_s(1) = Y(s, 1);
		        y_s(2) = Y(s, 2);
		        y_s_lag1(1) = Y(s-1, 1);
		        y_s_lag1(2) = Y(s-1, 2);
		        y_s_lag2(1) = Y(s-2, 1);
		        y_s_lag2(2) = Y(s-2, 2);
			    var_U_y(t-TL, s-TL) = GReg::var_U_ts (y_t, y_s, y_t_lag1, y_t_lag2, y_s_lag1, y_s_lag2, alpha_Y, beta_Y, lambda_Y, sigma_Y, rho, seed);
			    var_U_y(s-TL, t-TL) = var_U_y(t-TL, s-TL);
		    }
			//cout << var_U_x(t-TL, s-TL) << " , ";
		}
		//cout << "\n";
	}
	double aVar = 0.;
	for (s = TL+1; s <= T; s++) {
		for (t = s+1; t <= T; t++) {
			aVar += ((double) 1/pow(T-TL, 2.)) * pow(var_U_x(t-TL, s-TL) * var_U_y(t-TL, s-TL), 2.);
		}
	}
	double weight = 0., sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0., cov = 0., sum = 0.;
	//prod_Var = var(X, TL, alpha(1), beta(1), lambda(1), sigma(1)) * var(Y, TL, alpha(2), beta(2), lambda(2), sigma(2));//calculate product of distance variances
	//cout << "product of variances = " << prod_Var << endl;
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) firstprivate(weight,sum1,sum2,sum3,sum4,cov) private(lag,t,s,tau)
	for (lag = 1-T+TL; lag <= T-TL-1; lag++) {
        if (lag == 0) weight = 1.;
        else weight = kernel_k ((double) lag/lag_smooth);
        if ((weight > 0.0001) || (weight < -0.0001)) {
        	if (lag >= 0) {
        	    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
        	    for (t = lag+TL+1; t <= T; t++) {
			        for (s = lag+TL+1; s <= T; s++) {
			        	if (t != s) {
				            sum1 += var_U_x(t-TL, s-TL) * var_U_y(t-TL-lag, s-TL-lag);
				        }
				        sum3 += var_U_x(t-TL, s-TL);
				        sum4 += var_U_y(t-TL-lag, s-TL-lag) ;
				        for (tau = lag+TL+1; tau <= T; tau++) {
					        sum2 += 2 * var_U_x(t-TL, s-TL) * var_U_y(tau-TL-lag, s-TL-lag);
				        }
				    }
			    }
			    cov = ((double) 1/pow(T-TL-lag, 2.)) * sum1 - ((double) 1/pow(T-TL-lag, 3.)) * sum2 + ((double) 1/pow(T-TL-lag, 4.)) * sum3 * sum4;
		    }
		    else {
			    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
			    for (t = TL+1-lag; t <= T; t++) {
			        for (s = TL+1-lag; s <= T; s++) {
			        	if (t != s) {
				            sum1 += var_U_x(t-TL+lag, s-TL+lag) * var_U_y(t-TL, s-TL);
				        }
				        sum3 += var_U_x(t-TL+lag, s-TL+lag);
				        sum4 += var_U_y(t-TL, s-TL);
				        for (tau = TL+1-lag; tau <= T; tau++) {
					        sum2 += 2 * var_U_x(t-TL+lag, s-TL+lag) * var_U_y(t-TL, tau-TL);
				        }
			        }
		        }
			    cov = ((double) 1/pow(T-TL+lag, 2.)) * sum1 - ((double) 1/pow(T-TL+lag, 3.)) * sum2 + ((double) 1/pow(T-TL+lag, 4.)) * sum3 * sum4;
		    }
		    sum += (T-TL) * pow(weight, 2.) * cov;
            //cout << "counting lags: " << lag << endl;
        }
	}
	//cout << "sum = " << sum << endl;
	//return (sum - lag_smooth * kernel_QDSum * prod_Var) / (2 * sqrt(lag_smooth * kernel_QRSum * aVar));
	return sum / (2 * sqrt(lag_smooth * kernel_QRSum * aVar));
}

//calculate the test statistic for data generated from a bivarate Gaussian AR(2) process. INPUT: a lag truncation (TL), a lag-smoothing parameter (lag_smooth),
//value of the integral of the quartic function of a kernel (kernel_QRSum), intercepts (alpha), first-order AR slopes (beta), second-order AR slopes (lambda),
//standard deviations of the error term (sigma). OUTPUT: a double number
template <double kernel_k (double )>
double Dist_corr::do_Test (int TL, int lag_smooth, double kernel_QRSum, const Matrix &alpha, const Matrix &beta, const Matrix &lambda,
                           const Matrix &sigma, unsigned long seed) {
	int t = 1, s = 1, tau = 1, lag = 0;
	Matrix var_U_x(T-TL, T-TL), var_U_y(T-TL, T-TL);
	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s)
	for (t = TL+1; t <= T; t++) {
		for (s = TL+1; s <= T; s++) {
			var_U_x(t-TL, s-TL) = GReg::var_U_ts (X(t), X(s), X(t-1), X(t-2), X(s-1), X(s-2), alpha(1), beta(1), lambda(1), sigma(1), seed);
			var_U_y(t-TL, s-TL) = GReg::var_U_ts (Y(t), Y(s), Y(t-1), Y(t-2), Y(s-1), Y(s-2), alpha(2), beta(2), lambda(2), sigma(2), seed);
		}
	}
	double aVar = 0.;
	for (s = TL+1; s <= T; s++) {
		for (t = s+1; t <= T; t++) {
			aVar += ((double) 1/pow(T-TL, 2.)) * pow(var_U_x(t-TL, s-TL) * var_U_y(t-TL, s-TL), 2.);
		}
	}
	double weight = 0., sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0., cov = 0., sum = 0.;
	//prod_Var = var(X, TL, alpha(1), beta(1), lambda(1), sigma(1)) * var(Y, TL, alpha(2), beta(2), lambda(2), sigma(2));//calculate product of distance variances
	//cout << "product of variances = " << prod_Var << endl;
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) firstprivate(weight,sum1,sum2,sum3,sum4,cov) private(lag,t,s,tau)
	for (lag = 1-T+TL; lag <= T-TL-1; lag++) {
        if (lag == 0) weight = 1.;
        else weight = kernel_k ((double) lag/lag_smooth);
        if ((weight > 0.0001) || (weight < -0.0001)) {
        	if (lag >= 0) {
        	    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
        	    for (t = lag+TL+1; t <= T; t++) {
			        for (s = lag+TL+1; s <= T; s++) {
			        	if (t != s) {
				            sum1 += var_U_x(t-TL, s-TL) * var_U_y(t-TL-lag, s-TL-lag);
				        }
				        sum3 += var_U_x(t-TL, s-TL);
				        sum4 += var_U_y(t-TL-lag, s-TL-lag) ;
				        for (tau = lag+TL+1; tau <= T; tau++) {
					        sum2 += 2 * var_U_x(t-TL, s-TL) * var_U_y(tau-TL-lag, s-TL-lag);
				        }
				    }
			    }
			    cov = ((double) 1/pow(T-TL-lag, 2.)) * sum1 - ((double) 1/pow(T-TL-lag, 3.)) * sum2 + ((double) 1/pow(T-TL-lag, 4.)) * sum3 * sum4;
		    }
		    else {
			    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
			    for (t = TL+1-lag; t <= T; t++) {
			        for (s = TL+1-lag; s <= T; s++) {
			        	if (t != s) {
				            sum1 += var_U_x(t-TL+lag, s-TL+lag) * var_U_y(t-TL, s-TL);
				        }
				        sum3 += var_U_x(t-TL+lag, s-TL+lag);
				        sum4 += var_U_y(t-TL, s-TL);
				        for (tau = TL+1-lag; tau <= T; tau++) {
					        sum2 += 2 * var_U_x(t-TL+lag, s-TL+lag) * var_U_y(t-TL, tau-TL);
				        }
			        }
		        }
			    cov = ((double) 1/pow(T-TL+lag, 2.)) * sum1 - ((double) 1/pow(T-TL+lag, 3.)) * sum2 + ((double) 1/pow(T-TL+lag, 4.)) * sum3 * sum4;
		    }
		    sum += (T-TL) * pow(weight, 2.) * cov;
            //cout << "counting lags: " << lag << endl;
        }
	}
	//cout << "sum = " << sum << endl;
	//return (sum - lag_smooth * kernel_QDSum * prod_Var) / (2 * sqrt(lag_smooth * kernel_QRSum * aVar));
	return sum / (2 * sqrt(lag_smooth * kernel_QRSum * aVar));
}

//calculate the test statistic for data generated from a bivarate Student's t AR(2) process. INPUT: a lag truncation (TL), a lag-smoothing parameter (lag_smooth),
//value of the integral of the quartic function of a kernel (kernel_QRSum), intercepts (alpha), first-order AR slopes (beta), second-order AR slopes (lambda),
//degrees of freedom (nu), standard deviations of the error term (sigma), a seed for the random generator (seed). OUTPUT: a double number
template <double kernel_k (double )>
double Dist_corr::do_Test (int TL, int lag_smooth, double kernel_QRSum, const Matrix &alpha, const Matrix &beta, const Matrix &lambda, double nu,
                           const Matrix &sigma, unsigned long seed) {
	int t = 1, s = 1, tau = 1, lag = 0;
	Matrix var_U_x(T-TL, T-TL), var_U_y(T-TL, T-TL);
	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s)
	for (t = TL+1; t <= T; t++) {
		for (s = TL+1; s <= T; s++) {
			var_U_x(t-TL, s-TL) = TReg::var_U_ts (X(t), X(s), X(t-1), X(t-2), X(s-1), X(s-2), alpha(1), beta(1), lambda(1), nu, sigma(1), seed);
			var_U_y(t-TL, s-TL) = TReg::var_U_ts (Y(t), Y(s), Y(t-1), Y(t-2), Y(s-1), Y(s-2), alpha(2), beta(2), lambda(2), nu, sigma(2), seed);
		}
	}
	double aVar = 0.;
	for (s = TL+1; s <= T; s++) {
		for (t = s+1; t <= T; t++) {
			aVar += ((double) 1/pow(T-TL, 2.)) * pow(var_U_x(t-TL, s-TL) * var_U_y(t-TL, s-TL), 2.);
		}
	}
	double weight = 0., sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0., cov = 0., sum = 0.;
	//prod_Var = var(X, TL, alpha(1), beta(1), lambda(1), sigma(1)) * var(Y, TL, alpha(2), beta(2), lambda(2), sigma(2));//calculate product of distance variances
	//cout << "product of variances = " << prod_Var << endl;
	#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) firstprivate(weight,sum1,sum2,sum3,sum4,cov) private(lag,t,s,tau)
	for (lag = 1-T+TL; lag <= T-TL-1; lag++) {
        if (lag == 0) weight = 1.;
        else weight = kernel_k ((double) lag/lag_smooth);
        if ((weight > 0.0001) || (weight < -0.0001)) {
        	if (lag >= 0) {
        	    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
        	    for (t = lag+TL+1; t <= T; t++) {
			        for (s = lag+TL+1; s <= T; s++) {
			        	if (t != s) {
				            sum1 += var_U_x(t-TL, s-TL) * var_U_y(t-TL-lag, s-TL-lag);
				        }
				        sum3 += var_U_x(t-TL, s-TL);
				        sum4 += var_U_y(t-TL-lag, s-TL-lag) ;
				        for (tau = lag+TL+1; tau <= T; tau++) {
					        sum2 += 2 * var_U_x(t-TL, s-TL) * var_U_y(tau-TL-lag, s-TL-lag);
				        }
				    }
			    }
			    cov = ((double) 1/pow(T-TL-lag, 2.)) * sum1 - ((double) 1/pow(T-TL-lag, 3.)) * sum2 + ((double) 1/pow(T-TL-lag, 4.)) * sum3 * sum4;
		    }
		    else {
			    sum1 = 0.;
        	    sum2 = 0.;
        	    sum3 = 0.;
        	    sum4 = 0.;
			    for (t = TL+1-lag; t <= T; t++) {
			        for (s = TL+1-lag; s <= T; s++) {
			        	if (t != s) {
				            sum1 += var_U_x(t-TL+lag, s-TL+lag) * var_U_y(t-TL, s-TL);
				        }
				        sum3 += var_U_x(t-TL+lag, s-TL+lag);
				        sum4 += var_U_y(t-TL, s-TL);
				        for (tau = TL+1-lag; tau <= T; tau++) {
					        sum2 += 2 * var_U_x(t-TL+lag, s-TL+lag) * var_U_y(t-TL, tau-TL);
				        }
			        }
		        }
			    cov = ((double) 1/pow(T-TL+lag, 2.)) * sum1 - ((double) 1/pow(T-TL+lag, 3.)) * sum2 + ((double) 1/pow(T-TL+lag, 4.)) * sum3 * sum4;
		    }
		    sum += (T-TL) * pow(weight, 2.) * cov;
            //cout << "counting lags: " << lag << endl;
        }
	}
	//cout << "sum = " << sum << endl;
	//return (sum - lag_smooth * kernel_QDSum * prod_Var) / (2 * sqrt(lag_smooth * kernel_QRSum * aVar));
	return sum / (2 * sqrt(lag_smooth * kernel_QRSum * aVar));
}















#endif
