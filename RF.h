#ifndef RF_H_
#define RF_H_

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

using namespace std;

class RF {
	public:
		RF () {   }; //default constructor
		~RF () {   };//default destructor

        //generate TWO threshold AR processes of the second order. INPUT: A 5x1 vector of coefficients for X (alpha_X), a 5x1 vector of coefficients for Y (alpha_Y),
        //and a template function to generate two random errors for both the processes (gen_RAN). OUTPUT: Tx1 matrices (X and Y).
        template <void gen_RAN (double &, double &, const double, const double, const int, unsigned long)>
        static void gen_TAR (Matrix &X, Matrix &Y, const Matrix alpha_X, const Matrix alpha_Y, const double delta, const double rho, const int choose_alt,
		                     unsigned long seed);
};




#endif
