#ifndef REG_H
#define REG_H

#include <kernel.h>

using namespace std;

class NReg {
	public:
		NReg () {   };//default constructor
		~NReg () {   };//default destructor
		//do kernel regression. INPUT: X is a T by d_x matrix, x0 (1 by d_x) and x1 (TL by d_x) are the values of the regression function, 
        //TL is the truncation lag, banw is a bandwidth
        //OUTPUT: a double number
        template <double kernel_f (double )>
        double reg_F (const Matrix &X, Matrix x1, Matrix x0, int TL, double bandw)
	private:
		//calculate the denominator term of the nonparametric regression function
        template <double kernel_f (double )>
        double weight (const Matrix &X, Matrix x1, int TL, double bandw);
};
//do kernel regression. INPUT: X is a T by d_x matrix, x0 (1 by d_x) and x1 (TL by d_x) are the values of the regression function, 
//TL is the truncation lag, banw is a bandwidth
//OUTPUT: a double number
template <double kernel_f (double )>
double NReg::reg_F (const Matrix &X, Matrix x1, Matrix x0, int TL, double bandw) {
	int T = X.nRow(), d_x = X.nCol();
	Matrix Z(TL, d_x), Y(1, d_x);
	int t = 1, i = 1, j = 1;
	double result = 0.;
	for (t = TL + 1; t <= T; t++) {
		for (i = 1; i <= TL; i++) {
			for (j = 1; j <= d_x; j++) {
				Z(i,j) = X(t-i,j);
			}
		}
		for (j = 1; j <= d_x; j++) {
			Y(1,j) = X(t,j);
		}
		result += (1/NReg::weight<kernel_f>(X, x1, TL, bandw)) * ENorm(x_0 - Y) * kernel_f ((1/bandw) * ENorm(x1 - Z))
	}
	return result;
}

//calculate the denominator term of the nonparametric regression function. INPUT: X (T by d_x) and x1 (TL by dx)
template <double kernel_f (double )>
double NReg::weight (const Matrix &X, Matrix x1, int TL, double bandw) {
	int tau = 1, i = 1, j = 1;
	int T = X.nRow(), d_x = X.nCol();
	double sum = 0.;
	Matrix Z(TL, d_x);
	for (tau = TL + 1; tau <= T; tau++) {
			for (i = 1; i <= TL; i++) {
			    for (j = 1; j <= d_x; j++) {
				    Z(i,j) = X(t-i,j);
			    }
		    }
			sum += kernel_f ((1/bandw) * ENorm(x1 - Z));
	}
	return sum;
}









#endif
