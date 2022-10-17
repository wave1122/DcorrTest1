#ifndef NREG_H
#define NREG_H

#include <kernel.h>

#define CHUNK 1

using namespace std;

class NReg {
	public:
		NReg () {   };//default constructor
		~NReg () {   };//default destructor
		//do kernel regression. INPUT: X is a T by d_x matrix, x0 (1 by d_x) and x1 (TL by d_x) are the values of the regression function, 
        //TL is the truncation lag, banw is a bandwidth
        //OUTPUT: a double number
        template <double kernel_f (double )>
        double reg_F (const Matrix &X, const Matrix &x1, const Matrix &x0, int TL, double bandw);
        //do kernel estimation of m(Z_t,Z_s). INPUT: data is a T by d_x matrix X, x_1 and x_2 are TL by d_x matrices, TL is a truncation lag, and bandw is
        //a bandwidth.
        //OUTPUT: a double number
        template <double kernel_f (double )>
        double reg_BF (const Matrix &X, const Matrix &x1, const Matrix &x2, int TL, double bandw);
    protected:
        //calculate U_{t,s}. INPUT: X is a T by d_x matrix of data, x_t and x_s are TL by d_x matrices, y_t and y_s are 1 by d_x vectors.
        //OUTPUT: a double number
        template <double kernel_f(double )>
        double var_Ux_ts (const Matrix &X, const Matrix &x_t, const Matrix &x_s, const Matrix &y_t, const Matrix &y_s, int TL, double bandw);
	private:
		//calculate the denominator term of the nonparametric regression function
        template <double kernel_f (double )>
        double weight (const Matrix &X, const Matrix &x1, int TL, double bandw);
};

//do kernel regression. INPUT: X is a T by d_x matrix, x0 (1 by d_x) and x1 (TL by d_x) are the values of the regression function, 
//TL is the truncation lag, banw is a bandwidth
//OUTPUT: a double number
template <double kernel_f (double )>
double NReg::reg_F (const Matrix &X, const Matrix &x1, const Matrix &x0, int TL, double bandw) {
	int T = X.nRow(), d_x = X.nCol();
	Matrix Z(TL, d_x), Y(1, d_x);
	int t = 1, i = 1, j = 1;
	double result = 0.;
	//#pragma omp parallel for default(shared) reduction (+:result) schedule(dynamic,CHUNK) private(t,i,j) firstprivate(Z,Y)
	for (t = TL + 1; t <= T; t++) {
		for (i = 1; i <= TL; i++) {
			for (j = 1; j <= d_x; j++) {
				Z(i, j) = X(t-i, j);
			}
		}
		for (j = 1; j <= d_x; j++) {
			Y(1, j) = X(t, j);
		}
		result += ENorm(x0 - Y) * kernel_f ((1/bandw) * ENorm(x1 - Z));
		//cout << "result = " << result << endl;
	}
	return (1/NReg::weight <kernel_f> (X, x1, TL, bandw)) * result;
}

//calculate the denominator term of the nonparametric regression function. INPUT: X (T by d_x) and x1 (TL by dx)
template <double kernel_f (double )>
double NReg::weight (const Matrix &X, const Matrix &x1, int TL, double bandw) {
	int tau = 1, i = 1, j = 1;
	int T = X.nRow(), d_x = X.nCol();
	double sum = 0., temp = 0.;
	Matrix Z(TL, d_x);
	//#pragma omp parallel for default(shared) reduction (+:sum) schedule(dynamic,CHUNK) private(tau,i,j) firstprivate(Z,temp)
	for (tau = TL + 1; tau <= T; tau++) {
			for (i = 1; i <= TL; i++) {
			    for (j = 1; j <= d_x; j++) {
				    Z(i, j) = X(tau-i, j);
			    }
		    }
		    //cout << (1/bandw) * ENorm(x1 - Z) << endl;
		    temp = kernel_f ((1/bandw) * ENorm(x1 - Z));
			sum += temp;
	}
	return sum;
}

//do kernel estimation of m(Z_t,Z_s). INPUT: data is a T by d_x matrix X, x_1 and x_2 are TL by d_x matrices, TL is a truncation lag, and bandw is
//a bandwidth.
//OUTPUT: a double number
template <double kernel_f (double )>
double NReg::reg_BF (const Matrix &X, const Matrix &x1, const Matrix &x2, int TL, double bandw) {
	int T = X.nRow(), d_x = X.nCol();
	Matrix Y1(1, d_x), Y2(1, d_x), Z(TL, d_x);
	int t = 1, s = 1, i = 1, j = 1;
	Matrix kernel1(T-TL, 1), kernel2(T-TL, 1);
	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,i,j) firstprivate(Z)
	for (t = TL + 1; t <= T; t++) {
		for (i = 1; i <= TL; i++) {
			    for (j = 1; j <= d_x; j++) {
				    Z(i, j) = X(t-i, j);
			    }
		}
		kernel1(t-TL) = kernel_f ((1/bandw) * ENorm (x1 - Z)); 
		kernel2(t-TL) = kernel_f ((1/bandw) * ENorm (x2 - Z));
    }
    double result = 0.;
    #pragma omp parallel for default(shared) reduction (+:result) schedule(dynamic,CHUNK) private(t,s,j) firstprivate(Y1,Y2)
    for (t = TL + 1; t <= T; t++) {
	    for (s = TL + 1; s <= T; s++) {
	    	if (t != s) {
			    for (j = 1; j <= d_x; j++) {
				    Y1(1, j) = X(t, j);
				    Y2(1, j) = X(s, j);
			    }
			    result += kernel1(t-TL) * kernel2(s-TL) * ENorm(Y1 - Y2);
			}
			//cout << "result = " << result << endl;
		}
	}
    return (1/NReg::weight <kernel_f> (X, x1, TL, bandw)) * (1/NReg::weight <kernel_f> (X, x2, TL, bandw)) * result;
}

//calculate U_{t,s}. INPUT: X is a T by d_x matrix of data, x_t and x_s are TL by d_x matrices, y_t and y_s are 1 by d_x vectors.
//OUTPUT: a double number
template <double kernel_f(double )>
double NReg::var_Ux_ts (const Matrix &X, const Matrix &x_t, const Matrix &x_s, const Matrix &y_t, const Matrix &y_s, int TL, double bandw) {
	return (NReg::reg_F <kernel_f> (X, x_t, y_s, TL, bandw) + NReg::reg_F <kernel_f> (X, x_s, y_t, TL, bandw) 
	        - ENorm(y_t - y_s) - NReg::reg_BF <kernel_f> (X, x_t, x_s, TL, bandw));
}


#endif
