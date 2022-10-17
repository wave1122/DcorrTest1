/* unirandi.f -- translated by f2c (version 19940714).*/

#include  <math.h>
#include "globdef.h"

/* Subroutine */ void local(int *m, int *n, double *relcon, int *maxfn, 
                            VEKTOR_n x, double *f, int *nfev, void* parms, 
                            Matrix& data, VEKTOR_n minx, VEKTOR_n maxx)
{
    /* Initialized data */

    static double zero = (float)0.;
    static double onen3 = (float).001;
    static double half = (float).5;
    static double one = (float)1.;
    static double two = (float)2.;

    /* System generated locals */
    int i__1;
    double r__1;

    /* Local variables */
    static double a, h;
    static int i;
    static double deltf;
    static int irndm;
    static double f1;
    static int itest;
    static double x1[21], eps;
    TOMB_nx101 r;
if((r = (TOMB_nx101)malloc(sizeof(double)*(*n+1)*101))==NULL) exit(3);


    /* Parameter adjustments */
/*    --maxx;     bad idea because of
    --minx;       the habits of the caller
    r -= 101;     functions
    --x;
*/
    /* Function Body */
/* 				   FIRST EXECUTABLE STATEMENT */
/* 				   INITIAL STEP LENGTH */
    h = onen3;
    deltf = one;
    itest = 0;
    *nfev = 0;
    eps = *relcon;
/* 				   EVALUATE 100 RANDOM VECTORS */
L5:
    urdmn(r, n);
    irndm = 0;
L15:
    ++irndm;
    if (irndm > 100) {
	goto L5;
    }
/* 				   SELECT A RANDOM VECTOR HAVING NORM */
/* 				     LESS OR EQUAL TO 0.5 */
    a = zero;
    i__1 = *n;
    for (i = 1; i <= i__1; ++i) {
	(*r)[i][irndm] -= half;
/* L20: */
	a += (*r)[i][irndm] * (*r)[i][irndm];
    }
    if (a <= zero) {
	goto L15;
    }
    a = sqrt(a);
    if (a > half) {
	goto L15;
    }
/* 				   NEW TRIAL POINT */
    i__1 = *n;
    for (i = 1; i <= i__1; ++i) {
	(*r)[i][irndm] /= a;
/* L25: */
	x1[i] = x[i] + h * (*r)[i][irndm];
    }
    fun(x1, &f1, n, m, parms, data, minx, maxx);
    ++(*nfev);
    if (f1 < *f) {
	goto L35;
    }
    if (*nfev > *maxfn) {
	goto L50;
    }
/* 				   STEP IN THE OPPOSITE DIRECTION */
    h = -(double)h;
    i__1 = *n;
    for (i = 1; i <= i__1; ++i) {
/* L30: */
	x1[i] = x[i] + h * (*r)[i][irndm];
    }
    fun(x1, &f1, n, m, parms, data, minx, maxx);
    ++(*nfev);
    if (f1 < *f) {
	goto L35;
    }
    if (*nfev > *maxfn) {
	goto L50;
    }
    ++itest;
    if (itest < 2) {
	goto L15;
    }
/* 				   DECREASE STEP LENGTH */
    h *= half;
    itest = 0;
/* 				   RELATIVE CONVERGENCE TEST FOR THE */
/* 				     OBJECTIVE FUNCTION */
    if (deltf < eps) {
	goto L50;
    }
/* 				   CONVERGENCE TEST FOR THE STEP LENGTH */
    if (fabs(h) - *relcon >= (float)0.) {
	goto L15;
    } else {
	goto L50;
    }
L35:
    i__1 = *n;
    for (i = 1; i <= i__1; ++i) {
/* L40: */
	x[i] = x1[i];
    }
    deltf = (*f - f1) / fabs(f1);
    *f = f1;
/* 				   INCREASE STEP LENGTH */
    h *= two;
    i__1 = *n;
    for (i = 1; i <= i__1; ++i) {
/* L45: */
	x1[i] = x[i] + h * (*r)[i][irndm];
    }
    fun(x1, &f1, n, m, parms, data, minx, maxx);
    ++(*nfev);
    if (f1 < *f) {
	goto L35;
    }
/* 				   CHECK TOLERANCE MAXFN */
    if (*nfev > *maxfn) {
	goto L50;
    }
/* 				   DECREASE STEP LENGTH */
    h = (r__1 = h * half, fabs(r__1));
    goto L15;
L50:
    free(r);
} /* local */

