#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <NumMeth.h>
#include <valarray>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <asserts.h>

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

//#define n_threads 10

using namespace std;
using namespace shogun;
using namespace shogun::linalg;

/* Display a matrix */
void display_matrix(const Matrix &A, const char *name = "matrix") {
	int num_rows = A.nRow(), num_cols = A.nCol();

	// Copy to a Shogun matrix
	SGMatrix<double>B(num_rows, num_cols);
	for (int i = 0; i < num_rows; ++i) {
		for (int j = 0; j < num_cols; ++j) {
			B(i,j) = A(i+1,j+1);
		}
	}
	B.display_matrix(name);
}

Matrix inv(Matrix A)
// Input
//    A    -    Matrix A (N by N)
// Outputs
//   Ainv  -    Inverse of matrix A (N by N)
//  determ -    Determinant of matrix A	(return value)
{

    int N = A.nRow();
    ASSERT_( N == A.nCol() );
    Matrix Ainv(N,N);
    Ainv = A;  // Copy matrix to ensure Ainv is same size

    int i, j, k;
    Matrix scale(N), b(N,N);	 // Scale factor and work array
    int *index;
    index = new int [N+1];

    //* Matrix b is initialized to the identity matrix
    b.set(0.0);
    for( i=1; i<=N; i++ )
        b(i,i) = 1.0;

    //* Set scale factor, scale(i) = max( |a(i,j)| ), for each row
    for( i=1; i<=N; i++ )
    {
        index[i] = i;			  // Initialize row index list
        double scalemax = 0.;
        for( j=1; j<=N; j++ )
            scalemax = (scalemax > fabs(A(i,j))) ? scalemax : fabs(A(i,j));
        scale(i) = scalemax;
    }

    //* Loop over rows k = 1, ..., (N-1)
    int signDet = 1;
    for( k=1; k<=N-1; k++ )
    {
        //* Select pivot row from max( |a(j,k)/s(j)| )
        double ratiomax = 0.0;
        int jPivot = k;
        for( i=k; i<=N; i++ )
        {
            double ratio = fabs(A(index[i],k))/scale(index[i]);
            if( ratio > ratiomax )
            {
                jPivot=i;
                ratiomax = ratio;
            }
        }
        //* Perform pivoting using row index list
        int indexJ = index[k];
        if( jPivot != k )  	          // Pivot
        {
            indexJ = index[jPivot];
            index[jPivot] = index[k];   // Swap index jPivot and k
            index[k] = indexJ;
            signDet *= -1;			  // Flip sign of determinant
        }
        //* Perform forward elimination
        for( i=k+1; i<=N; i++ )
        {
            double coeff = A(index[i],k)/A(indexJ,k);
            for( j=k+1; j<=N; j++ )
                A(index[i],j) -= coeff*A(indexJ,j);
            A(index[i],k) = coeff;
            for( j=1; j<=N; j++ )
                b(index[i],j) -= A(index[i],k)*b(indexJ,j);
        }
    }
    //* Compute determinant as product of diagonal elements
    double determ = signDet;	   // Sign of determinant
    for( i=1; i<=N; i++ )
        determ *= A(index[i],i);

    //* Perform backsubstitution
    for( k=1; k<=N; k++ )
    {
        Ainv(N,k) = b(index[N],k)/A(index[N],N);
        for( i=N-1; i>=1; i--)
        {
            double sum = b(index[i],k);
            for( j=i+1; j<=N; j++ )
                sum -= A(index[i],j)*Ainv(j,k);
            Ainv(i,k) = sum/A(index[i],i);
        }
    }

    delete [] index;	// Release allocated memory
    return Ainv;
    //return( determ );
}

//Inverse a square matrix by using the GSL linear algebra library.
void inverse (Matrix &res, const Matrix &X) {
    int i=0,j=0,signum=0, N = X.nRow();
    ASSERT_(N == X.nCol()); //check for matrix size consistency
    gsl_matrix *A = gsl_matrix_alloc(N, N);
    gsl_matrix *A_inv = gsl_matrix_alloc(N, N);

    for (i=0;i<N;i++)
        for (j=0;j<N;j++)
            gsl_matrix_set(A, i, j, X(i+1,j+1)); //read all the elements of X to the matrix (A)

    //gsl_linalg_cholesky_decomp1(A); //get a Cholesky decomposition matrix
    //gsl_linalg_cholesky_invert(A); //calculate the inverse of  a positive definite matrix (X) from its Cholesky decomposition (A)
    gsl_permutation *p = gsl_permutation_alloc(N);
    gsl_linalg_LU_decomp(A, p, &signum); //do LU decomposition and inversion for any square invertible matrix
    gsl_linalg_LU_invert(A, p, A_inv);
    gsl_permutation_free(p); //free memory
    gsl_matrix_free(A); //free memory
    for (i=0;i<N;i++)
        for (j=0;j<N;j++)
            res(i+1,j+1) = gsl_matrix_get(A_inv, i, j); //read all outputs to the matrix (res)
    gsl_matrix_free(A_inv); //free memory
}

double determ (const Matrix& A0)
// Input
//    A    -    Matrix A (N by N)
// Outputs
//   Ainv  -    Inverse of matrix A (N by N)
//  determ -    Determinant of matrix A	(return value)
{

    int N = A0.nRow();
    ASSERT_( N == A0.nCol() );
    Matrix Ainv(N,N), A(N,N);
    A = A0;
    Ainv = A;  // Copy matrix to ensure Ainv is same size

    int i, j, k;
    Matrix scale(N), b(N,N);	 // Scale factor and work array
    int *index;
    index = new int [N+1];

    //* Matrix b is initialized to the identity matrix
    b.set(0.0);
    for( i=1; i<=N; i++ )
        b(i,i) = 1.0;

    //* Set scale factor, scale(i) = max( |a(i,j)| ), for each row
    for( i=1; i<=N; i++ )
    {
        index[i] = i;			  // Initialize row index list
        double scalemax = 0.;
        for( j=1; j<=N; j++ )
            scalemax = (scalemax > fabs(A(i,j))) ? scalemax : fabs(A(i,j));
        scale(i) = scalemax;
    }

    //* Loop over rows k = 1, ..., (N-1)
    int signDet = 1;
    for( k=1; k<=N-1; k++ )
    {
        //* Select pivot row from max( |a(j,k)/s(j)| )
        double ratiomax = 0.0;
        int jPivot = k;
        for( i=k; i<=N; i++ )
        {
            double ratio = fabs(A(index[i],k))/scale(index[i]);
            if( ratio > ratiomax )
            {
                jPivot=i;
                ratiomax = ratio;
            }
        }
        //* Perform pivoting using row index list
        int indexJ = index[k];
        if( jPivot != k )  	          // Pivot
        {
            indexJ = index[jPivot];
            index[jPivot] = index[k];   // Swap index jPivot and k
            index[k] = indexJ;
            signDet *= -1;			  // Flip sign of determinant
        }
        //* Perform forward elimination
        for( i=k+1; i<=N; i++ )
        {
            double coeff = A(index[i],k)/A(indexJ,k);
            for( j=k+1; j<=N; j++ )
                A(index[i],j) -= coeff*A(indexJ,j);
            A(index[i],k) = coeff;
            for( j=1; j<=N; j++ )
                b(index[i],j) -= A(index[i],k)*b(indexJ,j);
        }
    }
    //* Compute determinant as product of diagonal elements
    double determ = signDet;	   // Sign of determinant
    for( i=1; i<=N; i++ )
        determ *= A(index[i],i);

    delete [] index;	// Release allocated memory
    return( determ );
}

/* -----------------------------------------------
        Cholesky decomposition.

        input    n  size of matrix
        input    A  Symmetric positive def. matrix
        output   a  lower deomposed matrix
        uses        choldc1(int,MAT,VEC)
----------------------------------------------- */
/* ----------------------------------------------------
        main method for Cholesky decomposition.

        input         n  size of matrix
        input/output  a  Symmetric positive def. matrix
        output        p  vector of resulting diag of a
        author:       <Vadum Kutsyy, kutsyy@hotmail.com>
----------------------------------------------------- */

//using namespace std;
static void choldc1(int n, Matrix a, vector<double>  p)
{
    int i,j,k;
    double sum;
    for (i = 1; i <= n; i++)
    {
        for (j = i; j <= n; j++)
        {
            sum = a(i,j);
            for (k = i-1; k > 0; k--)
            {
                sum  = sum - a(i,k)*a(j,k);
                //cout << a(i,k) <<"," << a(j,k);
                //cout << sum;
            }
            if (i == j)
            {
                if (sum <= 0)
                {
                    printf(" a is not positive definite!\n");
                }
                p[i-1] = sqrt(sum);
            }
            else
            {
                a(j,i) = sum / p[i-1];
            }
        }
    }
}

void choldc(int n, Matrix A, Matrix  a)
{
    int i,j;
    vector<double> p(n);
    for (i = 1; i <= n; i++)
    {
        for (j = 1; j <= n; j++)
        {
            a(i,j) = A(i,j);
        }
    }
    choldc1(n, a, p);
    for (i = 1; i <= n; i++)
    {
        a(i,i) = p[i-1];
        for (j = i + 1; j <= n; j++)
        {
            a(i,j) = 0;
        }
    }
}
/* Compute the eigenvalues of a real SYMMETRIC matrix */
Matrix eigenv (Matrix X)
{
    int nr, nc, i, j;
    nr = X.nRow();
    nc = X.nCol();
    double *data;
    data = (double *) malloc(nr*nc*sizeof(double));
    for (i = 1; i <= nr; i++)
    {
        for (j = 1; j <= nc; j++)
        {
            data[(i-1)*nc+j-1] = X(i,j);
        }
    }
    gsl_matrix_view m = gsl_matrix_view_array (data, nr, nc);
    gsl_vector *eval = gsl_vector_alloc (nr);
    gsl_matrix *evec = gsl_matrix_alloc (nr, nr);
    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (nr);
    gsl_eigen_symmv (&m.matrix, eval, evec, w);
    gsl_eigen_symmv_free (w);
    gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);
    Matrix res(nr,1);
    for (i = 0; i < nr; i++)
    {
        res(i+1) = gsl_vector_get (eval, i);
    }
    gsl_vector_free (eval);
    gsl_matrix_free (evec);
    free(data);
    return res;
}

// Multiply matrices
Matrix mulMatrix(Matrix A, Matrix B)
{
    int N = A.nRow();
    int M = A.nCol();
    int M1 = B.nRow();
    int H = B.nCol();
    ASSERT_(M == M1);
    Matrix C(N,H);
    C.set(0.0);
    int i = 1, j = 1, k = 1;

    if ((N == 1) && (H == 1))
    {
        #pragma omp parallel for default(shared) schedule(dynamic) private(k) if (M > 1000)
        for (k = 1; k <= M; k++)
        {
            #pragma omp atomic
            C(1,1) += A(1,k)*B(k,1);
            //check_parallel = omp_in_parallel();
            //tid = omp_get_num_threads();
            //cout << "operator*(Matrix A, Matrix B): this section is executing in parallel = " << check_parallel << endl;
            //cout << "number of threads = " << tid << endl;
        }
    }
    else
    {
        #pragma omp parallel for default(shared) schedule(dynamic) private(i,j,k) if ((N > 1000) || (H > 1000) || (M > 1000))
        for (i = 1; i <= N; i++)
        {
            //check_parallel = omp_in_parallel();
            //tid = omp_get_num_threads();
            //cout << "operator*(Matrix A, Matrix B): this section is executing in parallel = " << check_parallel << endl;
            //cout << "number of threads = " << tid << endl;
            for (j = 1; j <= H; j++)
            {
                for (k = 1; k <= M; k++)
                {
                    #pragma omp atomic
                    C(i,j) += A(i,k)*B(k,j);
                }
            }
        }
    }
    return C;
}

double mulVec(Matrix A, Matrix B) //multiply a row vector by acolumn vector
{
    int M = A.nCol();
    int M1 = B.nRow();
    ASSERT_(M == M1);
    double C = 0.;
    int k = 0;
    #pragma omp parallel for default(shared) reduction(+:C) schedule(dynamic) private(k) if (M > 1000)
    for (k = 1; k <= M; k++)
    {
        C += A(1,k)*B(k,1);
    }
    return C;
}

class mulVecMat
{
public:
    mulVecMat(Matrix _A, Matrix _B)
    {
        A = _A;
        B = _B;
        used = false;
    }
    ~mulVecMat()
    {
        if (!used)
            cerr << "error: ambiguous use of function" << "overloaded on return type" << endl;
    }
    operator Matrix()
    {
        used = true;
        return mulMatrix(A,B);
    }
    operator double()
    {
        used = true;
        return mulVec(A,B);
    }
private:
    Matrix A;
    Matrix B;
    bool used;
};

mulVecMat operator*(Matrix A, Matrix B)
{
    return mulVecMat(A,B);
}



//template<Matrix f(Matrix)>
/*double mulvec(Matrix A, Matrix B)
{
 int M = A.nCol();
 int M1 = B.nRow();
 ASSERT_(M == M1);
 double C = 0;
 for (int k = 1; k <= M; k++)
     {
        C += A(1,k)*B(k,1);
     }
 return C;
}*/

// Multiply a matrix with a scalar
Matrix operator*(double a, Matrix B)
{
    Matrix C(B.nRow(),B.nCol());
    int i = 0, j = 0;
    #pragma omp parallel for default(shared) schedule(dynamic) private(i,j) if ((B.nRow() >= 1000) || (B.nCol() >= 1000))
    for (i = 1; i <= B.nRow(); i++)
    {
        for (j = 1; j <= B.nCol(); j++)
        {
            C(i,j) = a*B(i,j);
        }
    }
    return C;
}


//Transpose matrices
/*Matrix Tranvec(Matrix A) //1xN => Nx1
{
 int N = A.nCol();
 Matrix At(N,1);
 ASSERT_ (N ==At.nRow());
 for (int i = 1; i <= N; ++i)
 {
      At(i,1) = A(1,i);
 }
 return At;
}*/
Matrix Tr(Matrix A)//NxM => MxN
{
    int N = A.nCol();
    int M = A.nRow();
    int i, j;
    Matrix B(N,M);
    ASSERT_(N==B.nRow());
    #pragma omp parallel for default(shared) schedule(dynamic) private(j,i) if ((N > 1000) || (M > 1000))
    for (j = 1; j <= N; j++)
    {
        for (i = 1; i <= M; i++)
        {
            B(j,i) = A(i,j);
        }
    }
    return B;
}

//Euclean norm of a matrix
double ENorm(Matrix A)
{
	int N = A.nRow();
	int M = A.nCol();
	int i = 0, j = 0;
	double norm = 0.;
	#pragma omp parallel for default(shared) reduction(+:norm) schedule(dynamic) private(i,j) if ((N > 1000) && (M > 1000))
	for (i = 1; i <= N; i++) {
		for(j = 1; j <= M; j++) {
			norm += pow(A(i,j), 2.);
		}
	}
	return sqrt(norm);
}

double AbsNorm(Matrix A)
{
    int N = A.nRow();
    int M = A.nCol();
    int i = 0, j = 0;
    double norm = 0;
    #pragma omp parallel for default(shared) reduction(+:norm) schedule(dynamic) private(i,j) if ((N > 1000) && (M > 1000))
    for (i = 1; i <= N; i++)
    {
        for(j =1; j <= M; j++)
        {
            norm += fabs(A(i,j));
        }
    }
    return norm;
}

// Add two matrices
Matrix operator+(Matrix A, Matrix B)
{
    int N, M, i, j;
    N = A.nRow();
    M = A.nCol();
    Matrix C(N,M);
    #pragma omp parallel for default(shared) schedule(dynamic) private(i,j) if ((N > 1000) && (M > 1000))
    for (i = 1; i <= N; i++)
    {
        for(j = 1; j <= M; j++)
        {
            C(i,j) = A(i,j) + B(i,j);
        }
    }
    return C;
}

//sum of a const and a Matrix
Matrix operator+(double a, Matrix A)
{
    int m = A.nRow();
    int n = A.nCol();
    Matrix C(m,n);
    for (int i = 1; i <= m; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            C(i,j) = a + A(i,j);
        }
    }
    return C;
}


// Substract two matrices
Matrix operator-(Matrix A, Matrix B)
{
    int N, M, i, j;
    N = A.nRow();
    M = A.nCol();
    Matrix C(N,M);
    #pragma omp parallel for default(shared) schedule(dynamic) private(i,j) if ((N > 1000) && (M > 1000))
    for (i = 1; i <= N; i++)
    {
        for(j = 1; j <= M; j++)
        {
            C(i,j) = A(i,j) - B(i,j);
        }
    }
    return C;
}

// Create an identity matrix of size N
Matrix eye(int N)
{
    Matrix C(N,N);
    C.set(0.0);
    int i = 1;
    //#pragma omp parallel for default(shared) schedule(dynamic) private(i)
    for (i = 1; i <= N; i++)
    {
        C(i,i) = 1.0;
    }
    return C;
}

//Create a Nx1 column vector of ones
Matrix ones(int N)
{
    Matrix C(N,1);
    C.set(0.);
    for (int i = 1; i <= N; i++)
    {
        C(i,1) = 1.;
    }
    return C;
}

//compute sum of the elements of a column vector
double sumV(Matrix X)
{
    double res = 0.;
    for (int i = 1; i <= X.nRow(); i++)
    {
        res += X(i,1);
    }
    return res;
}

Matrix sumRow(Matrix& X)
{
    int N = 0;
    N = X.nRow();
    Matrix res(N,1);
    res.set(0.);
    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= X.nCol(); j++)
        {
            res(i,1) += X(i,j);
        }
    }
    return res;
}

Matrix sumCol(Matrix& X)
{
    int N = 0;
    N = X.nCol();
    Matrix res(N,1);
    res.set(0.);
    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= X.nRow(); j++)
        {
            res(i,1) += X(j,i);
        }
    }
    return res;
}

//Compute sample means
Matrix mean (const Matrix &data)
{
    int N, M;
    N = data.nRow();
    M = data.nCol();
    Matrix res(M,1);
    res.set(0.);
    for (int i = 1; i <= M; i++)
    {
        for (int j = 1; j <= N; j++)
        {
            res(i,1) += ((double) 1/N) * data(j,i);//sum over the rows
        }
    }
    return res;
}


//Compute sample means
double mean_u (const Matrix &data)
{
    int i, N;
    N = data.nRow();
    double res = 0.;
    for (i = 1; i <= N; i++)
    {
		res += ((double) 1/N) * data(i);//sum over the rows
    }
    return res;
}

//Compute sample variance
double variance (Matrix data)
{
    int i = 1, N = 1;
    N = data.nRow();
    double mean = 0., res = 0.;
    mean = mean_u(data);
    for (i = 1; i <= N; i++)
    {
        res += ((double) 1/N) * pow(data(i) - mean, 2.);
    }
    return res;
}

//Compute sample second moment
double secondmom (Matrix data)
{
    int i, N;
    N = data.nRow();
    double res = 0.;
    for (i = 1; i <= N; i++)
    {
        res += ((double) 1/N) * pow(data(i), 2.);
    }
    return res;
}

//Compute a sample variance-covariance matrix
Matrix covariance (Matrix& data)
{
    int N, M;
    N = data.nRow();
    M = data.nCol();
    Matrix mu(M,1), X(N,M), X_t(M,N), tmp(M,M);
    mu = mean (data);
    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= M; j++)
        {
            X(i,j) = data(i,j) - mu(j,1);
            if ((X(i,j) <= 0.0000001) && (X(i,j) >= -0.0000001))
            {
                X(i,j) = 0.;
            }
        }
    }
    X_t = Tr(X);
    tmp = X_t*X;
    return ((double) 1/N)*tmp;
}

//Construct a diagonal matrix from a column vector of numbers
Matrix diag (Matrix X)
{
    int N = 0;
    N = X.nRow();
    Matrix res (N,N);
    res.set(0.);
    for (int i = 1; i <= N; i++)
    {
        res(i,i) = X(i,1);
    }
    return res;
}

Matrix operator->*(Matrix A, Matrix B)//Hadamard (dot) product
{
    int N, M;
    N = A.nRow();
    M = A.nCol();
    ASSERT_(N == B.nRow());
    ASSERT_(M == B.nCol());
    Matrix res (N,M);
    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= M; j++)
        {
            res(i,j) = A(i,j)*B(i,j);
        }
    }
    return res;
}

Matrix operator^(Matrix A, double n)//raising a diagonal matrix to a power
{
    int N;
    N = A.nRow();
    ASSERT_ (N == A.nCol());
    for (int i = 1; i <= N; i++)
    {
        A(i,i) = pow(A(i,i), n);
    }
    return A;
}

SGVector<double> pow_vec(const SGVector<double> &A, double n) //raising a Shogun vector to a power
{
    int N = A.vlen;

    SGVector<double> A_pwr(N);
    A_pwr.zero();
	if (fabs(n - 1.) <= 1e-2) {
		for (int i = 0; i < N; ++i) {
			A_pwr[i] = fabs(A[i]);
		}
	}
	else {
		for (int i = 0; i < N; ++i) {
			A_pwr[i] = std::pow(A[i], n);
		}
	}

    return A_pwr;
}

double determot (Matrix& A)
{
    int N;
    N = A.nRow();
    ASSERT_(N == A.nCol());
    double res = 1.;
    for (int i = 1; i <= N; i++)
    {
        res *= A(i,i);
    }
    return res;
}

inline double scalar(Matrix A)
{
    return A(1,1);
}

inline double max(Matrix A)
{
    valarray<double> temp(A.nRow());
    for (int i = 0; i < A.nRow(); i++)
    {
        temp[i] = fabs(A(i+1));
    }
    return temp.max();
}

inline double min(Matrix A)
{
    valarray<double> temp(A.nRow());
    int i = 0;
    for (i = 0; i < A.nRow(); i++)
    {
        temp[i] = fabs(A(i+1));
    }
    return temp.min();
}

inline double minimum(Matrix A)
{
    valarray<double> temp(A.nRow());
    int i = 0;
    //#pragma omp parallel for default(shared) schedule(dynamic) private(i)
    for (i = 0; i < A.nRow(); i++)
    {
        temp[i] = A(i+1);
    }
    return temp.min();
}

inline double maximum (Matrix A)
{
    valarray<double> temp(A.nRow());
    int i = 0;
    //#pragma omp parallel for default(shared) schedule(dynamic) private(i)
    for (i = 0; i < A.nRow(); i++)
    {
        temp[i] = A(i+1);
    }
    return temp.max();
}

//find the maximum element of a matrix, A. OUTPUT: a 2 by 1 matrix of which the first element is the index of the maximum element in A
//and the second one is the value of this maximum element
Matrix maxx(Matrix A)
{
    int i, max_ind = 1;
    double max0 = 0.;
    max0 = A(1);
    for (i = 2; i <= A.nRow(); i++)
    {
        if (max0 < A(i))
        {
            max0 = A(i);
            max_ind = i;
        }
    }
    Matrix res(2,1);
    res(1) = max_ind;
    res(2) = max0;
    return res;
}

//find the minimum element of a matrix, A. OUTPUT: a 2 by 1 matrix of which the first element is the index of the minimum element in A
//and the second one is the value of this minimum element
Matrix minn(Matrix A)
{
    int i, min_ind = 1;
    double min0 = 0.;
    min0 = A(1);
    for (i = 2; i <= A.nRow(); i++)
    {
        if (min0 > A(i))
        {
            min0 = A(i);
            min_ind = i;
        }
    }
    Matrix res(2,1);
    res(1) = min_ind;
    res(2) = min0;
    return res;
}

Matrix pow (Matrix A, int n)//A is a N by 1 vector
{
    int i = 0, N = 0;
    N = A.nRow();
    Matrix res(N,1);
    #pragma omp parallel for default(shared) schedule(dynamic) private(i) if (N > 1000)
    for (i = 1; i <= N; i++)
    {
        res(i) = std::pow (A(i), n);
    }
    return res;
}

/* vectorize a matrix */
Matrix vec (const Matrix &A)
{
    int i = 1, j = 1, N = 1, M = 1;
    N = A.nRow();
    M = A.nCol();
    Matrix res(N*M,1);
    #pragma omp parallel for default(shared) schedule(dynamic) private(i,j) if ((M > 1000) && (N > 1000))
    for (j = 1; j <= M; j++)
    {
        for (i = 1; i <= N; i++)
        {
            res((j-1)*N+i) = A(i,j);
        }
    }
    return res;
}

/* vectorize a Shogun matrix */
SGVector<double> vec (const SGMatrix<double> &A)
{
    int i = 0, j = 0, N = 0, M = 0;
    N = A.num_rows;
    M = A.num_cols;
    SGVector<double> res(N*M);
    #pragma omp parallel for default(shared) schedule(dynamic) private(i,j) if ((M > 1000) && (N > 1000))
    for (j = 0; j < M; j++)
    {
        for (i = 0; i < N; i++)
        {
            res[j*N+i] = A(i,j);
        }
    }
    return res;
}

/* vectorize a symmetric Shogun matrix */
SGVector<double> vech(const SGMatrix<double> &A) {
	int nrow = A.num_rows;

	A.is_symmetric(); // check if this matrix is symmetric

	SGVector<double> res(nrow*(nrow+1)/2);
	int vindex = 0;
	for (int j = 0; j < nrow; ++j) {
		for (int i = j; i < nrow; ++i) {
			res[vindex] = A(i,j);
			vindex++;
		}
	}
	return res;
}

/* Compute the Kronecker product of two matrices */
Matrix Kronecker (const Matrix &A, const Matrix &B)
{
    int rowa, cola, rowb, colb;
    rowa = A.nRow();
    cola = A.nCol();
    rowb = B.nRow();
    colb = B.nCol();
    Matrix C(rowa*rowb,cola*colb);
    int i=1,j=1,k=1,l=1;
    #pragma omp parallel for default(shared) schedule(dynamic) private(i,j,k,l)  if ((rowa > 1000) && (cola > 1000))
    for(i = 1; i <= rowa; i++)
    {
        for(k = 1; k <= rowb; k++)
        {
            for(j = 1; j <= cola; j++)
            {
                for(l = 1; l <= colb; l++)
                {
                    C(rowb*(i-1)+k, colb*(j-1)+l) = A(i,j)*B(k,l);
                }
            }
        }
    }
    return C;
}

/* Compute the Kronecker product of two Shogun matrices */
SGMatrix<double> Kronecker (const SGMatrix<double> &A, const SGMatrix<double> &B)
{
    int rowa, cola, rowb, colb;
    rowa = A.num_rows;
    cola = A.num_cols;
    rowb = B.num_rows;
    colb = B.num_cols;
    SGMatrix<double> C(rowa*rowb,cola*colb);
    int i = 0,j = 0,k = 0,l = 0;
    #pragma omp parallel for default(shared) schedule(dynamic) private(i,j,k,l)  if ((rowa > 1000) && (cola > 1000))
    for(i = 0; i < rowa; i++)
    {
        for(k = 0; k < rowb; k++)
        {
            for(j = 0; j < cola; j++)
            {
                for(l = 0; l < colb; l++)
                {
                    C(rowb*i+k, colb*j+l) = A(i,j)*B(k,l);
                }
            }
        }
    }
    return C;
}


//sort the elements of a column vector, X, into ascending numerical order
void sort (Matrix &X)
{
    int i, N;
    N = X.nRow();
    gsl_vector * v = gsl_vector_alloc (N);
    for (i = 0; i < N; ++i)
    {
        gsl_vector_set (v, i, X(i+1));
    }
    gsl_sort_vector (v);
    for (i = 1; i <= N; ++i)
    {
        X(i) =  gsl_vector_get (v,i-1);
    }
    gsl_vector_free (v);
}

//sorts the N elements of X into ascending numerical order, while making the same rearrangement of Y
void sort2 (Matrix &X, Matrix &Y)
{
    int i, N;
    N = X.nRow();
    double *data1;
    data1 = (double *) malloc (N * sizeof(double));
    double *data2;
    data2 = (double *) malloc (N * sizeof(double));
    #pragma omp parallel for default(shared) schedule(dynamic) private(i)
    for (i = 0; i < N; i++)
    {
        data1[i] = X(i+1);
        data2[i] = Y(i+1);
    }
    gsl_sort2 (data1, 1, data2, 1, N);
    #pragma omp parallel for default(shared) schedule(dynamic) private(i)
    for (i = 1; i <= N; i++)
    {
        X(i) = data1[i-1];
        Y(i) = data2[i-1];
    }
    free(data1);
    free(data2);
}

//count the number of nonzero elements of a column vector (X)
int nonzero (Matrix X)
{
    int i, count_ = 0;
    auto eps = 0.000001;
    for (i = 1; i <= X.nRow(); i++)
    {
        if (std::fabs(X(i)) > eps)
        {
            count_ += 1;
        }
    }
    return count_;
}

double stopcdn (Matrix X_kp1, Matrix X_k)
{
    if (ENorm(X_k) > 1)
    {
        return pow(ENorm(X_kp1 - X_k)/ENorm(X_k), 2.);
    }
    else
    {
        return pow(ENorm(X_kp1 - X_k), 2.);
    }
}

//calculate quantiles. INPUT: a column vector of data (X), a cdf value (cdfValue) in [0,1]. OUTPUT: a double number
double quantile (const Matrix &X, const double cdfValue)
{
    int T = X.nRow();
    double data[T];
    for (int t = 0; t < T; t++)
    {
        data[t] = X(t + 1);
    }
    gsl_sort (data, 1, T);
    return gsl_stats_quantile_from_sorted_data (data, 1, T, cdfValue);
}

//calculate the matrix of sample cross-covariances between a T by dx matrix (X) and another T by dy matrix (Y). INPUT: lag is a lag length to be set.
//OUTPUT: a dx by dy matrix
Matrix cross_Cov (Matrix X, Matrix Y, int lag)
{
    int t = 1, i = 1, T = 1, dx = 1, dy = 1;
    T = X.nRow();
    dx = X.nCol();
    dy = Y.nCol();
    ASSERT_ (T == Y.nRow());
    Matrix meanX(dx, 1), meanY(dy, 1), x_t(dx, 1), y_t(1, dy), cov(dx, dy);
    meanX = mean (X);
    meanY = mean (Y);
    cov.set (0.);
    if ((lag >= 0) && (lag <= T-1))
    {
        for (t = 1+lag; t <= T; ++t)
        {
            for (i = 1; i <= dx; ++i)
            {
                x_t(i) = X(t, i) - meanX(i);
            }
            for (i = 1; i <= dy; ++i)
            {
                y_t(1,i) = Y(t-lag, i) - meanY(i);
            }
            cov = cov + (x_t * y_t);
        }
    }
    else if ((lag < 0) && (lag >= 1-T))
    {
        for (t = 1; t <= T+lag; ++t)
        {
            for (i = 1; i <= dx; ++i)
            {
                x_t(i) = X(t, i) - meanX(i);
            }
            for (i = 1; i <= dy; ++i)
            {
                y_t(1,i) = Y(t-lag, i) - meanY(i);
            }
            cov = cov + (x_t * y_t);
        }
    }
    else
    {
        cerr << "Warning: Invalid argument in `cross_Corr'!!!" << endl;
        abort();
    }
    return ((double) 1/T) * cov;
}

/* calculate the matrix of sample cross-covariances between a T by d1 Shogun matrix (eta1) and another T by d2 Shogun matrix (eta2).
OUTPUT: a d1 by d2 Shogun matrix */
SGMatrix<double> cross_cov(	const SGMatrix<double> &eta1, /*T by d1 matrix of residuals*/
							const SGMatrix<double> &eta2, /*T by d2 matrix of residuals*/
							int lag /*a lag*/) {
	int T = eta1.num_rows, d1 = eta1.num_cols, d2 = eta2.num_cols;
	ASSERT_(T == eta2.num_rows);

	int T0 = T - fabs(lag);
	SGMatrix<double> eta1a(T0, d1), eta2a(T0, d2);

	if (lag >= 0) {
		ASSERT_(lag <= T-1);
		for (int t = 0; t < T0; ++t) {
			for (int i = 0; i < d1; ++i) {
				eta1a(t, i) = eta1(t,i);
			}
			for (int i = 0; i < d2; ++i) {
				eta2a(t, i) = eta2(t+lag,i);
			}
		}
	}
	else {
		ASSERT_(lag >= 1-T);
		for (int t = 0; t < T0; ++t) {
			for (int i = 0; i < d1; ++i) {
				eta1a(t, i) = eta1(t-lag,i);
			}
			for (int i = 0; i < d2; ++i) {
				eta2a(t, i) = eta2(t,i);
			}
		}
	}


	SGMatrix<double> data(T0, 2), cov(2, 2), ccov(d1, d2);
	for (int i = 0; i < d1; ++i) {
		for (int j = 0; j < d2; ++j) {
			data.set_column( 0, eta1a.get_column(i) );
			data.set_column( 1, eta2a.get_column(j) );

			// compute covariance between each column of 'eta1' with each column of 'eta2'
			cov = Statistics::covariance_matrix(transpose_matrix(data), true);
			ccov(i,j) = cov(0,1);
		}
	}
	return ccov;
}

/* calculate the sample cross-correlation between two Shogun vectors */
double cross_corr(	const SGVector<double> &eta1, /*T by 1 vector of residuals*/
					const SGVector<double> &eta2, /*T by 1 vector of residuals*/
					int lag /*a lag*/) {
	int T = eta1.vlen;
	ASSERT_(T == eta2.vlen);

	int T0 = T - fabs(lag);
	SGVector<double> eta1a(T0), eta2a(T0);

	if (lag >= 0) {
		ASSERT_(lag <= T-1);
		for (int t = 0; t < T0; ++t) {
			eta1a[t] = eta1[t];
			eta2a[t] = eta2[t+lag];
		}
	}
	else {
		ASSERT_(lag >= 1-T);
		for (int t = 0; t < T0; ++t) {
			eta1a[t] = eta1[t-lag];
			eta2a[t] = eta2[t];
		}
	}

	SGMatrix<double> data(T0, 2);
	data.set_column(0, eta1a);
	data.set_column(1, eta2a);

	SGMatrix<double> cov = Statistics::covariance_matrix(transpose_matrix(data), true);

	return cov(0,1) / sqrt( cov(0,0) * cov(1,1) );
}


/* Create a Shogun diagonal matrix from a Shogun vector */
SGMatrix<double> create_diagonal_matrix(const SGVector<double> &a) {
	int N = a.vlen;
	SGMatrix<double> A(N, N);
	A.zero();
	for (int i = 0; i < N; ++i)
		A(i,i) = a[i];
	return A;
}

/* Calculate the sqrt of every element in a Shogun vector */
SGVector<double> sqrt_vector(const SGVector<double> &a) {
	int N = a.vlen;
	SGVector<double> b(N);
	for (int i = 0; i < N; ++i)
		b[i] = sqrt(a[i]);
	return b;
}

/* Calculate the inverse sqrt of every element in a Shogun vector */
SGVector<double> inv_sqrt_vector(const SGVector<double> &a) {
	int N = a.vlen;
	SGVector<double> b(N);
	for (int i = 0; i < N; ++i)
		b[i] = 1./sqrt(a[i]);
	return b;
}

/* Compute the standard deviation matrix from a variance-covariance matrix */
SGMatrix<double> sqrt_mat(const SGMatrix<double> &A) {

	ASSERT_(A.is_symmetric() == true); // check for the symmetry
	int N = A.num_rows;

	SGMatrix<double> A_sqrt(N, N), eigenvect_mat(N, N), eigenval_mat(N, N);
	A_sqrt.zero();
	SGVector<double> eigenval(N);
	eigenval_mat.zero();

	eigen_solver(A, eigenval, eigenvect_mat); // do the eigenvalue decomposition
	double min_eigenval = Math::min<double>(eigenval.vector, N);

	if (min_eigenval > 0) {
		for (int i = 0; i < N; ++i) {
			eigenval_mat(i, i) = sqrt(eigenval[i]);
		}
		A_sqrt = matrix_prod( eigenvect_mat, matrix_prod(eigenval_mat, eigenvect_mat, false, true) );
	}
	else {
		for (int i = 0; i < N; ++i) {
			A_sqrt(i, i) = sqrt( A(i,i) );
		}
	}
	return A_sqrt;
}

/* Compute the inverse of the standard deviation matrix from a variance-covariance matrix */
SGMatrix<double> inv_sqrt_mat(const SGMatrix<double> &A) {

	ASSERT_(A.is_symmetric() == true); // check for the symmetry
	int N = A.num_rows;

	SGMatrix<double> A_sqrt(N, N), eigenvect_mat(N, N), eigenval_mat(N, N);
	A_sqrt.zero();
	SGVector<double> eigenval(N);
	eigenval_mat.zero();

	eigen_solver(A, eigenval, eigenvect_mat); // do the eigenvalue decomposition
	double min_eigenval = Math::min<double>(eigenval.vector, N);

	if (min_eigenval > 0) {
		for (int i = 0; i < N; ++i) {
			eigenval_mat(i,i) = 1. / sqrt(eigenval[i]);
		}
		A_sqrt = matrix_prod( eigenvect_mat, matrix_prod(eigenval_mat, eigenvect_mat, false, true) );
	}
	else {
		for (int i = 0; i < N; ++i) {
			A_sqrt(i, i) = 1. / sqrt( A(i,i) );
		}
	}
	return A_sqrt;
}

//calculate the matrix of sample covariances between columns of a data matrix (X)
Matrix cov (const Matrix &X)
{
    int t = 1, i = 1, T = 1, dx = 1;
    T = X.nRow();
    dx = X.nCol();
    Matrix x_t(dx, 1), meanX(dx, 1), cov(dx, dx);
    x_t.set (0.);
    meanX = mean (X);
    cov.set(0.);
    for (t = 1; t <= T; ++t)
    {
        for (i = 1; i <= dx; ++i)
        {
            x_t(i) = X(t, i) - meanX(i);
        }
        cov = cov + (x_t * Tr(x_t));
    }
    return ((double) 1/T) * cov;
}

//check if a square matrix is symmetric
void isSymmetric (const Matrix &X) {
    bool test = true;
    auto eps = 0.00001;
    ASSERT_(X.nRow() == X.nCol()); //check for the matrix size consistency
    for (auto i = 1; i <= X.nRow(); ++i) {
        for (auto j = 1; j <= X.nCol(); ++j) {
            if (fabs(X(i,j) - X(j,i)) > eps) {
                test = false;
                break;
            }
        }
    }
    if (test == true) cout << "This matrix is indeed symmetric!" << endl;
    else cout << "This matrix is NOT symmetric!" << endl;
}

//normalize all the columns of a T0 by N matrix to the 0-1 range
void do_Norm (Matrix &X) {
    auto T0 = X.nRow(), N = X.nCol();
    valarray<double> temp(T0);
    for (auto j = 1; j <= N; ++j) {
        for (auto t = 0; t < T0; ++t) {
            temp[t] = X(t+1, j);
        }
        for (auto t = 1; t <= T0; ++t) {
            X(t,j) = (X(t,j) - temp.min()) / (temp.max() - temp.min());
        }
    }
}
void do_Norm (Matrix &X1, const Matrix &X) {
    auto T0 = X.nRow(), N = X.nCol();
    valarray<double> temp(T0);
    for (auto j = 1; j <= N; ++j) {
        for (auto t = 0; t < T0; ++t) {
            temp[t] = X(t+1, j);
        }
        for (auto t = 1; t <= T0; ++t) {
            X1(t,j) = (X(t,j) - temp.min()) / (temp.max() - temp.min());
        }
    }
}

/* Define a basis Shogun vector */
SGVector<double> base_vec(int i, int n) {
	SGVector<double> e(n);
	e.set_const(0.);
	e[i] = 1.;
	return e;
}

/* Get a subvector of a Shogun vector */
SGVector<double> get_subvector(	const SGVector<double> &X,
								int start_ind, /*start index (inclusive)*/
								int end_ind /*end index (inclusive)*/) {

	int T = X.vlen;
	ASSERT_(start_ind < end_ind && end_ind < T);

	SGVector<double> subvec(end_ind-start_ind+1);
	for (int i = start_ind; i <= end_ind; ++i)
		subvec[i-start_ind] = X[i];

	return subvec;
}

/* Get a row-based submatrix of a Shogun matrix */
SGMatrix<double> get_submatrix( const SGMatrix<double> &X,
								int start_ind, /*start row index (inclusive)*/
								int end_ind /*end row index (inclusive)*/ ) {

	int T = X.num_rows, dim = X.num_cols;
	ASSERT_(start_ind < end_ind && end_ind < T);

	SGMatrix<double> submat(end_ind-start_ind+1, dim);
	for (int t = start_ind; t <= end_ind; ++t) {
		for (int i = 0; i < dim; ++i){
			submat(t-start_ind, i) = X(t, i);
		}
	}
	return submat;
}






#endif













