#ifndef TESTS_H
#define TESTS_H

#include <matrix_ops2.h>

#include <boost/thread/thread.hpp>
//#include <Random.cpp>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/config.h>
//#include <shogun/base/init.h>
//#include <shogun/base/some.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/multiclass/tree/CARTree.h>
#include <shogun/util/iterators.h>


using namespace std;
using namespace shogun;
using namespace shogun::linalg;

void test_loop() {
	gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, 143523);

    int t, i, j, counter = 0;
    int T0 = 200, N = 4, M = N*(N-1)/2;

    SGMatrix<double> cov_mat(N, N), cov_mat_sqrt_inv(N, N), resid(N, T0), rand_noise(T0, M);
    //Matrix cov_mat(N, N), cov_mat_sqrt_inv(N, N), resid(N, T0), rand_noise(T0, M);

    for (int t = 0; t < T0; ++t) {
		for (int i = 0; i < M; ++i) {
			rand_noise(t,i) = gsl_ran_ugaussian(r);
		}
    }


    SGVector<double> row_vec1(N), row_vec(N);
    for (int i = 0; i < N; ++i)
		row_vec1[i] = gsl_ran_ugaussian(r);

	#if 0
	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,i,j,counter) firstprivate(cov_mat,cov_mat_sqrt_inv,row_vec)
	for (t = 0; t < T0; ++t) {
		counter = 0; // reset counter
		for (i = 0; i < N; ++i) {
			cov_mat(i, i) = 1.;
			for (j = 0; j < N; ++j) {
				#pragma omp critical
				{
					if (j > i) {
						cov_mat(i, j) = counter + i*j;
						cov_mat(j, i) = cov_mat(i, j);
						//#pragma omp atomic
						counter += 1;
					}
				}
			}
		}
		cov_mat_sqrt_inv = inv_sqrt_mat(cov_mat);
		row_vec = matrix_prod(cov_mat_sqrt_inv, row_vec1);
		resid.set_column(t, row_vec);
	}
	#endif

	// This could take longer than a serial script
	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,i,j) firstprivate(counter,cov_mat,cov_mat_sqrt_inv,row_vec)
	for (t = 0; t < T0; ++t) {
		#pragma omp critical
		{
			counter = 0; // reset counter
			for (i = 0; i < N; ++i) {
				cov_mat(i, i) = 1.;
				for (j = i+1; j < N; ++j) {
					cov_mat(i, j) = rand_noise(t, counter);
					cov_mat(j, i) = cov_mat(i, j);
					//sleep(10);
					//#pragma omp atomic
					counter += 1;
				}
			}

			cov_mat_sqrt_inv = inv_sqrt_mat(cov_mat);
			row_vec = matrix_prod(cov_mat_sqrt_inv, row_vec1);
			resid.set_column(t, row_vec);
		}
	}

	cout << "Mean of matrix is " << mean(resid) << endl;

	gsl_rng_free(r); // free up memory
}

/* Define a predicate to select nonzero elements from a SG vector */
bool isnotZero(double x) {
	return ( (x > 0) || (x < 0) );
}

void get_nonzeros() {

    gsl_rng * r;
    const gsl_rng_type * gen;//random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, 143523);

    SGVector<double> ranks(6);
    for (int i = 0; i < 4; ++i)
		ranks[i] = gsl_ran_ugaussian (r);
	ranks[4] = 0.;
	ranks[5] = 0.;
	ranks.display_vector();

	int max_index = Math::arg_max(ranks.vector, 1, 4);
	cout << max_index << endl;
	cout << ranks[max_index] << endl;


	SGVector<int> nonzero_indices = ranks.find_if(isnotZero);
	nonzero_indices.display_vector();
	SGVector<int> zero_indices = ranks.find(0.);
	zero_indices.display_vector();

//	SGVector<double> ranks_nonzero( nonzero_indices.size() );
//	for (int i = 0; i < nonzero_indices.size(); ++i)
//		ranks_nonzero[i] = ranks[i];
//	ranks_nonzero.display_vector();

	SGVector<double> ranks_nonzero( 6 - zero_indices.size() );
	int index = 0;
	for (int i = 0; i < 6; ++i) {
		if (ranks[i] != 0) {
			ranks_nonzero[index] = ranks[i];
			index += 1;
		}
	}
	ranks_nonzero.display_vector();


	gsl_rng_free(r);
}

void pass_by_ref (double &x) {
	x += 5; //modify the value of x
}

Matrix multi_norm (gsl_matrix *x, gsl_rng *r, unsigned long seed) { //x is a var-cov matrix
    gsl_vector *mean = gsl_vector_calloc (3); //vector of zero means
    gsl_matrix * L = gsl_matrix_calloc (3, 3);
    gsl_matrix_memcpy (L, x); //copy x into L
    gsl_linalg_cholesky_decomp1 (L); //get a Cholesky decomposition matrix
    gsl_vector *xi_vec = gsl_vector_calloc (3);
    const gsl_rng_type *gen; //random number generator
    gsl_rng_env_setup();
    gen = gsl_rng_taus;
    r = gsl_rng_alloc(gen);
    gsl_rng_set(r, seed);
	gsl_ran_multivariate_gaussian (r, mean, L, xi_vec); //call the multivariate normal random generator
	Matrix result(3, 1);
	result(1) = gsl_vector_get (xi_vec, 0);
	result(2) = gsl_vector_get (xi_vec, 1);
	result(3) = gsl_vector_get (xi_vec, 2);
	gsl_vector_free (mean); //free memory
	gsl_matrix_free (L);
    gsl_vector_free (xi_vec);
    return result;
}


auto pos = [](double x) { //a lambda function
    		if (x > 0.)
    		    return x;
    		else
    		    return 0.;
};

//test `class' template
template <class Dgp>
void reff(Matrix& X, Matrix& Y, Matrix alpha_x, Matrix alpha_y, Matrix beta_x, Matrix beta_y)
{
	Dgp obj(alpha_x, alpha_y, beta_x, beta_y);
	obj.gen_AR1 (X, Y, 0.5, 1000);
}

void test_task(int x) {
	int k = 0;
    #pragma omp parallel for
    for (int i = 0; i < 100000; i++) {

        if (i % 2){ /** Conditional based on i **/
            int c;
            #pragma omp atomic capture
            {
                c = k;
                k++;
            }

            usleep(1000 * ((float)std::rand() / RAND_MAX));

            #pragma omp task
            std::cout << c << std::endl; /** Some sort of task **/
        }
    }
    std::cout << k << std::endl; /** Some sort of task **/
    std::cout.flush();
}

double maximum1 (Matrix arr) {
	int i = 1, N = 1;
	N = arr.nRow();
	double maxx = 0.;
	#pragma omp parallel for
    for (i = 1 ; i <= N ; ++i ) {
        #pragma omp flush (max)
        if (arr(i) > maxx) {
           #pragma omp critical
           {
               if (arr(i) > maxx)
				maxx = arr(i); //critical is required to prevent a multiple access to the shared variable (max)
           }
        }
    }
    return maxx;
}

//test unique_ptr
double test_unique_ptr (std::unique_ptr <double> &x) {
	*x = pow(*x, 2.);
	return *x;
}

bool compare(Matrix X, Matrix Y)
{
	bool res = true;
	cout << "res = true: " << res << endl;
	for (int i = 1; i <= X.nRow(); i++)
	{
		for (int j = 1; j <= X.nCol(); j++)
		{
			if (X(i,j) != Y(i,j))
			{
				res = false;
				goto endloop;
			}
		}
	}
	endloop: return res;
}

double maximum (Matrix A, Matrix B)
{
	int i, N1;
	double res1 = 0., res2 = 0., max;
	N1 = A.nRow();
	#pragma omp parallel default(shared)
	{
		#pragma omp for reduction (+:res1) private(i) schedule(dynamic,CHUNK)
		for (i = 1; i <= N1; i++)
		{
			res1 += A(i);
			res2 += B(i);
		}
		#pragma omp critical //allow entry of only one thread at a time while others wait
		{
			if (res1 < res2)
			{
				max = res2;
			}
			else
			{
				max = res1;
			}
			cout << "(res1, res2) = " << res1 << ", " << res2 << endl;
		}
	}
	return max;
}

Matrix grad (Matrix x, Matrix u, Matrix par)
{
	int i = 0, j = 0, N_x = 0, G_u = 0, N_u = 0;
	N_x = x.nRow();
	G_u = u.nRow();
	N_u = u.nCol();
	Matrix res(N_x + G_u*N_u,1);
	for (i = 1; i <= N_x; i++)
	{
		res(i) = par(1) * x(i) + par(2) * pow(x(i), 2.);
	}
	for (i = 1; i <= G_u; i++)
	{
		for (j = 1; j <= N_u; j++)
		{
			res(N_x + (j-1)*G_u + i) = par(3) * u(i,j) - par(4) * pow(u(i,j), 3.) - 2;
		}
	}
	return res;
}


double rosenbrock (Matrix x, Matrix u, Matrix parms)
{
	int i = 0, j = 0, G = 0, N = 0, M = 0;
	M = x.nRow();
	G = u.nRow();
	N = u.nCol();
	double res = 0.;
	for (i = 1; i < G; i++)
	{
		for (j = 1; j <= N; j++)
		{
		    res += 100 * pow(u(i+1,j) - pow(u(i,j),2.), 2.) + pow(1-u(i,j),2.);
		}
	}
	for (i = 1; i <= M; i++)
	{
		res += pow (x(i) - parms(i), 2.);
	}
	return res;
}


void randm (Matrix& y, Matrix& x, unsigned long seed)
{
            gsl_rng * r;
            const gsl_rng_type * T;//random number generator
            gsl_rng_env_setup();
            T = gsl_rng_taus;
            r = gsl_rng_alloc (T);
            gsl_rng_set(r, seed);
            double test = 0.;
            test = gsl_rng_get(r);
            cout << test << endl;
            for (int i = 1; i <= y.nRow(); i++)
            {
               top:
			    x(i,1) = gsl_ran_gaussian (r,1);
                if ((x(i) >= 0.5) && (x(i) <= 0.8))
                {
                	y(i) = x(i);
                }
                else
                {
                	cout << "goto top..." << endl;
					goto top;
                }

            }
            gsl_rng_free (r);
            delete T;
}

Matrix RetValModi (Matrix x, valarray<int> parms)//the values of parms will be modified
{
	int N = x.nRow();
	int M = parms.size();
	for (int i = 1; i <= N; i++)
	{
		x(i) = pow(x(i),2.);
	}
	for (int i = 0; i < M; i++)
	{
		parms[i] = parms[i] + 2;
	}
	return x;
}

Matrix inverting_loops (Matrix A)
{
	int N, M, i, j;
	N = A.nRow();
	M = A.nCol();
	#pragma omp parallel for default(shared) schedule(static) private(i)
	for (j = 1; j <= M; j++)
	{
		for (i = 2; i <= N; i++)
		{
			A(i,j) = 2*A(i-1,j);//data dependences between rows
		}
	}
	return A;
}

Matrix parallel_break (Matrix A, Matrix B, Matrix C)
{
	int M = A.nRow();
	int i, j, low, high;
	#pragma omp parallel default(shared) private(i,j,low,high)
	{
	for (i = 1; i <= M; i++)
	{
		low = ((int) A(i));
		high = ((int) B(i));//make low and high private
		if (low > high)
		{
			#pragma omp single
			cout << "Exiting during iteration " << i << endl;
			break; //there is a break statement
		}
		//#pragma omp for nowait
		for (j = low; i <= high; ++j)
		{
			C(j) = (C(j)-A(i))/B(i);
		}
	}
    }
	return C;
}


Matrix func1 (int N)
{
	int i, j;
	double test = 0.;
	Matrix res(N,1);
	#pragma omp parallel for default(shared) schedule(dynamic,50) private(i,j)
	for (i = 1; i <= N; i++)
	{
		res(i) = log(i);
		for (j = 1; j <= i; j++)
		{
			res(i) *= j;
		}
	}
	test = test + 1;
	return res;
}

inline double func3 (double x)
{
	return sin (x);
}
inline double func4 (double x)
{
	return cos (x);
}
double func2 (int N)
{
	int i = 0;
	double x = 0., y = 0., a = 0., b = 0., q = 0.;
	gsl_rng * r;
    const gsl_rng_type * T;//random number generator
    gsl_rng_env_setup();
    T = gsl_rng_taus;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, N);
	x = gsl_ran_gaussian(r, 5);
	#pragma omp parallel for default(shared) schedule(dynamic,1) private(i)
	for (i = 1; i <= N; i++)
	{
		printf ("section 1 id = %d \n", omp_get_thread_num());
		y = gsl_ran_gaussian(r, 5);
        a = func3 (x);
        b = func4 (y);
        if (a < b)
        {
        	x = y;
        }
        else
        {
		    q = gsl_rng_uniform (r);
			if (q < exp(-(a-b)/100))
			{
			    	x = y;
			    	//cout << "number of moves: " << move << endl;
			}
	    }
	}
	cout << "x " << "y " << "a " << "b " << endl;
	cout << x << ", " << y << ", " << a << ", " << b << endl;
	if (a < b)
	{
		return a;
	}
	else
	{
		return b;
	}
	gsl_rng_free (r);
    delete T;
}

double triple_loop (int G, int N, int T)
{
    int g, i, j;
    double res1 = 0., a1 = 0;
	#pragma omp parallel default(shared)
    #pragma omp for reduction (+:res1) private (a1, g,i,j) schedule(dynamic,CHUNK)
	for (g = 1; g <= G; g++)
	{
		a1 = 0;
		for (i = 1; i <= N; i++)
	    {
		     for (j = 1; j <= T; j++)
		     {

				  a1 += i*j;
		     }
	    }
	    res1 += a1;
	}
	return res1;
}

double critical (int N, int M)
{
	int i, j;
	double res = 0.;
	Matrix tmp(N,1);
	#pragma omp parallel for default(shared) private(i,j) schedule(dynamic,CHUNK)
	for (i = 1; i <= N; i++)
	{
	  #pragma omp critical //allow entry of only one thread at a time while others wait
	  {
		for (j = 1; j <= M; j++)
		{
			tmp(i)  = i*j;
		}
		res = res + tmp(i);
	  }
	}
	return res;
}

void calculate(int N) /*The function that calculates the elements of a*/
{
   int i = 0;
   long w;
   long a[N];
   long sum = 0;

   /*forks off the threads and starts the work-sharing construct*/
   #pragma omp parallel for private(w) reduction(+:sum) schedule(static,1)//reduction(operation:variable) to calculate reduced sums
   for(i = 0; i < N; i++)
   {
       w = i*i;
       a[i] = std::sin(w);
       sum = sum + w*a[i];
       cout << i << endl;
       //boost::this_thread::sleep( boost::posix_time::seconds(20) );
   }
   printf("\n %li", sum);
}

Matrix g(int MAX_NUM, int N_THREADS, Matrix A, Matrix B)
{
 Matrix Y(MAX_NUM,MAX_NUM);
 int i, j, k;
 #pragma omp parallel for default(shared) schedule(static) num_threads(N_THREADS) private(i, j, k)
 for (i = 1; i <= MAX_NUM; i++) {
      for (j = 1; j <= MAX_NUM; j++) {
           for (k = 1; k <= MAX_NUM; k++)
                Y(i,j) += A(i,k) * B(k,j);
                //cout << "testing..." << endl;

       }
 }
 return Y;
}

Matrix ordered_loop (int N, int M)
{
	int i = 0, j = 0, tid;
	Matrix tmp(M,1), res(M, 1);
	#pragma omp parallel for ordered default(shared) private(i,j) schedule(dynamic,CHUNK)
	for (i = 1; i <= N; i++)
	{
		tid = omp_get_num_threads();
        cout << "number of threads = " << tid << endl;
		//#pragma ordered
		for (j = 1; j <= M; j++)
		{
			tmp(j) = j * j + i;
		}
		res = res + tmp;
	}
	return res;
}

Matrix loop1 (int N_THREADS, Matrix A)
{
	int i = 0;
	double b = 0.;
	  #pragma omp parallel for default(shared) schedule(dynamic,1) num_threads(N_THREADS) private(i)
	  for (i = 1; i <= A.nRow(); i++)
	  {
		#pragma omp atomic//when several threads are modifying b.
		b += A(i);
      }
      	//#pragma omp barrier//wait till all the threads complete.
        //#pragma omp single
        for (i = 1; i <= A.nRow(); i++)
        {
           cout<< A(i) <<endl;
        }
    return A;
}

Matrix sections1 (Matrix A, Matrix B)
{
int i = 0;
Matrix C(A.nRow(),1), D(A.nRow(),1);
cout << "B(1) = " << B(1) << endl;
//#pragma omp threadprivate (B)
#pragma omp parallel sections private(i) firstprivate(B) num_threads(2)//each thread creates its own instance of B, thus two
//threads are independent.
{
  //#pragma omp sections //nowait
    #pragma omp section
    {
     #pragma omp critical
     {
     printf ("section 1 id = %d \n", omp_get_thread_num());
     }
	 for (i = 1; i <= A.nRow(); i++)
     {
	 B(i) = 2*i;
	 D(i) = A(i);
     }
    }
    #pragma omp section
    {
     #pragma omp critical
     {
	 printf ("section 2 id = %d \n", omp_get_thread_num());
     }
	 for (i = 1; i <= A.nRow(); i++)
	    C(i) = A(i)+B(i);
    }
  }
cout << "B(1) = " << B(1) << endl;
return C;
}


Matrix use_barrier(Matrix A)
{
    int i, n, tid;
    n = A.nRow();
    Matrix B(n,1);
	#pragma omp parallel
    {
    	tid = omp_get_num_threads();
        cout << "number of threads = " << tid << endl;
        // Perform some computation.
        #pragma omp for
        for (i = 1; i <= n; i++)
            B(i) = i*i;

        // Print intermediate results.
        #pragma omp master
            for (i = 1; i <= n; i++)
            {
                cout << B(i) << endl;
                //cout << "..............." << endl;
            }

        // Wait.
        #pragma omp barrier

        // Continue with the computation.
        #pragma omp for
        for (i = 1; i <= n; i++)
            B(i) += i;
    }
    return B;
}

double use_paralell_for(Matrix A)
{
	int i, j, N, M;
	N = A.nRow();
	M = A.nCol();
	double res = 0.;
	#pragma omp parallel default(shared)
    {
          #pragma omp for private (i,j) schedule(dynamic,CHUNK) nowait
	      for (i = 1; i <= N; i++)
	      {
	      	  for (j = 1; j <= M; j++)
	      	  {
	      		   res = A(i,j);
	      	  }
		  }
          #pragma omp for private (i,j) schedule(dynamic,CHUNK) nowait
          for (i = 1; i <= N; i++)
          {
          	  for (j = 1; j <= M; j++)
          	  {
          	  	   #pragma omp atomic
				   res += A(i,j)/2;
          	  }
          }
    }
    return 0.;
}



/*double f (int& x)
{
       return pow(static_cast<double>(x),2)+0.5;
}*/

/* Paraboloid centered on (p[0],p[1]), with
   scale factors (p[2],p[3]) and minimum p[4] */

/*double
my_f (const gsl_vector *v, void *params)
{
  double x, y;
  double *p = (double *) params;

  x = gsl_vector_get(v, 0);
  y = gsl_vector_get(v, 1);

  return p[2] * (x - p[0]) * (x - p[0]) +
           p[3] * (y - p[1]) * (y - p[1]) + p[4];
}*/



#endif
