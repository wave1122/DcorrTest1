#ifndef _mgarchop
#define _mgarchop
#include <mgarch2.h>

using namespace std;

extern void (*aktfgv)(double *,double *,int *,int *,void *,Matrix&);

void log_likelihood_var (double* x, double *f, int *nparm, int* i, void* parms, Matrix& data)
{
     int N; // x consists of 4xN elements
     N = data.nCol();
     Matrix Y(2*N+2,1);
     for (int i = 1; i <= N; i++)
     {
         Y(i,1) = x[i]; //lambda
         Y(N+i,1) = x[N+i]; //beta
     }
     Y(2*N+1,1) = x[2*N+1];//kappa
     Y(2*N+2,1) = x[2*N+2];//zeta
     mgarch obj1;
     *f = - obj1.log_likelihood_var (Y, data);
     //cout << "log-likelihood-var: " << *f << endl;
}

void log_likelihood_cor (double* x, double *f, int *nparm, int* i, void* parm0, Matrix& data)
{
     int N;
     N = data.nCol();
     Matrix coeff(2*N+2,1);
     double *parms = (double *) parm0; 
     for (int i = 1; i <= 2*N; i++)
     {
         coeff(i,1) = parms[i];//lambda and beta
     }
     coeff(2*N+1,1) = parms[2*N+1];//kappa
     coeff(2*N+2,1) = parms[2*N+2];//zeta
     mgarch obj2;
     *f = - obj2.log_likelihood_cor (x[1], x[2], data, coeff);
     cout << "log-likelihood-cor: " << *f << endl;
}

Matrix log_likelihood_cor_sup (Matrix& coeff, Matrix& data)//maximize the correlation part of the log-likelihood
{
       FILE *out;
       VEKTOR_n mmin;
       VEKTOR_n mmax;
       int nparm;
       int m;
       int nsampl;
       int nsel;
       int nsig;
       TOMB_nx21 x0;
       int nc;
       int maxnc;
       double f0[21];
       int fe =1;
       int N;
       N = data.nCol();
       nparm = 2;
       double* parms = new double[2*N+3];
       for (int i = 1; i <= 2*N+2; i++)
       {
            parms[i] = coeff(i,1);
       }
       if((x0  = (TOMB_nx21)malloc(sizeof(double)*(nparm+1)*21))==NULL) exit(3);
       if((mmin=(double*)malloc( sizeof(double)*(nparm+1) ))==NULL) exit(3);
       if((mmax=(double*)malloc( sizeof(double)*(nparm+1) ))==NULL) exit(3);
       for (int i = 1; i <= nparm; i++)
       {
           mmax[i] = 0.5;
           mmin[i] = 0;
           cout << mmin[i] << "," << mmax[i] << endl;
       }
       m = 1;
       nsampl = 100;
       nsel=20;
       nsig=6;
       maxnc = 1;
       aktfgv = log_likelihood_cor;
       if ((out = fopen("dcc.txt", "w+")) == NULL) {
             fprintf(out, "Cannot open output file.\n");
             exit(1);
        };
       global(mmin,mmax,&nparm,&m,parms,data,&nsampl,&nsel,out,&nsig,x0,&nc,f0,&fe,maxnc);
       Matrix res(2,1);
       res(1,1) = (*x0)[1][1];//a
       res(2,1) = (*x0)[2][1];//b
       fclose(out);
       free(mmin);
       free(mmax);
       free(x0);
       return res;   
}    


Matrix log_likelihood_var_sup (Matrix& data) //maximize the garch part of the log-likelihood
{
       FILE *out;
       VEKTOR_n mmin;
       VEKTOR_n mmax;
       int nparm;
       int m;
       int nsampl;
       int nsel;
       int nsig;
       TOMB_nx21 x0;
       int nc;
       int maxnc;
       double f0[21];
       int fe =1, N = 0;
       N = data.nCol();
       nparm = 2*N+2;
       if((x0  = (TOMB_nx21)malloc(sizeof(double)*(nparm+1)*21))==NULL) exit(3);
       if((mmin=(double*)malloc( sizeof(double)*(nparm+1) ))==NULL) exit(3);
       if((mmax=(double*)malloc( sizeof(double)*(nparm+1) ))==NULL) exit(3);
       for (int i = 1; i <= N; i++)
       {
           mmax[i] = 10;
           mmin[i] = -10;
           cout << mmin[i] << "," << mmax[i] << endl;
       }
       for (int i = N+1; i <= 2*N; i++)
       {
           mmax[i] = 1;
           mmin[i] = -1;
           cout << mmin[i] << "," << mmax[i] << endl;
       }
       mmin[2*N+1] = 0;
       mmax[2*N+1] = 0.5;
       cout << mmin[2*N+1] << "," << mmax[2*N+1] << endl;
       mmin[2*N+2] = 0;
       mmax[2*N+2] = 0.5;
       cout << mmin[2*N+2] << "," << mmax[2*N+2] << endl;
       m = 1;
       nsampl = 1000;
       nsel=20;
       nsig=6;
       maxnc = 1;
       aktfgv = log_likelihood_var;
       if ((out = fopen("garch.txt", "w+")) == NULL) {
             fprintf(out, "Cannot open output file.\n");
             exit(1);
        };
       double* parms;
       global(mmin,mmax,&nparm,&m,parms,data,&nsampl,&nsel,out,&nsig,x0,&nc,f0,&fe,maxnc);
       Matrix res(2*N+2,1);
       for (int i = 1; i <= 2*N+2; i++) 
       {
               res(i,1) = (*x0)[i][1];
       }
       fclose(out);
       free(mmin);
       free(mmax);
       free(x0);
       delete parms;
       return res;   
}    


#endif
