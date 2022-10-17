#include        <stdio.h>
#include        <stdlib.h>
#include        <math.h>
#include        "globdef.h"
//using namespace std;

static  double  /*zero = 0.0, one = 1.0, */ ax=0.1,p1=0.1,half=0.5,
                reps=1.1921e-07,five=5.0,
                seven=7.0,twelve=12.0;

void local (int *m,int *n,double *eps,int *maxfn,VEKTOR_n x,double *f,
            int *nfev,void* parms,Matrix& data,VEKTOR_n minx,VEKTOR_n maxx)

/*                        SPECIFICATIONS FOR ARGUMENTS                     */
{
/*                        SPECIFICATIONS FOR LOCAL VARIABLES               */

               int      ig,igg,is,idiff,ir,ij,i,j,nm1,jj,jp1,l,kj,
                          k,link,itn,ii,im1,jnt,np1,jb,nj,ier;

               //int      test; 
               double     hh,hjj,v,df,relx,gs0,diff,aeps,alpha,ff,
                          tot,f1,f2,z,gys,dgs,sig,zz/*gnrm*/,hhh,ghh,
						  iopt/*d2*/  /* ,h[180300] */ ;
	           static double h[180300];

               VEKTOR_n  g,w; 
               if((g = (VEKTOR_n)malloc(sizeof(double)*(*n+1)))==NULL) exit(3);
               if((w = (VEKTOR_n)malloc(sizeof(double)*(*n+1)*3))==NULL) exit(3);

/*                        INITIALIZATION                                   */
/*                        FIRST EXECUTABLE STATEMENT                       */
                iopt = 0;
/*            IOPT     - OPTIONS SELECTOR. (INPUT)                         */
/*                  IOPT = 0 CAUSES LOCAL     TO INITIALIZE THE            */
/*                    HESSIAN MATRIX H TO     THE IDENTITY MATRIX.         */
/*                  IOPT = 1 INDICATES THAT H HAS     BEEN INITIALIZED     */
/*                    BY THE USER     TO A POSITIVE DEFINITE MATRIX.       */
/*                  IOPT = 2 CAUSES LOCAL     TO COMPUTE THE DIAGONAL      */
/*                    VALUES OF THE HESSIAN MATRIX AND SET H TO            */
/*                    A DIAGONAL MATRIX CONTAINING THESE VALUES.           */
/*                  IOPT = 3 CAUSES LOCAL     TO COMPUTE AN ESTIMATE       */
/*                    OF THE HESSIAN IN H.                                 */
                ier = 0;
                hh = sqrt(reps);
                ig = *n;
                igg = *n+*n;
                is = igg;
                idiff = 1;
                ir = *n;
                w[1] = -one;
                w[2] = zero;
                w[3] = zero;
/*                        EVALUATE FUNCTION AT     STARTING POINT          */
                for(i=1;i<=*n;i++)
                     g[i] = x[i];
                fun (g,f,n,m,parms,data,minx,maxx);
                *nfev = 1;
                if (iopt == 1) goto label_45;
/*                        SET OFF-DIAGONAL ELEMENTS OF     H TO 0.0        */
                if (*n == 1) goto label_25;
                ij = 2;
                for(i=2;i<=*n;i++) {
                     for(j=2;j<=i;j++) {
                          h[ij] = zero;
                          ij = ij+1;
                     }
                     ij = ij+1;
                }
                if (iopt != 0) goto label_25;
/*                        SET DIAGONAL     ELEMENTS OF H TO one            */
                ij = 0;
                for(i=1;i<=*n;i++) {
                     ij = ij+i;
                     h[ij] = one;
                }
                goto label_80;
/*                        GET DIAGONAL     ELEMENTS OF HESSIAN             */
  label_25:     im1 = 1;
                nm1 = 1;
                np1 = *n+1;
                for(i=2;i<=np1;i++) {
                     hhh = hh*MaX(fabs(x[im1]),ax);
                     g[im1] = x[im1]+hhh;
                     fun (g,&f2,n,m,parms,data,minx,maxx);
                     g[im1] = g[im1]+hhh;
                     fun (g,&ff,n,m,parms,data,minx,maxx);
                     h[nm1] = (ff-f2+*f-f2)/(hhh*hhh);
                     g[im1] = x[im1];
                     im1 = i;
                     nm1 = i+nm1;
                }
                *nfev = *nfev+*n+*n;
                if ((iopt != 3) || (*n == 1)) goto label_45;
/*                        GET THE REST     OF THE HESSIAN                  */
                jj = 1;
                ii = 2;
                for(i=2;i<=*n;i++) {
                     ghh = hh*MaX(fabs(x[i]),ax);
                     g[i] = x[i]+ghh;
                     fun (g,&f2,n,m,parms,data,minx,maxx);
                     for(j=1;j<=jj;j++) {
                          hhh = hh*MaX(fabs(x[j]),ax);
                          g[j] = x[j]+hhh;
                          fun (g,&ff,n,m,parms,data,minx,maxx);
                          g[i] = x[i];
                          fun (g,&f1,n,m,parms,data,minx,maxx);
/*          H(II) = (FF-F1-F2+F)*SQREPS                                    */
                          h[ii] = (ff-f1-f2+*f)/(hhh*ghh);
                          ii = ii+1;
                          g[j] = x[j];
                     }
                     jj = jj+1;
                     ii = ii+1;
                }
                *nfev = *nfev+((*n**n-*n)/2);
/*                        FACTOR H TO L*D*L-TRANSPOSE                      */
  label_45:     ir = *n;
                if (*n  > 1) goto label_50;
                if (h[1]  > zero)     goto label_80;
                h[1] = zero;
                ir = 0;
                goto label_75;
  label_50:     nm1 = *n-1;
                jj = 0;
                for(j=1;j<=*n;j++) {
                     jp1 = j+1;
                     jj = jj+j;
                     hjj = h[jj];
                     if (hjj  > zero) goto label_55;
                     h[jj] = zero;
                     ir = ir-1;
                     continue;
  label_55:          if (j == *n) continue;
                     ij = jj;
                     l = 0;
                     for(i=jp1;i<=*n;i++) {
                          l = l+1;
                          ij = ij+i-1;
                          v = h[ij]/hjj;
                          kj = ij;
                          for(k=i;k<=*n;k++) {
                               h[kj+l] = h[kj+l]-h[kj]*v;
                               kj = kj+k;
                          }
                          h[ij] = v;
                     }
                }
  label_75:     if (ir == *n) goto label_80;
                ier = 129;
                goto label_9000;
  label_80:     itn = 0;
                df = -one;
/*                        EVALUATE GRADIENT W(IG+I),I=1,...,N              */
  label_85:     link = 1;
                goto label_260;
  label_90:     if (*nfev >= *maxfn) goto label_9000;//label_235;
/*                        BEGIN ITERATION LOOP                             */
                itn = itn+1;
                for(i=1;i<=*n;i++)
                     w[i] = -w[ig+i];
/*                        DETERMINE SEARCH DIRECTION W                     */
/*                          BY     SOLVING     H*W = -G WHERE              */
/*                          H = L*D*L-TRANSPOSE                            */
                if (ir  < *n) goto label_125;
/*                        N .EQ. 1                                         */
                g[1] = w[1];
                if (*n  > 1) goto label_100;
                w[1] = w[1]/h[1];
                goto label_125;
/*                        N .GT. 1                                         */
  label_100:    ii = 1;
/*                        SOLVE L*W = -G                                   */
                for(i=2;i<=*n;i++) {
                     ij = ii;
                     ii = ii+i;
                     v = w[i];
                     im1 = i-1;
                     for(j=1;j<=im1;j++) {
                          ij = ij+1;
                          v = v-h[ij]*w[j];
                     }
                     g[i] = v;
                     w[i] = v;
                }
/*                        SOLVE (D*LT)*Z = W WHERE                         */
/*                                     LT = L-TRANSPOSE                    */
                w[*n] = w[*n]/h[ii];
                jj = ii;
                nm1 = *n-1;
                for(nj=1;nj<=nm1;nj++) {
/*                        J = N-1,N-2,...,1                                */
                     j = *n-nj;
                     jp1 = j+1;
                     jj = jj-jp1;
                     v = w[j]/h[jj];
                     ij = jj;
                     for(i=jp1;i<=*n;i++) {
                          ij = ij+i-1;
                          v = v-h[ij]*w[i];
                     }
                     w[j] = v;
                }
/*                        DETERMINE STEP LENGTH ALPHA                      */
  label_125:    relx = zero;
                gs0 = zero;
                for(i=1;i<=*n;i++) {
                     w[is+i] = w[i];
                     diff = fabs(w[i])/MaX(fabs(x[i]),ax);
                     relx = MaX(relx,diff);
                     gs0 = gs0+w[ig+i]*w[i];
                }
                if (relx == zero)     goto label_230;
                aeps = *eps/relx;
                ier = 130;
                if (gs0 >= zero) goto label_230;
                if (df == zero) goto label_230;
                ier = 0;
                alpha = (-df-df)/gs0;
                if (alpha <= zero) alpha = one;
                alpha = MiN(alpha,one);
                if (idiff == 2) alpha = MaX(p1,alpha);
                ff = *f;
                tot = zero;
                jnt = 0;
/*                        SEARCH ALONG     X+ALPHA*W                       */
  label_135:    if (*nfev >= *maxfn) goto label_9000; //label_235;
                for(i=1;i<=*n;i++)
                     w[i] = x[i]+alpha*w[is+i];
                     fun (w,&f1,n,m,parms,data,minx,maxx);
                *nfev = *nfev+1;
                if (f1 >= *f) goto label_165;
                f2 = *f;
                tot = tot+alpha;
  label_145:    ier = 0;
                *f = f1;
                for(i=1;i<=*n;i++)
                    x[i] = w[i];
                if (jnt-1 <0)      goto label_155;
                else if (jnt-1==0) goto label_185;
                else               goto label_190;
  label_155:    if (*nfev >= *maxfn) goto label_9000;//label_235;
                for(i=1;i<=*n;i++)
                     w[i] = x[i]+alpha*w[is+i];
                     fun (w,&f1,n,m,parms,data,minx,maxx);
                *nfev = *nfev+1;
                if (f1 >= *f) goto label_190;
                if ((f1+f2 >= *f+*f) && (seven*f1+five*f2  > twelve*(*f))) jnt = 2;
                tot = tot+alpha;
                alpha = alpha+alpha;
                goto label_145;
  label_165:    if ((*f == ff) && (idiff == 2) && (relx  > *eps)) ier = 130;
                if (alpha  < aeps) goto label_230;
                if (*nfev >= *maxfn) goto label_9000;//label_235;
                alpha = half*alpha;
                for(i=1;i<=*n;i++)
                     w[i] = x[i]+alpha*w[is+i];
                     fun (w,&f2,n,m,parms,data,minx,maxx);
                *nfev = *nfev+1;
                if (f2 >= *f) goto label_180;
                tot = tot+alpha;
                ier = 0;
                *f = f2;
                for(i=1;i<=*n;i++)
                     x[i] = w[i];
                goto label_185;
  label_180:    z = p1;
                if (f1+*f  > f2+f2) z = one+half*(*f-f1)/(*f+f1-f2-f2);
                z = MaX(p1,z);
                alpha = z*alpha;
                jnt = 1;
                goto label_135;
  label_185:    if (tot  < aeps) goto label_230;
  label_190:    alpha = tot;
/*                        SAVE     OLD GRADIENT                            */
                for(i=1;i<=*n;i++)
                     w[i] = w[ig+i];
/*                        EVALUATE GRADIENT W(IG+I), I=1,...,N             */
                link = 2;
                goto label_260;
  label_200:    if(*nfev >= *maxfn) goto label_9000;//label_235;
                gys = zero;
                for(i=1;i<=*n;i++) {
                     gys = gys+w[ig+i]*w[is+i];
                     w[igg+i] = w[i];
                }
                df = ff-*f;
                dgs = gys-gs0;
                if (dgs <= zero) goto label_90;
                if (dgs+alpha*gs0  > zero) goto label_215;
/*                        UPDATE HESSIAN H USING                           */
/*                          COMPLEMENTARY DFP FORMULA                      */
                sig = one/gs0;
                ir = -ir;
                update (h,n,w,&sig,g,&ir,0,&zero);
                for(i= 1;i<=*n;i++)
                     g[i] = w[ig+i]-w[igg+i];
                sig = one/(alpha*dgs);
                ir = -ir;
                                update (h,n,g,&sig,w,&ir,0,&reps);
                goto label_90;
/*                        UPDATE HESSIAN USING                             */
/*                          DFP FORMULA                                    */
  label_215:    zz = alpha/(dgs-alpha*gs0);
                sig = -zz;
                                update (h,n,w,&sig,g,&ir,0,&reps);
                z = dgs*zz-one;
                for(i=1;i<=*n;i++)
                     g[i] = w[ig+i]+z*w[igg+i];
                sig = one/(zz*dgs*dgs);
                                update (h,n,g,&sig,w,&ir,0,&zero);
                goto label_90;
/*                        MaxFN FUNCTION EVALUATIONS                       */
  label_230:    if (idiff == 2) goto label_235;
/*                        CHANGE TO CENTRAL DIFFERENCES                    */
                idiff = 2;
                goto label_85;
  label_235:    if ((relx  > *eps) && (ier == 0)) goto label_85;
/*                        COMPUTE H = L*D*L-TRANSPOSE AND                  */
/*                          OUTPUT                                         */
                if (*n == 1) goto label_9000;
                np1 = *n+1;
                nm1 = *n-1;
                jj = (*n*(np1))/2;
                for(jb=1;jb<=nm1;jb++) {
                     jp1 = np1-jb;
                     jj = jj-jp1;
                     hjj = h[jj];
                     ij = jj;
                     l = 0;
                     for(i=jp1;i<=*n;i++) {
                          l = l+1;
                          ij = ij+i-1;
                          v = h[ij]*hjj;
                          kj = ij;
                          for(k=i;k<=*n;k++) {
                               h[kj+l] = h[kj+l]+h[kj]*v;
                               kj = kj+k;
                          }
                          h[ij] = v;
                     }
                     hjj = h[jj];
                }
                goto label_9000;
/*                         EVALUATE GRADIENT                               */
  label_260:    if (idiff == 2) goto label_270;
/*                        FORWARD DIFFERENCES                              */
/*                          GRADIENT =     W(IG+I), I=1,...,N              */
                for(i=1;i<=*n;i++) {
                     z = hh*MaX(fabs(x[i]),ax);
                     zz = x[i];
                     x[i] = zz+z;
                     fun (x,&f1,n,m,parms,data,minx,maxx);
                     w[ig+i] = (f1-*f)/z;
                     x[i] = zz;
                }
                *nfev = *nfev+*n;
                if (link==1) goto label_90;
                 else if (link==2) goto label_200;
/*                              CENTRAL DIFFERENCES                        */
/*                          GRADIENT =     W(IG+I), I=1,...,N              */
  label_270:    for(i=1;i<=*n;i++) {
                     z = hh*MaX(fabs(x[i]),ax);
                     zz = x[i];
                     x[i] = zz+z;
                     fun (x,&f1,n,m,parms,data,minx,maxx);
                     x[i] = zz-z;
                     fun (x,&f2,n,m,parms,data,minx,maxx);
                     w[ig+i] = (f1-f2)/(z+z);
                     x[i] = zz;
                }
                *nfev = *nfev+*n;
                if (link==1) goto label_90;
                 else if (link==2) goto label_200;
/*                        RETURN                                           */
  label_9000:     free(g);
                  free(w);
                  return;
}


