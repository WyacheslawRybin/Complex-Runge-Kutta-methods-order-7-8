/* Complex Cash-Karp (cascade) search for projected explicit RK methods.
   Port of crksearch.c of the order 7-8 archive (raw coordinates only) with a second
   weight vector bh on the leading MH stages:
     level 1:  gamma(t) Re(b^T  Phi_t)        - 1 = 0,  trees 0..ma-1  (all S stages)
     level 2:  gamma(t) Re(bh^T Phi_t[:MH])   - 1 = 0,  trees 0..mb-1  (leading MH stages)
   Trees are ordered by order, so "order <= q" is a prefix of the list (dagdata.h).
   x = [Re a (NA), Re b (S), Re bh (MH), Im a (NA), Im b (S), Im bh (MH)].
   Solver: the session-6 LM of crksearch.c (4 damping lanes, AVX2 over the lanes) and the
   session-7 Gauss-Newton endgame with central differences for points that stop in
   [1e-12, 1e-5); staged seeding: LM on lower orders first, then on the target.
   Build:  cl /O2 /fp:fast /arch:AVX2 /openmp /I cfgSP /DMH=m cascsearch.c
   Usage:  cascsearch nstart seed STAGES ITERS out.bin [seeds.bin nseed spread]
           STAGES = "ma:mb,ma:mb,..." tree counts per stage, the last one is the target;
           ITERS  = "it,it,..."; a start goes on to the next stage only if max|r| < 1e-10.
   Prints the number of starts that reached the last stage (pass1), hits (max|r| < 1e-12)
   and a histogram of final residuals; hits go to out.bin, near-misses to out.bin.near. */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "dagdata.h"

#ifndef MH
#define MH 0
#endif
#define NX (NP + 2*MH)
#define HALF (NX/2)
#define VW 4
#define MEXT (2*M)

static double g_fdh = 1e-7, g_rc = 1e-12, g_fdc = 2e-6, g_polish = 1e-5;

static inline void cmac(__m256d ar,__m256d ai,__m256d br,__m256d bi,__m256d*cr,__m256d*ci){
    *cr=_mm256_fmadd_pd(ar,br,*cr); *cr=_mm256_fnmadd_pd(ai,bi,*cr);
    *ci=_mm256_fmadd_pd(ar,bi,*ci); *ci=_mm256_fmadd_pd(ai,br,*ci);
}
static inline void cmul(__m256d ar,__m256d ai,__m256d br,__m256d bi,__m256d*cr,__m256d*ci){
    __m256d tr=_mm256_mul_pd(ar,br); tr=_mm256_fnmadd_pd(ai,bi,tr);
    __m256d ti=_mm256_mul_pd(ar,bi); ti=_mm256_fmadd_pd(ai,br,ti);
    *cr=tr; *ci=ti;
}

typedef struct {
    __m256d *wr,*wi;
    __m256d Ar[S*S],Ai[S*S],brr[S],bii[S],hr[MH+1],hi[MH+1],pr[S],pii[S];
    double *Rall,*J,*H,*g,*r0,*step,*xtry,*delta,*Xbuf,*Rbuf,*Q;
} work_t;

static work_t* work_new(void){
    work_t*W=(work_t*)_aligned_malloc(sizeof(work_t),64); memset(W,0,sizeof(work_t));
    W->wr=(__m256d*)_aligned_malloc(sizeof(__m256d)*NW*S,64);
    W->wi=(__m256d*)_aligned_malloc(sizeof(__m256d)*NW*S,64);
    W->Rall=(double*)_aligned_malloc(sizeof(double)*(NX+VW)*MEXT,64);
    W->J   =(double*)_aligned_malloc(sizeof(double)*MEXT*NX,64);
    W->H   =(double*)_aligned_malloc(sizeof(double)*NX*NX,64);
    W->g   =(double*)_aligned_malloc(sizeof(double)*NX,64);
    W->r0  =(double*)_aligned_malloc(sizeof(double)*MEXT,64);
    W->step=(double*)_aligned_malloc(sizeof(double)*NX*VW,64);
    W->xtry=(double*)_aligned_malloc(sizeof(double)*NX*VW,64);
    W->delta=(double*)_aligned_malloc(sizeof(double)*NX,64);
    W->Xbuf=(double*)_aligned_malloc(sizeof(double)*NX*VW,64);
    W->Rbuf=(double*)_aligned_malloc(sizeof(double)*MEXT*VW,64);
    W->Q   =(double*)_aligned_malloc(sizeof(double)*(MEXT+NX)*(NX+1),64);
    return W;
}

/* Evaluate VW parameter vectors. X: VW x NX row-major. out: VW x MEXT row-major, rows ma+mb. */
static void resid4(work_t*W,const double*X,int ma,int mb,double*out){
    const __m256d zero=_mm256_setzero_pd(), one=_mm256_set1_pd(1.0);
    #define LDRE(k) _mm256_set_pd(X[3*NX+(k)],X[2*NX+(k)],X[NX+(k)],X[(k)])
    #define LDIM(k) _mm256_set_pd(X[3*NX+HALF+(k)],X[2*NX+HALF+(k)],X[NX+HALF+(k)],X[HALF+(k)])
    for(int k=0;k<S*S;k++){ W->Ar[k]=zero; W->Ai[k]=zero; }
    for(int k=0;k<NA;k++){ int i=AI[k],j=AJ[k]; W->Ar[i*S+j]=LDRE(k); W->Ai[i*S+j]=LDIM(k); }
    for(int i=0;i<S;i++){ W->brr[i]=LDRE(NA+i); W->bii[i]=LDIM(NA+i); }
    for(int i=0;i<MH;i++){ W->hr[i]=LDRE(NA+S+i); W->hi[i]=LDIM(NA+S+i); }
    #undef LDRE
    #undef LDIM
    const int mt = ma>mb ? ma : mb;
    for(int t=0;t<mt;t++){
        int c0=CHPTR[t],c1=CHPTR[t+1];
        if(c0==c1){ for(int i=0;i<S;i++){ W->pr[i]=one; W->pii[i]=zero; } }
        else{
            const __m256d*ur=W->wr+(size_t)CHIDX[c0]*S,*ui=W->wi+(size_t)CHIDX[c0]*S;
            for(int i=0;i<S;i++){ W->pr[i]=ur[i]; W->pii[i]=ui[i]; }
            for(int c=c0+1;c<c1;c++){
                const __m256d*vr=W->wr+(size_t)CHIDX[c]*S,*vi=W->wi+(size_t)CHIDX[c]*S;
                for(int i=0;i<S;i++) cmul(W->pr[i],W->pii[i],vr[i],vi[i],&W->pr[i],&W->pii[i]);
            }
        }
        const __m256d gm=_mm256_set1_pd(GAM[t]);
        double tmp[VW];
        if(t<ma){
            __m256d acc=zero;
            for(int i=0;i<S;i++){ acc=_mm256_fmadd_pd(W->brr[i],W->pr[i],acc); acc=_mm256_fnmadd_pd(W->bii[i],W->pii[i],acc); }
            _mm256_storeu_pd(tmp,_mm256_fmsub_pd(gm,acc,one));
            for(int l=0;l<VW;l++) out[(size_t)l*MEXT+t]=tmp[l];
        }
        if(t<mb){
            __m256d acc=zero;
            for(int i=0;i<MH;i++){ acc=_mm256_fmadd_pd(W->hr[i],W->pr[i],acc); acc=_mm256_fnmadd_pd(W->hi[i],W->pii[i],acc); }
            _mm256_storeu_pd(tmp,_mm256_fmsub_pd(gm,acc,one));
            for(int l=0;l<VW;l++) out[(size_t)l*MEXT+ma+t]=tmp[l];
        }
        int o=WOUT[t];
        if(o>=0){
            __m256d*qr=W->wr+(size_t)o*S,*qi=W->wi+(size_t)o*S;
            qr[0]=zero; qi[0]=zero;
            for(int i=1;i<S;i++){
                __m256d cr=zero,ci=zero;
                for(int j=0;j<i;j++) cmac(W->Ar[i*S+j],W->Ai[i*S+j],W->pr[j],W->pii[j],&cr,&ci);
                qr[i]=cr; qi[i]=ci;
            }
        }
    }
}
static void resid1(work_t*W,const double*x,int ma,int mb,double*out){
    for(int l=0;l<VW;l++) memcpy(W->Xbuf+(size_t)l*NX,x,sizeof(double)*NX);
    resid4(W,W->Xbuf,ma,mb,W->Rbuf);
    memcpy(out,W->Rbuf,sizeof(double)*(ma+mb));
}
static double fnorm(const double*r,int m){ double s=0; for(int i=0;i<m;i++) s+=r[i]*r[i]; return s; }
static double fmaxabs(const double*r,int m){ double s=0; for(int i=0;i<m;i++){ double a=fabs(r[i]); if(a>s)s=a; } return s; }

static int chol_solve(const double*H,const double*g,double lam,int n,double*d,double*Lw){
    for(int i=0;i<n;i++) for(int j=0;j<=i;j++) Lw[i*n+j]=H[i*n+j];
    for(int i=0;i<n;i++){ double dg=H[i*n+i]; if(dg<1e-14) dg=1e-14; Lw[i*n+i]+=lam*dg; }
    for(int i=0;i<n;i++){
        for(int j=0;j<i;j++){
            double s=Lw[i*n+j];
            for(int k=0;k<j;k++) s-=Lw[i*n+k]*Lw[j*n+k];
            Lw[i*n+j]=s/Lw[j*n+j];
        }
        double s=Lw[i*n+i];
        for(int k=0;k<i;k++) s-=Lw[i*n+k]*Lw[i*n+k];
        if(!(s>0.0)) return 1;
        Lw[i*n+i]=sqrt(s);
    }
    for(int i=0;i<n;i++){ double s=-g[i]; for(int k=0;k<i;k++) s-=Lw[i*n+k]*d[k]; d[i]=s/Lw[i*n+i]; }
    for(int i=n-1;i>=0;i--){ double s=d[i]; for(int k=i+1;k<n;k++) s-=Lw[k*n+i]*d[k]; d[i]=s/Lw[i*n+i]; }
    return 0;
}

/* min |[J; sqrt(lam D)] d + [r; 0]| by Householder QR (D = diag(J^T J), or I if H == NULL) */
static int qr_solve(const double*J,const double*r,const double*H,double lam,int m,int n,double*d,double*Q){
    const int R=m+n;
    for(int k=0;k<n;k++){ double*col=Q+(size_t)k*R;
        for(int t=0;t<m;t++) col[t]=J[(size_t)t*n+k];
        for(int t=0;t<n;t++) col[m+t]=0.0;
        double dg=H?H[k*n+k]:1.0; if(dg<1e-14) dg=1e-14; col[m+k]=sqrt(lam*dg); }
    double*y=Q+(size_t)n*R;
    for(int t=0;t<m;t++) y[t]=-r[t];
    for(int t=0;t<n;t++) y[m+t]=0.0;
    for(int k=0;k<n;k++){
        double*a=Q+(size_t)k*R; double nrm2=0;
        for(int t=k;t<R;t++) nrm2+=a[t]*a[t];
        double al=sqrt(nrm2); if(!(al>0.0)) return 1;
        if(a[k]>0) al=-al;
        double v0=a[k]-al; a[k]=al;
        double vn=v0*v0; for(int t=k+1;t<R;t++) vn+=a[t]*a[t];
        if(vn==0.0) continue;
        for(int j=k+1;j<=n;j++){ double*b=Q+(size_t)j*R;
            double s=v0*b[k]; for(int t=k+1;t<R;t++) s+=a[t]*b[t];
            s*=2.0/vn; b[k]-=s*v0; for(int t=k+1;t<R;t++) b[t]-=s*a[t]; }
    }
    for(int k=n-1;k>=0;k--){ double s=y[k];
        for(int j=k+1;j<n;j++) s-=Q[(size_t)j*R+k]*d[j];
        double rkk=Q[(size_t)k*R+k]; if(!(fabs(rkk)>0.0)) return 1; d[k]=s/rkk; }
    for(int k=0;k<n;k++) if(!isfinite(d[k])) return 1;
    return 0;
}

/* forward-difference Jacobian, H = J^T J, g = J^T r; returns |r|^2 */
static double lm_jac(work_t*W,const double*x,int ma,int mb){
    const int mr=ma+mb;
    for(int k=0;k<NX;k++) W->delta[k]=g_fdh*fmax(1.0,fabs(x[k]));
    for(int base=0;base<NX+1;base+=VW){
        for(int l=0;l<VW;l++){
            int pt=base+l; if(pt>NX) pt=0;
            memcpy(W->Xbuf+(size_t)l*NX,x,sizeof(double)*NX);
            if(pt>0) W->Xbuf[(size_t)l*NX+pt-1]+=W->delta[pt-1];
        }
        resid4(W,W->Xbuf,ma,mb,W->Rbuf);
        for(int l=0;l<VW;l++){ int pt=base+l; if(pt>NX) continue;
            memcpy(W->Rall+(size_t)pt*MEXT,W->Rbuf+(size_t)l*MEXT,sizeof(double)*mr); }
    }
    memcpy(W->r0,W->Rall,sizeof(double)*mr);
    for(int t=0;t<mr;t++) for(int k=0;k<NX;k++)
        W->J[(size_t)t*NX+k]=(W->Rall[(size_t)(k+1)*MEXT+t]-W->r0[t])/W->delta[k];
    memset(W->H,0,sizeof(double)*NX*NX); memset(W->g,0,sizeof(double)*NX);
    for(int t=0;t<mr;t++){
        const double*Jt=W->J+(size_t)t*NX; double rt=W->r0[t];
        for(int i=0;i<NX;i++){
            double ji=Jt[i]; W->g[i]+=ji*rt;
            __m256d vi=_mm256_set1_pd(ji); double*Hi=W->H+(size_t)i*NX;
            int j=0; for(;j+3<NX;j+=4)
                _mm256_storeu_pd(Hi+j,_mm256_fmadd_pd(vi,_mm256_loadu_pd(Jt+j),_mm256_loadu_pd(Hi+j)));
            for(;j<NX;j++) Hi[j]+=ji*Jt[j];
        }
    }
    return fnorm(W->r0,mr);
}

/* session-6 LM of crksearch.c: lanes lam*{1/3,1,10,100}; on failure lam grows */
static double lm_old(work_t*W,double*x,int ma,int mb,int iters,double tol,double*Lw){
    const int mr=ma+mb;
    resid1(W,x,ma,mb,W->r0);
    double f=fnorm(W->r0,mr), lam=1e-3;
    const double cand[VW]={1.0/3.0,1.0,10.0,100.0};
    int okl[VW];
    for(int it=0;it<iters;it++){
        if(fmaxabs(W->r0,mr)<tol) break;
        if(lam>1e12) break;
        f=lm_jac(W,x,ma,mb);
        int accepted=0;
        for(int att=0;att<6 && !accepted;att++){
            int nv=0; double lamv[VW];
            for(int l=0;l<VW;l++){
                lamv[l]=lam*cand[l];
                okl[l]=chol_solve(W->H,W->g,lamv[l],NX,W->step+(size_t)l*NX,Lw)==0;
                if(okl[l]){ nv++;
                    for(int k=0;k<NX;k++) W->xtry[(size_t)l*NX+k]=x[k]+W->step[(size_t)l*NX+k];
                } else memcpy(W->xtry+(size_t)l*NX,x,sizeof(double)*NX);
            }
            if(!nv){ lam*=1e3; continue; }
            resid4(W,W->xtry,ma,mb,W->Rbuf);
            int bl=-1; double bf=f;
            for(int l=0;l<VW;l++){ if(!okl[l]) continue;
                double fn=fnorm(W->Rbuf+(size_t)l*MEXT,mr); if(fn<bf){ bf=fn; bl=l; } }
            if(bl>=0){
                memcpy(x,W->xtry+(size_t)bl*NX,sizeof(double)*NX);
                memcpy(W->r0,W->Rbuf+(size_t)bl*MEXT,sizeof(double)*mr);
                f=bf; lam=fmax(lamv[bl]*0.3,1e-12); accepted=1;
            } else lam*=1e3;
        }
        if(!accepted && lam>1e12) break;
    }
    resid1(W,x,ma,mb,W->r0);
    return fmaxabs(W->r0,mr);
}

/* session-7 endgame: GN with central differences, min-norm step, no descent test, best point */
static double lm_gn(work_t*W,double*x,int ma,int mb,int iters,double tol){
    const int mr=ma+mb;
    resid1(W,x,ma,mb,W->r0);
    double f=fnorm(W->r0,mr), fbest=f, xbest[NX]; memcpy(xbest,x,sizeof(double)*NX);
    for(int it=0;it<iters;it++){
        if(fmaxabs(W->r0,mr)<tol) break;
        for(int k=0;k<NX;k++) W->delta[k]=g_fdc*fmax(1.0,fabs(x[k]));
        for(int base=0;base<2*NX;base+=VW){
            for(int l=0;l<VW;l++){ int id=base+l; memcpy(W->Xbuf+(size_t)l*NX,x,sizeof(double)*NX);
                if(id<2*NX) W->Xbuf[(size_t)l*NX+id/2]+=(id&1)?-W->delta[id/2]:W->delta[id/2]; }
            resid4(W,W->Xbuf,ma,mb,W->Rbuf);
            for(int l=0;l<VW;l+=2){ int id=base+l; if(id>=2*NX) continue; int k=id/2;
                const double*rp=W->Rbuf+(size_t)l*MEXT,*rm=W->Rbuf+(size_t)(l+1)*MEXT;
                for(int t=0;t<mr;t++) W->J[(size_t)t*NX+k]=(rp[t]-rm[t])/(2.0*W->delta[k]); }
        }
        double jf=0; for(size_t q=0;q<(size_t)mr*NX;q++) jf+=W->J[q]*W->J[q];
        double*d=W->step;
        if(qr_solve(W->J,W->r0,NULL,g_rc*g_rc*jf,mr,NX,d,W->Q)) break;
        for(int k=0;k<NX;k++) x[k]+=d[k];
        resid1(W,x,ma,mb,W->r0); f=fnorm(W->r0,mr);
        if(!isfinite(f)) break;
        if(f<fbest){ fbest=f; memcpy(xbest,x,sizeof(double)*NX); }
    }
    memcpy(x,xbest,sizeof(double)*NX);
    resid1(W,x,ma,mb,W->r0);
    return fmaxabs(W->r0,mr);
}

static double lm_one(work_t*W,double*x,int ma,int mb,int iters,double tol,double*Lw){
    double e=lm_old(W,x,ma,mb,iters,tol,Lw);
    if(e>=tol && e<g_polish){
        double xs[NX]; memcpy(xs,x,sizeof(double)*NX);
        double e2=lm_gn(W,x,ma,mb,30,tol);
        if(e2<e) e=e2; else { memcpy(x,xs,sizeof(double)*NX); resid1(W,x,ma,mb,W->r0); }
    }
    return e;
}

static inline uint64_t sm64(uint64_t*s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static inline double u01(uint64_t*s){ return (double)(sm64(s)>>11)*(1.0/9007199254740992.0); }
static inline double nrm(uint64_t*s){ double u=u01(s),v=u01(s); if(u<1e-300)u=1e-300;
    return sqrt(-2.0*log(u))*cos(6.283185307179586*v); }

static void write_status(const char*outf,long done,int nstart,long pass1,long nhit,double best,double t0){
    char path[1024]; snprintf(path,sizeof(path),"%s.status",outf);
    FILE*f=fopen(path,"w"); if(!f) return;
    double el=difftime(time(NULL),(time_t)t0), frac=nstart>0?(double)done/nstart:0.0;
    fprintf(f,"starts=%ld\ntotal=%d\npercent=%.1f\npass1=%ld\nhits=%ld\nbest=%.6e\nelapsed_s=%.0f\neta_s=%.0f\n",
            done,nstart,100.0*frac,pass1,nhit,best,el,(frac>0.001)?el*(1.0/frac-1.0):-1.0);
    fclose(f);
}

int main(int argc,char**argv){
    if(argc<6){ fprintf(stderr,
        "usage: %s nstart seed ma:mb,ma:mb,... it,it,... out.bin [seeds.bin nseed spread]\n",argv[0]); return 1; }
    int nstart=atoi(argv[1]); uint64_t seed0=strtoull(argv[2],0,10);
    int nst=0, sa[8], sb[8], sit[8];
    { const char*p=argv[3]; while(*p && nst<8){ sa[nst]=atoi(p); p=strchr(p,':'); if(!p) break; sb[nst]=atoi(++p); nst++;
        p=strchr(p,','); if(!p) break; p++; } }
    { const char*p=argv[4]; for(int k=0;k<nst;k++){ sit[k]=atoi(p); const char*q=strchr(p,','); if(q) p=q+1; } }
    const char*outf=argv[5];
    if(nst<1){ fprintf(stderr,"no stages\n"); return 1; }
    for(int k=0;k<nst;k++) if(sa[k]>M||sb[k]>M||(MH==0&&sb[k])){ fprintf(stderr,"bad tree counts\n"); return 1; }
    double*seeds=0; long nseed=0; double spread=0.3;
    if(argc>7){ nseed=atol(argv[7]); if(argc>8) spread=atof(argv[8]);
        seeds=(double*)malloc(sizeof(double)*(size_t)nseed*NX);
        FILE*fp=fopen(argv[6],"rb"); if(!fp){ fprintf(stderr,"no seedfile\n"); return 1; }
        if(fread(seeds,sizeof(double),(size_t)nseed*NX,fp)!=(size_t)nseed*NX){ fprintf(stderr,"short seedfile\n"); return 1; }
        fclose(fp); }
    { const char*e=getenv("CRK_POLISH"); if(e) g_polish=atof(e); }
    fprintf(stderr,"S=%d MH=%d NX=%d M=%d seeds %ld spread %.3f stages:",S,MH,NX,M,nseed,spread);
    for(int k=0;k<nst;k++) fprintf(stderr," (%d,%d)x%d",sa[k],sb[k],sit[k]);
    fprintf(stderr,"\n");
    const int ma2=sa[nst-1], mb2=sb[nst-1];
    long cap=20000;
    double*hits=(double*)malloc(sizeof(double)*NX*cap), *nears=(double*)malloc(sizeof(double)*NX*cap);
    long nhit=0,nnear=0,done=0,pass1=0; double best=1e9; long hist[20]={0};
    double t0=(double)time(NULL), t_last=t0;
#pragma omp parallel
    {
        work_t*W=work_new(); double*Lw=(double*)_aligned_malloc(sizeof(double)*NX*NX,64);
        double x[NX]; int r;
#pragma omp for schedule(dynamic,2)
        for(r=0;r<nstart;r++){
            uint64_t st=seed0*1000003ULL+0x9E37ULL*(uint64_t)r+1;
            for(int w=0;w<4;w++) sm64(&st);
            if(seeds){
                long si=(long)(u01(&st)*(double)nseed); if(si>=nseed) si=nseed-1;
                double sc=spread*(0.15+0.85*u01(&st));
                for(int k=0;k<NX;k++) x[k]=seeds[(size_t)si*NX+k]+sc*nrm(&st);
            } else {
                double sc=0.2+0.8*u01(&st);
                for(int k=0;k<NX;k++) x[k]=sc*nrm(&st);
            }
            double e=1e9; int got1=1;
            for(int k=0;k<nst-1 && got1;k++){ e=lm_one(W,x,sa[k],sb[k],sit[k],1e-12,Lw); got1=(e<1e-10); }
            if(got1) e=lm_one(W,x,ma2,mb2,sit[nst-1],1e-12,Lw);
#pragma omp critical
            {
                done++; pass1+=got1;
                if(got1){ if(e<best) best=e;
                    int hb=(e>0)?(int)floor(log10(e))+16:0; if(hb<0)hb=0; if(hb>19)hb=19; hist[hb]++;
                    if(e<1e-12 && nhit<cap){ memcpy(hits+(size_t)nhit*NX,x,sizeof(double)*NX); nhit++;
                        FILE*hf=fopen(outf,"wb"); if(hf){ fwrite(hits,sizeof(double),(size_t)nhit*NX,hf); fclose(hf); } }
                    else if(e<1e-5 && nnear<cap){ memcpy(nears+(size_t)nnear*NX,x,sizeof(double)*NX); nnear++; } }
                if(done%200==0 || difftime(time(NULL),(time_t)t_last)>=30.0){
                    write_status(outf,done,nstart,pass1,nhit,best,t0); t_last=(double)time(NULL); }
            }
        }
        _aligned_free(Lw);
    }
    write_status(outf,done,nstart,pass1,nhit,best,t0);
    FILE*fp=fopen(outf,"wb"); fwrite(hits,sizeof(double),(size_t)nhit*NX,fp); fclose(fp);
    { char np_[1024]; snprintf(np_,sizeof(np_),"%s.near",outf);
      FILE*fn=fopen(np_,"wb"); if(fn){ fwrite(nears,sizeof(double),(size_t)nnear*NX,fn); fclose(fn); } }
    printf("starts=%d pass1=%ld hits=%ld best=%.3e  S=%d MH=%d NX=%d stages=%s iters=%s seeds=%ld spread=%.3f time=%.0fs\n",
           nstart,pass1,nhit,best,S,MH,NX,argv[3],argv[4],nseed,spread,difftime(time(NULL),(time_t)t0));
    printf("  final max|r| by decade:");
    for(int k=0;k<20;k++) if(hist[k]) printf(" 1e%+d:%ld",k-16,hist[k]);
    printf("\n");
    return 0;
}
