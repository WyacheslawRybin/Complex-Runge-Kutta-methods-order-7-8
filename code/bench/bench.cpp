// CPU-time benchmark: cost of one fixed step of every method on every test
// problem, real arithmetic for the real tableaux and std::complex<double> for
// the projected complex tableaux (the state stays real, stages are complex).
// Output: CSV  problem,method,stages,evals_per_step,ns_per_step,ns_per_f_real,ns_per_f_complex
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include "coeffs.h"


using cd = std::complex<double>;

// ---------------------------------------------------------------- problems
template <class T> struct Exponential { static const int D = 1; static constexpr double TEND = 5;
  static void f(T t, const T* y, T* d) { (void)t; d[0] = -std::exp(y[0]); }
  static void y0(double* y) { y[0] = 1; } };
template <class T> struct NonAut { static const int D = 1; static constexpr double TEND = 5;
  static void f(T t, const T* y, T* d) { T s = std::sin(t); d[0] = 4.0 * y[0] * s * s * s * std::cos(t); }
  static void y0(double* y) { y[0] = 1; } };
template <class T> struct VdP { static const int D = 2; static constexpr double TEND = 5;
  static void f(T, const T* y, T* d) { d[0] = y[1]; d[1] = (1.0 - y[0] * y[0]) * y[1] - y[0]; }
  static void y0(double* y) { y[0] = 2; y[1] = 0; } };
template <class T> struct Rossler { static const int D = 3; static constexpr double TEND = 5;
  static void f(T, const T* y, T* d) { d[0] = -y[1] - y[2]; d[1] = y[0] + 0.2 * y[1]; d[2] = 0.2 + y[2] * (y[0] - 5.7); }
  static void y0(double* y) { y[0] = 1; y[1] = 1; y[2] = 1; } };
template <class T> struct Lorenz { static const int D = 3; static constexpr double TEND = 5;
  static void f(T, const T* y, T* d) { d[0] = 10.0 * (y[1] - y[0]); d[1] = y[0] * (28.0 - y[2]) - y[1]; d[2] = y[0] * y[1] - (8.0 / 3.0) * y[2]; }
  static void y0(double* y) { y[0] = 1; y[1] = 1; y[2] = 20; } };
template <class T> struct Bouali { static const int D = 3; static constexpr double TEND = 5;
  static void f(T, const T* y, T* d) { d[0] = 3.0 * y[0] * (1.0 - y[1]) - 2.2 * y[2]; d[1] = -y[1] * (1.0 - y[0] * y[0]); d[2] = 0.001 * y[0]; }
  static void y0(double* y) { y[0] = 1; y[1] = 0.1; y[2] = 0; } };
template <class T> struct Kepler { static const int D = 4; static constexpr double TEND = 20;
  static void f(T, const T* y, T* d) { T r2 = y[0] * y[0] + y[1] * y[1]; T r3 = r2 * std::sqrt(r2);
    d[0] = y[2]; d[1] = y[3]; d[2] = -y[0] / r3; d[3] = -y[1] / r3; }
  static void y0(double* y) { y[0] = 0.5; y[1] = 0; y[2] = 0; y[3] = std::sqrt(3.0); } };
template <class T> struct Rigid { static const int D = 3; static constexpr double TEND = 12;
  static void f(T, const T* y, T* d) { d[0] = y[1] * y[2]; d[1] = -y[0] * y[2]; d[2] = -0.51 * y[0] * y[1]; }
  static void y0(double* y) { y[0] = 0; y[1] = 1; y[2] = 1; } };
template <class T> struct Brusselator { static const int D = 2; static constexpr double TEND = 20;
  static void f(T, const T* y, T* d) { T q = y[0] * y[0] * y[1]; d[0] = 1.0 + q - 4.0 * y[0]; d[1] = 3.0 * y[0] - q; }
  static void y0(double* y) { y[0] = 1.5; y[1] = 3; } };
template <class T> struct Arenstorf { static const int D = 4; static constexpr double TEND = 17.0652165601579625588917206249;
  static void f(T, const T* y, T* d) { const double mu = 0.012277471, mp = 1 - mu;
    T a = y[0] + mu, b = y[0] - mp; T r1 = a * a + y[1] * y[1], r2 = b * b + y[1] * y[1];
    T D1 = r1 * std::sqrt(r1), D2 = r2 * std::sqrt(r2);
    d[0] = y[2]; d[1] = y[3]; d[2] = y[0] + 2.0 * y[3] - mp * a / D1 - mu * b / D2;
    d[3] = y[1] - 2.0 * y[2] - mp * y[1] / D1 - mu * y[1] / D2; }
  static void y0(double* y) { y[0] = 0.994; y[1] = 0; y[2] = 0; y[3] = -2.00158510637908252240537862224; } };

// ---------------------------------------------------------------- integrators
volatile double sink;

template <template <class> class P>
double run_real(const Tab& tb, long N) {
  const int D = P<double>::D; const int s = tb.s;
  double y[D], K[NS][D], Y[D]; P<double>::y0(y);
  const double h = P<double>::TEND / N; double t = 0;
  bool have_k1 = false;
  for (long n = 0; n < N; ++n) {
    if (!(tb.fsal && have_k1)) P<double>::f(t, y, K[0]);
    for (int i = 1; i < s; ++i) {
      for (int q = 0; q < D; ++q) { double acc = 0; for (int j = 0; j < i; ++j) if (tb.are[i][j] != 0) acc += tb.are[i][j] * K[j][q]; Y[q] = y[q] + h * acc; }
      P<double>::f(t + tb.cre[i] * h, Y, K[i]);
    }
    if (tb.fsal) { for (int q = 0; q < D; ++q) y[q] = Y[q]; for (int q = 0; q < D; ++q) K[0][q] = K[s - 1][q]; have_k1 = true; }
    else for (int q = 0; q < D; ++q) { double acc = 0; for (int i = 0; i < s; ++i) if (tb.bre[i] != 0) acc += tb.bre[i] * K[i][q]; y[q] += h * acc; }
    t += h;
  }
  double r = 0; for (int q = 0; q < D; ++q) r += y[q]; return r;
}

template <template <class> class P>
double run_cplx(const Tab& tb, long N) {
  const int D = P<cd>::D; const int s = tb.s;
  double y[D]; cd K[NS][D], Y[D]; P<cd>::y0(y);
  cd a[NS][NS], b[NS], c[NS];
  for (int i = 0; i < NS; ++i) { b[i] = cd(tb.bre[i], tb.bim[i]); c[i] = cd(tb.cre[i], tb.cim[i]);
    for (int j = 0; j < NS; ++j) a[i][j] = cd(tb.are[i][j], tb.aim[i][j]); }
  const double h = P<cd>::TEND / N; double t = 0;
  const int nr = tb.nreal;                 // leading real stages: real rows, real arithmetic
  double KR[NS][D], YR[D];
  for (long n = 0; n < N; ++n) {
    P<double>::f(t, y, KR[0]);
    for (int i = 1; i < nr; ++i) {
      for (int q = 0; q < D; ++q) { double acc = 0; for (int j = 0; j < i; ++j) acc += tb.are[i][j] * KR[j][q]; YR[q] = y[q] + h * acc; }
      P<double>::f(t + tb.cre[i] * h, YR, KR[i]);
    }
    for (int i = nr; i < s; ++i) {
      for (int q = 0; q < D; ++q) { cd acc = 0; for (int j = 0; j < nr; ++j) acc += a[i][j] * KR[j][q];
        for (int j = nr; j < i; ++j) acc += a[i][j] * K[j][q]; Y[q] = y[q] + h * acc; }
      P<cd>::f(cd(t, 0) + c[i] * h, Y, K[i]);
    }
    for (int q = 0; q < D; ++q) { double acc = 0; for (int i = 0; i < nr; ++i) acc += b[i].real() * KR[i][q];
      for (int i = nr; i < s; ++i) acc += (b[i] * K[i][q]).real(); y[q] += h * acc; }
    t += h;
  }
  double r = 0; for (int q = 0; q < D; ++q) r += y[q]; return r;
}

template <template <class> class P>
double time_f(bool cplx, long M) {
  const int D = P<double>::D;
  auto t0 = std::chrono::high_resolution_clock::now();
  if (cplx) { cd y[D], d[D]; double y0[D]; P<double>::y0(y0);
    for (int q = 0; q < D; ++q) y[q] = cd(y0[q], 1e-3 * (q + 1));
    for (long m = 0; m < M; ++m) { P<cd>::f(cd(0.1, 0.01), y, d); for (int q = 0; q < D; ++q) y[q] += 1e-12 * d[q]; }
    sink = y[0].real();
  } else { double y[D], d[D]; P<double>::y0(y);
    for (long m = 0; m < M; ++m) { P<double>::f(0.1, y, d); for (int q = 0; q < D; ++q) y[q] += 1e-12 * d[q]; }
    sink = y[0]; }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / M;
}

template <template <class> class P>
void bench(const char* pname) {
  const long M = 4000000;
  double fr = 1e30, fc = 1e30;
  for (int rep = 0; rep < 5; ++rep) { fr = std::fmin(fr, time_f<P>(false, M)); fc = std::fmin(fc, time_f<P>(true, M)); }
  for (int k = 0; k < NMETH; ++k) {
    const Tab& tb = TABS[k];
    const long N = 400000;
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
      auto t0 = std::chrono::high_resolution_clock::now();
      sink = tb.cplx ? run_cplx<P>(tb, N) : run_real<P>(tb, N);
      auto t1 = std::chrono::high_resolution_clock::now();
      best = std::fmin(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / N);
    }
    int ev = tb.fsal ? tb.s - 1 : tb.s;
    std::printf("%s,%s,%d,%d,%.3f,%.3f,%.3f\n", pname, tb.name, tb.s, ev, best, fr, fc);
  }
}

template <template <class> class P>
void check(const char* pname) {        // sum of the components at TEND: all methods must agree
  for (int k = 0; k < NMETH; ++k) {
    const Tab& tb = TABS[k];
    double r1 = tb.cplx ? run_cplx<P>(tb, 4000) : run_real<P>(tb, 4000);
    double r2 = tb.cplx ? run_cplx<P>(tb, 8000) : run_real<P>(tb, 8000);
    std::printf("%-14s %-14s %.15e  %.1e\n", pname, tb.name, r2, std::fabs(r2 - r1));
  }
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "check") {
    check<Exponential>("exponential"); check<NonAut>("nonautonomous"); check<VdP>("VanderPol");
    check<Rossler>("Rossler"); check<Lorenz>("Lorenz"); check<Bouali>("Bouali"); check<Kepler>("Kepler");
    check<Rigid>("RigidBody"); check<Brusselator>("Brusselator"); check<Arenstorf>("Arenstorf");
    return 0;
  }
  std::printf("problem,method,stages,evals_per_step,ns_per_step,ns_per_f_real,ns_per_f_complex\n");
  bench<Exponential>("exponential"); bench<NonAut>("nonautonomous"); bench<VdP>("VanderPol");
  bench<Rossler>("Rossler"); bench<Lorenz>("Lorenz"); bench<Bouali>("Bouali"); bench<Kepler>("Kepler");
  bench<Rigid>("RigidBody"); bench<Brusselator>("Brusselator"); bench<Arenstorf>("Arenstorf");
  return 0;
}
