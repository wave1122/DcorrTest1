#ifndef DATA_H
#define DATA_H

#include <vector>
#include <gsl/gsl_randist.h>
#include <matrix_ops2.h>

using DataType = double;
using Values = std::vector<DataType>;

using namespace std;

/*
Values LinSpace(double s, double e, size_t n);

std::pair<Values, Values> GenerateData(double s,
                                       double e,
                                       size_t n,
                                       size_t seed,
                                       bool noise);
*/

#include <random>

Values LinSpace(double s, double e, size_t n) {
  Values x_values(n);
  double step = (e - s) / n;

  double v = s;
  for (size_t i = 0; i < n; ++i) {
    x_values[i] = v;
    v += step;
  }
  x_values[n - 1] = e;

  return x_values;
}

std::pair<Values, Values> GenerateData(double s,
                                       double e,
                                       size_t n,
                                       size_t seed,
                                       bool noise) {
  std::vector<DataType> x;
  std::vector<DataType> y;
  if (noise) {
	std::mt19937 re(seed);
    std::uniform_real_distribution<DataType> dist(s, e);
    std::normal_distribution<DataType> noise_dist;

    x.resize(n);
    y.resize(n);
    for (size_t i = 0; i < n; ++i) {
      auto x_val = dist(re);
      auto y_val = std::cos(M_PI * x_val) + (noise_dist(re) * 0.3);
      x[i] = x_val;
      y[i] = y_val;
    }
  } else {
    x = LinSpace(s, e, n);
    y.reserve(x.size());
    for (auto x_val : x) {
      auto y_val = std::cos(M_PI * x_val);
      y.push_back(y_val);
    }
  }

  return {x, y};
}


std::pair<Matrix, Matrix> GenerateData(double s,
                                        double e,
                                        int n,
                                        int dim,
										int seed) {

    std::mt19937 re(seed);
    std::uniform_real_distribution<DataType> unif(s, e);
    std::normal_distribution<DataType> normal_dist;

	Matrix x(dim, n), y(1, n);
    x.set(0.);
    y.set(0.);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) {
			x(j+1, i+1) = unif(re);
			y(1, i+1) += -0.5 * x(j+1, i+1);
        }
        y(1, i+1) += 0.3 * normal_dist(re);
    }
    return {x, y};
}








#endif  // DATA_H
