#ifndef ML_REG_DCORR_H_
#define ML_REG_DCORR_H_

#include <unistd.h>
#include <omp.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/Statistics.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <nl_dgp.h>

#include<ShogunML/ml_reg_6_1_4.h>

#define CHUNK 1

using namespace std;
using namespace shogun;
using namespace shogun::linalg;

class ML_REG_DCORR {
	public:
		ML_REG_DCORR () {   };//default constructor
		~ML_REG_DCORR () {   };//default destructor


		/* Calculate the conditional expectations of  ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
		a ML regression algorithm.
		INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
		OUTPUT: a T0 = T-L by T-L matrix (mat_reg_first, mat_reg_second, mat_breg). */
		template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																			const SGVector<double> &, /*train labels*/
																			int, /*number of subsets for TSCV*/
																			int, /*minimum subset size for TSCV*/
																			SGVector<int>, /*list of tree max depths (for GBM)*/
																			SGVector<int>, /*list of numbers of iterations (for GBM)*/
																			SGVector<double>, /*list of learning rates (for GBM)*/
																			SGVector<double>, /*list of subset fractions (for GBM)*/
																			SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int>, /*list of number of bags (for RF)*/
																			int seed /*seed for random number generator*/),
				SGVector<double> ML_reg(const SGMatrix<double> &,
									  const SGVector<double> &,
									  int, /*tree max depth*/
									  int, /*number of iterations*/
									  double, /*learning rate*/
									  double, /*subset fraction*/
									  int, /*number of random features used for bagging (for RF)*/
									  int, /*number of bags (for RF)*/
									  int /*seed for the random number generator*/)>
		static void reg (	SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
							SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
							SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
							const SGMatrix<double> &X, const int L, const double expn,
							int num_subsets, /*number of subsets for TSCV*/
							int min_subset_size, /*minimum subset size for TSCV*/
							SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
							SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
							SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
							SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
							SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
							SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
							int seed /*seed for random number generator*/);


		/* Calculate the conditional expectations of  ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
		a ML regression algorithm applied to model (2.15).
		INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
		OUTPUT: a T0 = T-L by T-L matrix (mat_reg_first, mat_reg_second, mat_breg). */
		template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																					const SGVector<double> &, /*train labels*/
																					int, /*number of subsets for TSCV*/
																					int, /*minimum subset size for TSCV*/
																					SGVector<int>, /*list of tree max depths (for GBM)*/
																					SGVector<int>, /*list of numbers of iterations (for GBM)*/
																					SGVector<double>, /*list of learning rates (for GBM)*/
																					SGVector<double>, /*list of subset fractions (for GBM)*/
																					SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																					SGVector<int>, /*list of number of bags (for RF)*/
																					int seed /*seed for random number generator*/),
				SGVector<double> ML_reg(const SGMatrix<double> &,
										const SGVector<double> &,
										int, /*tree max depth*/
										int, /*number of iterations*/
										double, /*learning rate*/
										double, /*subset fraction*/
										int, /*number of random features used for bagging (for RF)*/
										int, /*number of bags (for RF)*/
										int /*seed for the random number generator*/)>
		static void reg1(	SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
							SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
							SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
							const SGMatrix<double> &X,
							const int L, /*maximum truncation lag*/
							const double expn, /*exponent of distances*/
							int num_subsets, /*number of subsets for TSCV*/
							int min_subset_size, /*minimum subset size for TSCV*/
							SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
							SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
							SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
							SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
							SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
							SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
							int seed /*seed for random number generator*/);


		/* Calculate the conditional expectations of  ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
		a ML regression algorithm with model (2.15).
		INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
		OUTPUT: a T0 = T-L by T-L matrix (mat_reg_first, mat_reg_second, mat_breg). */
		template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																			const SGVector<double> &, /*train labels*/
																			int, /*number of subsets for TSCV*/
																			int, /*minimum subset size for TSCV*/
																			SGVector<int>, /*list of tree max depths (for GBM)*/
																			SGVector<int>, /*list of numbers of iterations (for GBM)*/
																			SGVector<double>, /*list of learning rates (for GBM)*/
																			SGVector<double>, /*list of subset fractions (for GBM)*/
																			SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int>, /*list of number of bags (for RF)*/
																			int seed /*seed for random number generator*/),
				SGVector<double> ML_reg(const SGMatrix<double> &,
									  const SGVector<double> &,
									  int, /*tree max depth*/
									  int, /*number of iterations*/
									  double, /*learning rate*/
									  double, /*subset fraction*/
									  int, /*number of random features used for bagging (for RF)*/
									  int, /*number of bags (for RF)*/
									  int /*seed for the random number generator*/)>
		static void reg1(	SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
							SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
							SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
							const SGMatrix<double> &X,
							const int L, /*maximum truncation lag*/
							const double expn, /*exponent of distances*/
							const double expn_x, /*exponent of data*/
							int num_subsets, /*number of subsets for TSCV*/
							int min_subset_size, /*minimum subset size for TSCV*/
							SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
							SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
							SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
							SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
							SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
							SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
							int seed /*seed for random number generator*/);


		/* Calculate the conditional expectations of ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
		moving averages.
		INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
		OUTPUT: a T0 = T-L by T-L matrix (mat_reg_first, mat_reg_second, mat_breg). */
		static void reg(SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
						SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
						SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
						const SGMatrix<double> &X,
						const int L,
						const double expn);


		/* Calculate centered Euclidean distances by ML methods.
		INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
		OUTPUT: a T0 = T-L by T-L matrix (mat_U). */
		template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																					const SGVector<double> &, /*train labels*/
																					int, /*number of subsets for TSCV*/
																					int, /*minimum subset size for TSCV*/
																					SGVector<int>, /*list of tree max depths (for GBM)*/
																					SGVector<int>, /*list of numbers of iterations (for GBM)*/
																					SGVector<double>, /*list of learning rates (for GBM)*/
																					SGVector<double>, /*list of subset fractions (for GBM)*/
																					SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																					SGVector<int>, /*list of number of bags (for RF)*/
																					int seed /*seed for random number generator*/),
				SGVector<double> ML_reg(const SGMatrix<double> &,
										const SGVector<double> &,
										int, /*tree max depth*/
										int, /*number of iterations*/
										double, /*learning rate*/
										double, /*subset fraction*/
										int, /*number of random features used for bagging (for RF)*/
										int, /*number of bags (for RF)*/
										int /*seed for the random number generator*/)>
		static SGMatrix<double> var_U(	const SGMatrix<double> &X, const int L, const double expn,
										int num_subsets, /*number of subsets for TSCV*/
										int min_subset_size, /*minimum subset size for TSCV*/
										SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
										SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
										SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
										SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
										SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
										SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
										int seed /*seed for random number generator*/);

		/* Calculate centered Euclidean distances by ML methods.
		INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
		OUTPUT: a T0 = T-L by T-L matrix (mat_U). */
		template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																					const SGVector<double> &, /*train labels*/
																					int, /*number of subsets for TSCV*/
																					int, /*minimum subset size for TSCV*/
																					SGVector<int>, /*list of tree max depths (for GBM)*/
																					SGVector<int>, /*list of numbers of iterations (for GBM)*/
																					SGVector<double>, /*list of learning rates (for GBM)*/
																					SGVector<double>, /*list of subset fractions (for GBM)*/
																					SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																					SGVector<int>, /*list of number of bags (for RF)*/
																					int seed /*seed for random number generator*/),
				SGVector<double> ML_reg(const SGMatrix<double> &,
										const SGVector<double> &,
										int, /*tree max depth*/
										int, /*number of iterations*/
										double, /*learning rate*/
										double, /*subset fraction*/
										int, /*number of random features used for bagging (for RF)*/
										int, /*number of bags (for RF)*/
										int /*seed for the random number generator*/)>
		static SGMatrix<double> var_U(	const SGMatrix<double> &X,
										const int L, /*maximum truncation lag*/
										const double expn, /*exponent of distances*/
										const double expn_x, /*exponent of data*/
										int num_subsets, /*number of subsets for TSCV*/
										int min_subset_size, /*minimum subset size for TSCV*/
										SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
										SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
										SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
										SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
										SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
										SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
										int seed /*seed for random number generator*/);

		/* Calculate centered Euclidean distances by moving averages.
		INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
		OUTPUT: a T0 = T-L by T-L matrix (mat_U). */
		static SGMatrix<double> var_U(	const SGMatrix<double> &X,
										const int L,
										const double expn);
};


/* Calculate the conditional expectations of  ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
a ML regression algorithm.
INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
OUTPUT: a T0 = T-L by T-L matrix (mat_reg_first, mat_reg_second, mat_breg). */
template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																			const SGVector<double> &, /*train labels*/
																			int, /*number of subsets for TSCV*/
																			int, /*minimum subset size for TSCV*/
																			SGVector<int>, /*list of tree max depths (for GBM)*/
																			SGVector<int>, /*list of numbers of iterations (for GBM)*/
																			SGVector<double>, /*list of learning rates (for GBM)*/
																			SGVector<double>, /*list of subset fractions (for GBM)*/
																			SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int>, /*list of number of bags (for RF)*/
																			int seed /*seed for random number generator*/),
		SGVector<double> ML_reg(const SGMatrix<double> &,
								const SGVector<double> &,
								int, /*tree max depth*/
								int, /*number of iterations*/
								double, /*learning rate*/
								double, /*subset fraction*/
								int, /*number of random features used for bagging (for RF)*/
								int, /*number of bags (for RF)*/
								int /*seed for the random number generator*/)>
void ML_REG_DCORR::reg (SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
						SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
						SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
						const SGMatrix<double> &X, const int L, const double expn,
						int num_subsets, /*number of subsets for TSCV*/
						int min_subset_size, /*minimum subset size for TSCV*/
						SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
						SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
						SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
						SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
						SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
						SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
						int seed /*seed for random number generator*/) {
	auto T = X.num_rows, N = X.num_cols;
	auto T0 = T - L;
	ASSERT_(T0 == mat_reg_first.num_rows && T0 == mat_reg_first.num_cols && T0 == mat_breg.num_rows && T0 == mat_breg.num_cols);

	mat_reg_first.zero(); // reset the output matrix
	mat_reg_second.zero();
	mat_breg.zero();
	auto t = 0, s = 0, ell = 0, i = 0;
	SGVector<double> labels(T0), new_labels(T0), row_vec(N);
	SGMatrix<double> features(T0, N*L);

	int opt_tree_max_depth = 0, opt_num_iters = 0, opt_num_rand_feats = 0., opt_num_bags = 0.;
	double opt_learning_rate = 0, opt_subset_fraction = 0, label_mean = 0.;

	/* 1. Estimate E[||x_t - x_s||^\kappa| lags of x_t] */

	// do hyperparameter tuning
	for (t = L; t < T; ++t) {
		// construct labels
		row_vec = add(X.get_row_vector(t), X.get_row_vector(L), 1., -1.);
		labels[t-L] = std::pow(norm(row_vec), expn);
		//row_vec.display_vector("row_vec");
		//cout << "norm of row_vec = " << norm(row_vec) << endl;

		// construct features
		for (i = 0; i < N; ++i) {
			for (ell = 0; ell < L; ++ell) {
				features(t-L, i*L+ell) = X(t-ell-1, i);
			}
		}
	}

	labels.add( -Statistics::mean(labels) ); // de-mean labels
	std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
												ML_cv(	features, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
														learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);


	// estimate conditional expectations
	//#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(s,t,i,ell) firstprivate(labels,row_vec,features,new_labels)
	for (s = L; s < T; ++s) {
		for (t = L; t < T; ++t) {
			// construct labels
			row_vec = add(X.get_row_vector(t), X.get_row_vector(s), 1., -1.);
			labels[t-L] = std::pow(norm(row_vec), expn);
			//row_vec.display_vector("row_vec");
			//cout << "norm of row_vec = " << norm(row_vec) << endl;

			// construct features
			for (i = 0; i < N; ++i){
				for (ell = 0; ell < L; ++ell){
					features(t-L, i*L+ell) = X(t-ell-1, i);
				}
			}
		}

		label_mean = Statistics::mean(labels);
		labels.add(-label_mean); // de-mean labels
		new_labels = ML_reg(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																						opt_num_rand_feats, opt_num_bags, seed);
		//new_labels.add(label_mean); // add back the label mean
		mat_reg_first.set_column(s-L, new_labels);
	}

	/* 2. Estimate E[||x_t - x_s||^\kappa| lags of x_s] */
	mat_reg_second = transpose_matrix(mat_reg_first);

	/*//#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,i,ell) firstprivate(labels,row_vec,features,new_labels)
	for (t = L; t < T; ++t) {
		for (s = L; s < T; ++s) {
			// construct labels
			row_vec = add(X.get_row_vector(s), X.get_row_vector(t), 1., -1.);
			labels[s-L] = std::pow(norm(row_vec), expn);
			//row_vec.display_vector("row_vec");
			//cout << "norm of row_vec = " << norm(row_vec) << endl;

			// construct features
			for (i = 0; i < N; ++i){
				for (ell = 0; ell < L; ++ell){
					features(s-L, i*L+ell) = X(s-ell-1, i);
				}
			}
		}

		label_mean = Statistics::mean(labels);
		labels.add(-label_mean); // de-mean labels
		new_labels = ML_reg(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																						opt_num_rand_feats, opt_num_bags, seed);
		new_labels.add(label_mean); // add back the label mean
		mat_reg_second.set_column(t-L, new_labels);
	}*/


	/* 3. Estimate E[||x_t - x_s||^\kappa| lags of x_t and x_s] */

	// do hyperparameter tuning
	for (s = 0; s < T0; ++s) {
		// construct labels
		labels[s] = mat_reg_first(0, s);

		// construct features
		for (i = 0; i < N; ++i){
			for (ell = 0; ell < L; ++ell){
				features(s, i*L+ell) = X(s+L-ell-1, i);
			}
		}
	}

	// call a ML hyperparamter tuner
	labels.add( -Statistics::mean(labels) ); // de-mean labels
	std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
									ML_cv(	features, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
											learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);


	//#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,i,ell) firstprivate(labels,features,new_labels)
	for(t = 0; t < T0; ++t) {
		for (s = 0; s < T0; ++s) {
			// construct labels
			labels[s] = mat_reg_first(t, s);

			// construct features
			for (i = 0; i < N; ++i){
				for (ell = 0; ell < L; ++ell){
					features(s, i*L+ell) = X(s+L-ell-1, i);
				}
			}
		}

		label_mean = Statistics::mean(labels);
		labels.add(-label_mean); // de-mean labels
		new_labels = ML_reg(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																					opt_num_rand_feats, opt_num_bags, seed);
		new_labels.add(label_mean); // add back the label mean
		mat_breg.set_column(t, new_labels);
	}
	mat_breg = transpose_matrix(mat_breg);
}

/* Calculate the conditional expectations of ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
moving averages.
INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
OUTPUT: a T0 = T-L by T-L matrix (mat_reg_first, mat_reg_second, mat_breg). */
void ML_REG_DCORR::reg(	SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
						SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
						SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
						const SGMatrix<double> &X,
						const int L,
						const double expn) {
	auto T = X.num_rows, N = X.num_cols;
	auto T0 = T - L;
	ASSERT_(T0 == mat_reg_first.num_rows && T0 == mat_reg_first.num_cols && T0 == mat_breg.num_rows && T0 == mat_breg.num_cols);

	mat_reg_first.zero(); // reset the output matrix
	mat_reg_second.zero();
	mat_breg.zero();

	SGVector<double> row_vec(N);

	/* 1. Estimate E[||x_t - x_s||^\kappa| lags of x_t] */

	for (int s = L; s < T; ++s) {
		for (int t = L; t < T; ++t) {
			for (int ell = 0; ell < t; ++ell) {
				row_vec = add(X.get_row_vector(ell), X.get_row_vector(s), 1., -1.);
				mat_reg_first(t-L, s-L) += std::pow(norm(row_vec), expn) / L;
			}
		}
	}

	/* 2. Estimate E[||x_t - x_s||^\kappa| lags of x_s] */

	for (int t = L; t < T; ++t) {
		for (int s = L; s < T; ++s) {
			for (int ell = 0; ell < s; ++ell) {
				row_vec = add(X.get_row_vector(t), X.get_row_vector(ell), 1., -1.);
				mat_reg_second(s-L, t-L) += std::pow(norm(row_vec), expn) / L;
			}
		}
	}

	/* 3. Estimate E[||x_t - x_s||^\kappa| lags of x_t and x_s] */

	for (int t = L; t < T; ++t) {
		for (int s = L; s < T; ++s) {
			for (int ell = 0; ell < t; ++ell) {
				for (int k = 0; k < s; ++k) {
					row_vec = add(X.get_row_vector(ell), X.get_row_vector(k), 1., -1.);
					mat_breg(t-L, s-L) += std::pow(norm(row_vec), expn) / pow(L, 2.);
				}
			}
		}
	}
}

/* Calculate the conditional expectations of  ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
a ML regression algorithm applied to model (2.15).
INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
OUTPUT: a T0 = T-L by T-L matrix (mat_reg_first, mat_reg_second, mat_breg). */
template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																			const SGVector<double> &, /*train labels*/
																			int, /*number of subsets for TSCV*/
																			int, /*minimum subset size for TSCV*/
																			SGVector<int>, /*list of tree max depths (for GBM)*/
																			SGVector<int>, /*list of numbers of iterations (for GBM)*/
																			SGVector<double>, /*list of learning rates (for GBM)*/
																			SGVector<double>, /*list of subset fractions (for GBM)*/
																			SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int>, /*list of number of bags (for RF)*/
																			int seed /*seed for random number generator*/),
		SGVector<double> ML_reg(const SGMatrix<double> &,
								const SGVector<double> &,
								int, /*tree max depth*/
								int, /*number of iterations*/
								double, /*learning rate*/
								double, /*subset fraction*/
								int, /*number of random features used for bagging (for RF)*/
								int, /*number of bags (for RF)*/
								int /*seed for the random number generator*/)>
void ML_REG_DCORR::reg1(SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
						SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
						SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
						const SGMatrix<double> &X,
						const int L, /*maximum truncation lag*/
						const double expn, /*exponent of distances*/
						int num_subsets, /*number of subsets for TSCV*/
						int min_subset_size, /*minimum subset size for TSCV*/
						SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
						SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
						SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
						SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
						SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
						SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
						int seed /*seed for random number generator*/) {
	auto T = X.num_rows, N = X.num_cols;
	auto T0 = T - L;
	ASSERT_(T0 == mat_reg_first.num_rows && T0 == mat_reg_first.num_cols && T0 == mat_breg.num_rows && T0 == mat_breg.num_cols);

	mat_reg_first.zero(); // reset the output matrix
	mat_reg_second.zero();
	mat_breg.zero();

	if (N == 1) {
		int t = 0, s = 0, tau = 0, ell = 0;
		double row = 0.;
		SGVector<double> labels(T0);
		SGMatrix<double> features(T0, L);

		int opt_tree_max_depth = 0, opt_num_iters = 0, opt_num_rand_feats = 0., opt_num_bags = 0.;
		double opt_learning_rate = 0, opt_subset_fraction = 0, label_mean = 0.;

		/* 1. Estimate E[x_t | lags of x_t] */

		SGVector<double> mu(T0);
		for (int t = L; t < T; ++t) {
			labels[t-L] = X(t,0);
			for (int ell = 0; ell < L; ++ell) {
				features(t-L, ell) = X(t-ell-1,0);
			}
		}
		// do hyperparameter tuning
		std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
												ML_cv(	features, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
														learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);
		// estimate conditional expectations
		mu = ML_reg(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																						opt_num_rand_feats, opt_num_bags, seed);



		/* 2. Estimate Var[x_t | lags of x_t] */

		SGVector<double> var(T0);
		var.set_const(1.);
		for (int t = L; t < T; ++t) {
			labels[t-L] = std::pow(X(t,0) - mu[t-L], 2.);
			for (int ell = 0; ell < L; ++ell) {
				features(t-L, ell) = X(t-ell-1,0);
			}
		}

		// do hyperparameter tuning
		std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
												ML_cv(	features, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
														learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);
		// estimate conditional expectations
		var = ML_reg(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																						opt_num_rand_feats, opt_num_bags, seed);



		/* 3. Compute residuals of a univariate model */

		SGVector<double> resid(T0);
		for (t = 0; t < T0; ++t) {
			resid[t] = (X(t+L,0) - mu[t]) / sqrt(var[t]);
		}

		/* 4. Estimate E[||x_t - x_s||^\kappa| lags of x_t] */

		for (t = 0; t < T0; ++t) {
			for (s = 0; s < T0; ++s) {
				for (tau = 0; tau < T0; ++tau) {
					row = mu[t] + sqrt(var[t])*resid[tau] - X(s+L,0);
					mat_reg_first(t, s) += std::pow(fabs(row), expn) / T0;
				}
			}
		}

		/* 5. Estimate E[||x_t - x_s||^\kappa| lags of x_s] */

		mat_reg_second = transpose_matrix(mat_reg_first);


		/* 6. Estimate E[||x_t - x_s||^\kappa| lags of x_t and x_s] */
		int tau1 = 0, tau2 = 0;

		#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,tau1,tau2) firstprivate(row)
		for (t = 0; t < T0; ++t) {
			for (s = t; s < T0; ++s) {
				if (T0 <= 400) {
					for (tau1 = 0; tau1 < T0; ++tau1) {
						for (tau2 = 0; tau2 < T0; ++tau2) {
							row = mu[t] + sqrt(var[t])*resid[tau1] - (mu[s] + sqrt(var[s])*resid[tau2]);
							//#pragma omp atomic
							mat_breg(t,s) += std::pow(fabs(row), expn) / pow(T0, 2.);
						}
					}
				}
				else {
					for (tau1 = 0; tau1 < T0; ++tau1) {
						row = mu[t] + sqrt(var[t])*resid[tau1] - (mu[s] + sqrt(var[s])*resid[tau1]);
						//#pragma omp atomic
						mat_breg(t,s) += std::pow(fabs(row), expn) / T0;
					}
				}
				mat_breg(s, t) = mat_breg(t,s);
			}
		}
	}
	else {
		auto t = 0, s = 0, tau = 0, ell = 0, i = 0, j = 0;
		SGVector<double> labels(T0), new_labels(T0), row_vec(N);
		SGMatrix<double> features(T0, L);

		int opt_tree_max_depth = 0, opt_num_iters = 0, opt_num_rand_feats = 0., opt_num_bags = 0.;
		double opt_learning_rate = 0, opt_subset_fraction = 0, label_mean = 0.;

		/* 1. Estimate E[x_t | lags of x_t] */

		SGMatrix<double> mu(T0, N);
		for (int i = 0; i < N; ++i) {
			for (int t = L; t < T; ++t) {
				labels[t-L] = X(t,i);
				for (int ell = 0; ell < L; ++ell) {
					features(t-L, ell) = X(t-ell-1,i);
				}
			}
			// do hyperparameter tuning
			std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
													ML_cv(	features, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
															learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);
			// estimate conditional expectations
			new_labels = ML_reg(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																							opt_num_rand_feats, opt_num_bags, seed);
			mu.set_column(i, new_labels);
		}


		/* 2. Estimate Var[x_t | lags of x_t] */

		SGMatrix<double> var(T0, N);
		var.set_const(1.);
		for (int i = 0; i < N; ++i) {
			for (int t = L; t < T; ++t) {
				labels[t-L] = std::pow(X(t,i) - mu(t-L,i), 2.);
				for (int ell = 0; ell < L; ++ell) {
					features(t-L, ell) = X(t-ell-1,i);
				}
			}

			// do hyperparameter tuning
			std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
													ML_cv(	features, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
															learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);
			// estimate conditional expectations
			new_labels = ML_reg(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																							opt_num_rand_feats, opt_num_bags, seed);
			var.set_column(i, new_labels);
		}


		/* 3. Estimate Cov[x_it, x_jt | lags of x_it and x_jt] */

		SGMatrix<double> cov(T0, N*(N-1)/2), features_c(T0, 2*L);
		cov.set_const(0.); // zero out all covariances
		int counter = 0;
		for (int i = 0; i < N; ++i) {
			for (int j = i+1; j < N; ++j) {
				for (int t = L; t < T; ++t) {
					labels[t-L] = ( X(t,i) - mu(t-L,i) ) * ( X(t,j) - mu(t-L,j) );
					for (int ell = 0; ell < L; ++ell) {
						features_c(t-L, ell) = X(t-ell-1,i);
						features_c(t-L, L+ell) = X(t-ell-1,j);
					}
				}

				// do hyperparameter tuning
				std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
														ML_cv(	features_c, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
																learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);
				// estimate conditional expectations
				new_labels = ML_reg(features_c, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																								opt_num_rand_feats, opt_num_bags, seed);
				cov.set_column(counter, new_labels);
				counter++;
			}
		}


		/* 4. Compute residuals of a multivariate model */
		SGMatrix<double> cov_mat(N, N), cov_mat_sqrt_inv(N, N), resid(N, T0);

		for (t = 0; t < T0; ++t) {
			counter = 0; // reset counter
			for (i = 0; i < N; ++i) {
				cov_mat(i, i) = var(t,i);
				for (j = i+1; j < N; ++j) {
					cov_mat(i, j) = cov(t,counter);
					cov_mat(j, i) = cov_mat(i, j);

					//#pragma omp atomic
					counter += 1;
				}
			}
			cov_mat_sqrt_inv = inv_sqrt_mat(cov_mat);
			row_vec = matrix_prod( cov_mat_sqrt_inv, add(X.get_row_vector(t+L), mu.get_row_vector(t), 1., -1.) );
			resid.set_column(t, row_vec);
		}

		SGMatrix<double> resid1 = transpose_matrix(resid); // transpose the residual matrix



		/* 5. Estimate E[||x_t - x_s||^\kappa| lags of x_t] */

		SGMatrix<double> cov_mat_sqrt(N, N);

		//#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,i,j,s,tau) firstprivate(counter,cov_mat,cov_mat_sqrt,row_vec)
		for (t = 0; t < T0; ++t) {
			counter = 0; // reset counter
			for (i = 0; i < N; ++i) {
				cov_mat(i, i) = var(t,i);
				for (j = i+1; j < N; ++j) {
					cov_mat(i, j) = cov(t,counter);
					cov_mat(j, i) = cov_mat(i, j);

					//#pragma omp atomic
					counter += 1;
				}
			}
			cov_mat_sqrt = sqrt_mat(cov_mat);

			for (s = 0; s < T0; ++s) {
				for (tau = 0; tau < T0; ++tau) {
					row_vec = add(add(mu.get_row_vector(t), matrix_prod( cov_mat_sqrt, resid1.get_row_vector(tau) ), 1., 1.), X.get_row_vector(s+L), 1., -1.);

					//#pragma omp atomic
					mat_reg_first(t, s) += std::pow(norm(row_vec), expn) / T0;
				}
			}
		}


		/* 6. Estimate E[||x_t - x_s||^\kappa| lags of x_s] */

		mat_reg_second = transpose_matrix(mat_reg_first);


		/* 7. Estimate E[||x_t - x_s||^\kappa| lags of x_t and x_s] */

		//SGMatrix<double> cov_mat_t(N, N), cov_mat_s(N, N), cov_mat_sqrt_t(N, N), cov_mat_sqrt_s(N, N);
		int i1 = 0, j1 = 0, i2 = 0, j2 = 0, counter_t = 0, counter_s = 0, tau1 = 0, tau2 = 0;

		#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,i1,j1,i2,j2,tau1,tau2) firstprivate(counter_t,counter_s)
		for (t = 0; t < T0; ++t) {
			SGMatrix<double> cov_mat_t(N, N), cov_mat_sqrt_t(N, N), cov_mat_s(N, N), cov_mat_sqrt_s(N, N);
			SGVector<double> row_vec(N);
			//#pragma omp critical
			{
				counter_t = 0; // reset counter
				for (i1 = 0; i1 < N; ++i1) {
					cov_mat_t(i1, i1) = var(t, i1);
					for (j1 = i1+1; j1 < N; ++j1) {
						cov_mat_t(i1, j1) = cov(t, counter_t);
						cov_mat_t(j1, i1) = cov_mat_t(i1, j1);

						//#pragma omp atomic
						counter_t += 1;
					}
				}
			}
			cov_mat_sqrt_t = sqrt_mat(cov_mat_t);


			for (s = t; s < T0; ++s) {
				//#pragma omp critical
				{
					counter_s = 0; // reset counter
					for (i2 = 0; i2 < N; ++i2) {
						cov_mat_s(i2, i2) = var(s,i2);
						for (j2 = i2+1; j2 < N; ++j2) {
							cov_mat_s(i2, j2) = cov(s, counter_s);
							cov_mat_s(j2, i2) = cov_mat_s(i2, j2);

							//#pragma omp atomic
							counter_s += 1;
						}
					}
				}
				cov_mat_sqrt_s = sqrt_mat(cov_mat_s);

				if (T0 <= 400) {
					for (tau1 = 0; tau1 < T0; ++tau1) {
						for (tau2 = 0; tau2 < T0; ++tau2) {
							row_vec = add(add(mu.get_row_vector(t), matrix_prod( cov_mat_sqrt_t, resid1.get_row_vector(tau1) ), 1., 1.), \
											add(mu.get_row_vector(s), matrix_prod( cov_mat_sqrt_s, resid1.get_row_vector(tau2) ), 1., 1.), 1., -1.);
							//#pragma omp atomic
							mat_breg(t,s) += std::pow(norm(row_vec), expn) / pow(T0, 2.);
						}
					}
				}
				else {
					for (tau1 = 0; tau1 < T0; ++tau1) {
						row_vec = add(add(mu.get_row_vector(t), matrix_prod( cov_mat_sqrt_t, resid1.get_row_vector(tau1) ), 1., 1.), \
										add(mu.get_row_vector(s), matrix_prod( cov_mat_sqrt_s, resid1.get_row_vector(tau1) ), 1., 1.), 1., -1.);
						//#pragma omp atomic
						mat_breg(t,s) += std::pow(norm(row_vec), expn) / T0;
					}
				}
				mat_breg(s, t) = mat_breg(t,s);
			}
		}
	}
}



/* Calculate the conditional expectations of  ||x_t - x_s||^\alpha (conditioning on 'L' lags of x_t or/and x_s) for t = 1, ..., T0 and s = 1, ..., T0 by
a ML regression algorithm applied to model (2.15).
INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
OUTPUT: a T0 = T-L by T-L matrix (mat_reg_first, mat_reg_second, mat_breg). */
template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																			const SGVector<double> &, /*train labels*/
																			int, /*number of subsets for TSCV*/
																			int, /*minimum subset size for TSCV*/
																			SGVector<int>, /*list of tree max depths (for GBM)*/
																			SGVector<int>, /*list of numbers of iterations (for GBM)*/
																			SGVector<double>, /*list of learning rates (for GBM)*/
																			SGVector<double>, /*list of subset fractions (for GBM)*/
																			SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int>, /*list of number of bags (for RF)*/
																			int seed /*seed for random number generator*/),
		SGVector<double> ML_reg(const SGMatrix<double> &,
								const SGVector<double> &,
								int, /*tree max depth*/
								int, /*number of iterations*/
								double, /*learning rate*/
								double, /*subset fraction*/
								int, /*number of random features used for bagging (for RF)*/
								int, /*number of bags (for RF)*/
								int /*seed for the random number generator*/)>
void ML_REG_DCORR::reg1(SGMatrix<double> &mat_reg_first, /*E[||x_t - x_s||^\kappa| lags of x_t]*/
						SGMatrix<double> &mat_reg_second, /*E[||x_t - x_s||^\kappa| lags of x_s]*/
						SGMatrix<double> &mat_breg, /*E[||x_t - x_s||^\kappa| lags of x_t and x_s]*/
						const SGMatrix<double> &X,
						const int L, /*maximum truncation lag*/
						const double expn, /*exponent of distances*/
						const double expn_x, /*exponent of data*/
						int num_subsets, /*number of subsets for TSCV*/
						int min_subset_size, /*minimum subset size for TSCV*/
						SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
						SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
						SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
						SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
						SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
						SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
						int seed /*seed for random number generator*/) {
	auto T = X.num_rows, N = X.num_cols;
	auto T0 = T - L;
	ASSERT_(T0 == mat_reg_first.num_rows && T0 == mat_reg_first.num_cols && T0 == mat_breg.num_rows && T0 == mat_breg.num_cols);

	mat_reg_first.zero(); // reset the output matrix
	mat_reg_second.zero();
	mat_breg.zero();

	auto t = 0, s = 0, tau = 0, ell = 0, i = 0, j = 0;
	SGVector<double> labels(T0), new_labels(T0), row_vec(N);
	SGMatrix<double> features(T0, L);

	int opt_tree_max_depth = 0, opt_num_iters = 0, opt_num_rand_feats = 0., opt_num_bags = 0.;
	double opt_learning_rate = 0, opt_subset_fraction = 0, label_mean = 0.;

	/* 1. Estimate E[x_t | lags of x_t] */

	SGMatrix<double> mu(T0, N);
	for (int i = 0; i < N; ++i) {
		for (int t = L; t < T; ++t) {
			labels[t-L] = X(t,i);
			for (int ell = 0; ell < L; ++ell) {
				features(t-L, ell) = X(t-ell-1,i);
			}
		}
		// do hyperparameter tuning
		std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
												ML_cv(	features, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
														learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);
		// estimate conditional expectations
		new_labels = ML_reg(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																						opt_num_rand_feats, opt_num_bags, seed);
		mu.set_column(i, new_labels);
	}


	/* 2. Estimate Var[x_t | lags of x_t] */

	SGMatrix<double> var(T0, N);
	for (int i = 0; i < N; ++i) {
		for (int t = L; t < T; ++t) {
			labels[t-L] = std::pow(X(t,i) - mu(t-L,i), 2.);
			for (int ell = 0; ell < L; ++ell) {
				features(t-L, ell) = X(t-ell-1,i);
			}
		}
		// do hyperparameter tuning
		std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
												ML_cv(	features, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
														learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);
		// estimate conditional expectations
		new_labels = ML_reg(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																						opt_num_rand_feats, opt_num_bags, seed);
		var.set_column(i, new_labels);
	}


	/* 3. Estimate Cov[x_it, x_jt | lags of x_it and x_jt] */

	SGMatrix<double> cov(T0, N*(N-1)/2), features_c(T0, 2*L);
	int counter = 0;
	for (int i = 0; i < N; ++i) {
		for (int j = i+1; j < N; ++j) {
			for (int t = L; t < T; ++t) {
				labels[t-L] = ( X(t,i) - mu(t-L,i) ) * ( X(t,j) - mu(t-L,j) );
				for (int ell = 0; ell < L; ++ell) {
					features_c(t-L, ell) = X(t-ell-1,i);
					features_c(t-L, L+ell) = X(t-ell-1,j);
				}
			}

			// do hyperparameter tuning
			std::tie(opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, opt_num_rand_feats, opt_num_bags) =
													ML_cv(	features_c, labels, num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, \
															learning_rates_list, subset_fractions_list, num_rand_feats_list, num_bags_list, seed);
			// estimate conditional expectations
			new_labels = ML_reg(features_c, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, \
																							opt_num_rand_feats, opt_num_bags, seed);
			cov.set_column(counter, new_labels);
			counter++;
		}
	}


	/* 4. Compute residuals of a multivariate model */
	SGMatrix<double> cov_mat(N, N), cov_mat_sqrt_inv(N, N), resid(N, T0);

	for (t = 0; t < T0; ++t) {
		counter = 0; // reset counter
		for (i = 0; i < N; ++i) {
			cov_mat(i, i) = var(t,i);
			for (j = i+1; j < N; ++j) {
				cov_mat(i, j) = cov(t,counter);
				cov_mat(j, i) = cov_mat(i, j);

				//#pragma omp atomic
				counter += 1;
			}
		}
		cov_mat_sqrt_inv = inv_sqrt_mat(cov_mat);
		row_vec = matrix_prod( cov_mat_sqrt_inv, add(X.get_row_vector(t+L), mu.get_row_vector(t), 1., -1.) );
		resid.set_column(t, row_vec);
	}

	SGMatrix<double> resid1 = transpose_matrix(resid); // transpose the residual matrix



	/* 5. Estimate E[||x_t - x_s||^\kappa| lags of x_t] */

	SGMatrix<double> cov_mat_sqrt(N, N);

	//#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,i,j,s,tau) firstprivate(counter,cov_mat,cov_mat_sqrt,row_vec)
	for (t = 0; t < T0; ++t) {
		counter = 0; // reset counter
		for (i = 0; i < N; ++i) {
			cov_mat(i, i) = var(t,i);
			for (j = i+1; j < N; ++j) {
				cov_mat(i, j) = cov(t,counter);
				cov_mat(j, i) = cov_mat(i, j);

				//#pragma omp atomic
				counter += 1;
			}
		}
		cov_mat_sqrt = sqrt_mat(cov_mat);

		for (s = 0; s < T0; ++s) {
			for (tau = 0; tau < T0; ++tau) {
				row_vec = add(pow_vec(add(mu.get_row_vector(t), matrix_prod( cov_mat_sqrt, resid1.get_row_vector(tau) ), 1., 1.), expn_x), \
							  pow_vec(X.get_row_vector(s+L), expn_x), 1., -1.);

				//#pragma omp atomic
				mat_reg_first(t, s) += std::pow(norm(row_vec), expn) / T0;
			}
		}
	}


	/* 6. Estimate E[||x_t - x_s||^\kappa| lags of x_s] */

	mat_reg_second = transpose_matrix(mat_reg_first);


	/* 7. Estimate E[||x_t - x_s||^\kappa| lags of x_t and x_s] */

	SGMatrix<double> cov_mat_sqrt_t(N, N), cov_mat_sqrt_s(N, N);

	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s,i,j,tau) firstprivate(counter,cov_mat,cov_mat_sqrt_t,cov_mat_sqrt_s,row_vec)
	for (t = 0; t < T0; ++t) {
		#pragma omp critical
		{
			counter = 0; // reset counter
			for (i = 0; i < N; ++i) {
				cov_mat(i, i) = var(t,i);
				for (j = i+1; j < N; ++j) {
					cov_mat(i, j) = cov(t,counter);
					cov_mat(j, i) = cov_mat(i, j);

					//#pragma omp atomic
					counter += 1;
				}
			}
			cov_mat_sqrt_t = sqrt_mat(cov_mat);

			for (s = 0; s < T0; ++s) {
				counter = 0; // reset counter
				for (i = 0; i < N; ++i) {
					cov_mat(i, i) = var(s,i);
					for (j = i+1; j < N; ++j) {
						cov_mat(i, j) = cov(s,counter);
						cov_mat(j, i) = cov_mat(i, j);

						//#pragma omp atomic
						counter += 1;
					}
				}
				cov_mat_sqrt_s = sqrt_mat(cov_mat);

				for (tau = 0; tau < T0; ++tau) {
					row_vec = add(pow_vec(add(mu.get_row_vector(t), matrix_prod( cov_mat_sqrt_t, resid1.get_row_vector(tau) ), 1., 1.), expn_x), \
								  pow_vec(add(mu.get_row_vector(s), matrix_prod( cov_mat_sqrt_s, resid1.get_row_vector(tau) ), 1., 1.), expn_x), 1., -1.);
					//#pragma omp atomic
					mat_breg(t, s) += std::pow(norm(row_vec), expn) / T0;
				}
			}
		}
	}
}



/* Calculate centered Euclidean distances by ML methods.
INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
OUTPUT: a T0 = T-L by T-L matrix (mat_U). */
template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																			const SGVector<double> &, /*train labels*/
																			int, /*number of subsets for TSCV*/
																			int, /*minimum subset size for TSCV*/
																			SGVector<int>, /*list of tree max depths (for GBM)*/
																			SGVector<int>, /*list of numbers of iterations (for GBM)*/
																			SGVector<double>, /*list of learning rates (for GBM)*/
																			SGVector<double>, /*list of subset fractions (for GBM)*/
																			SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int>, /*list of number of bags (for RF)*/
																			int seed /*seed for random number generator*/),
		SGVector<double> ML_reg(const SGMatrix<double> &,
								const SGVector<double> &,
								int, /*tree max depth*/
								int, /*number of iterations*/
								double, /*learning rate*/
								double, /*subset fraction*/
								int, /*number of random features used for bagging (for RF)*/
								int, /*number of bags (for RF)*/
								int /*seed for the random number generator*/)>
SGMatrix<double> ML_REG_DCORR::var_U(const SGMatrix<double> &X, const int L, const double expn,
									int num_subsets, /*number of subsets for TSCV*/
									int min_subset_size, /*minimum subset size for TSCV*/
									SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
									SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
									SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
									SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
									SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
									SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
									int seed /*seed for random number generator*/) {
	auto T = X.num_rows, N = X.num_cols;
	auto T0 = T - L;
	SGMatrix<double> mat_reg_first(T0, T0), mat_reg_second(T0, T0), mat_breg(T0, T0), mat_U(T0, T0);
	mat_U.zero(); // reset the output matrix

	// Calculate conditional expectations
	ML_REG_DCORR::reg1 <ML_cv, ML_reg> (mat_reg_first, mat_reg_second, mat_breg, X, L, expn,
										num_subsets, /*number of subsets for TSCV*/
										min_subset_size, /*minimum subset size for TSCV*/
										tree_max_depths_list, /*list of tree max depths (for GBM)*/
										num_iters_list, /*list of numbers of iterations (for GBM)*/
										learning_rates_list, /*list of learning rates (for GBM)*/
										subset_fractions_list, /*list of subset fractions (for GBM)*/
										num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
										num_bags_list, /*list of number of bags (for RF)*/
										seed);

	// Calculate 'mat_U'
	int t = 0, s = 0;
	//SGVector<double> row_vec(N);

	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s)
	for (t = L; t < T; ++t) {
		for (s = L; s < T; ++s) {
			//row_vec = add(X.get_row_vector(t), X.get_row_vector(s), 1., -1.);
			mat_U(t-L, s-L) = std::pow(shogun::linalg::norm( add(X.get_row_vector(t), X.get_row_vector(s), 1., -1.) ), expn) - mat_reg_first(t-L, s-L) \
																										- mat_reg_second(t-L, s-L) + mat_breg(t-L, s-L);
			//mat_U(s-L, t-L) = mat_U(t-L,s-L);
		}
	}

	for (t = 0; t < T0; ++t) {
		mat_U(t, t) = 0.; //zero out all the diagonal elements
	}

	return mat_U;
}

/* Calculate centered Euclidean distances by ML methods.
INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
OUTPUT: a T0 = T-L by T-L matrix (mat_U). */
template<std::tuple<int, int, double, double, int, int> ML_cv(	const SGMatrix<double> &, /*columns are features*/
																			const SGVector<double> &, /*train labels*/
																			int, /*number of subsets for TSCV*/
																			int, /*minimum subset size for TSCV*/
																			SGVector<int>, /*list of tree max depths (for GBM)*/
																			SGVector<int>, /*list of numbers of iterations (for GBM)*/
																			SGVector<double>, /*list of learning rates (for GBM)*/
																			SGVector<double>, /*list of subset fractions (for GBM)*/
																			SGVector<int>, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int>, /*list of number of bags (for RF)*/
																			int seed /*seed for random number generator*/),
		SGVector<double> ML_reg(const SGMatrix<double> &,
								const SGVector<double> &,
								int, /*tree max depth*/
								int, /*number of iterations*/
								double, /*learning rate*/
								double, /*subset fraction*/
								int, /*number of random features used for bagging (for RF)*/
								int, /*number of bags (for RF)*/
								int /*seed for the random number generator*/)>
SGMatrix<double> ML_REG_DCORR::var_U(const SGMatrix<double> &X,
									const int L, /*maximum truncation lag*/
									const double expn, /*exponent of distances*/
									const double expn_x, /*exponent of data*/
									int num_subsets, /*number of subsets for TSCV*/
									int min_subset_size, /*minimum subset size for TSCV*/
									SGVector<int> tree_max_depths_list, /*list of tree max depths (for GBM)*/
									SGVector<int> num_iters_list, /*list of numbers of iterations (for GBM)*/
									SGVector<double> learning_rates_list, /*list of learning rates (for GBM)*/
									SGVector<double> subset_fractions_list, /*list of subset fractions (for GBM)*/
									SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
									SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
									int seed /*seed for random number generator*/) {
	auto T = X.num_rows, N = X.num_cols;
	auto T0 = T - L;
	SGMatrix<double> mat_reg_first(T0, T0), mat_reg_second(T0, T0), mat_breg(T0, T0), mat_U(T0, T0);
	mat_U.zero(); // reset the output matrix

	// Calculate conditional expectations
	ML_REG_DCORR::reg1 <ML_cv, ML_reg> (mat_reg_first, mat_reg_second, mat_breg, X, L, expn, expn_x,
										num_subsets, /*number of subsets for TSCV*/
										min_subset_size, /*minimum subset size for TSCV*/
										tree_max_depths_list, /*list of tree max depths (for GBM)*/
										num_iters_list, /*list of numbers of iterations (for GBM)*/
										learning_rates_list, /*list of learning rates (for GBM)*/
										subset_fractions_list, /*list of subset fractions (for GBM)*/
										num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
										num_bags_list, /*list of number of bags (for RF)*/
										seed);

	// Calculate 'mat_U'
	int t = 0, s = 0;
	//SGVector<double> row_vec(N);

	//#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s)
	for (t = L; t < T; ++t) {
		for (s = L; s < T; ++s) {
			//#pragma omp critical
			{
				if (t != s) {
					//row_vec = add(X.get_row_vector(t), X.get_row_vector(s), 1., -1.);
					mat_U(t-L, s-L) = std::pow(shogun::linalg::norm( add(pow_vec(X.get_row_vector(t), expn_x), pow_vec(X.get_row_vector(s), expn_x), 1., -1.) ), expn)
										- mat_reg_first(t-L, s-L) - mat_reg_second(t-L, s-L) + mat_breg(t-L, s-L);
					//mat_U(s-L, t-L) = mat_U(t-L,s-L);
				}
			}
		}
	}
	return mat_U;
}



/* Calculate centered Euclidean distances by moving averages.
INPUT: T by N matrix of data (X), a number of lags (L), and an exponent of the distance correlation (expn).
OUTPUT: a T0 = T-L by T-L matrix (mat_U). */
SGMatrix<double> ML_REG_DCORR::var_U(	const SGMatrix<double> &X,
										const int L,
										const double expn) {
	auto T = X.num_rows, N = X.num_cols;
	auto T0 = T - L;
	SGMatrix<double> mat_reg_first(T0, T0), mat_reg_second(T0, T0), mat_breg(T0, T0), mat_U(T0, T0);
	mat_U.zero(); // reset the output matrix

	// Calculate conditional expectations
	ML_REG_DCORR::reg(mat_reg_first, mat_reg_second, mat_breg, X, L, expn);

	// Calculate 'mat_U'
	int t = 0, s = 0;
	//SGVector<double> row_vec(N);

	#pragma omp parallel for default(shared) schedule(dynamic,CHUNK) private(t,s)
	for (t = L; t < T; ++t) {
		for (s = t+1; s < T; ++s) {
			//row_vec = add(X.get_row_vector(t), X.get_row_vector(s), 1., -1.);
			mat_U(t-L, s-L) = std::pow(shogun::linalg::norm( add(X.get_row_vector(t), X.get_row_vector(s), 1., -1.) ), expn) - mat_reg_first(t-L, s-L) \
																										- mat_reg_second(t-L, s-L) + mat_breg(t-L, s-L);
			mat_U(s-L, t-L) = mat_U(t-L,s-L);
		}
	}

	return mat_U;
}












#endif
