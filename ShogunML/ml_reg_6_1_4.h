#ifndef ML_REG_H
#define ML_REG_H

#include <omp.h>
#include <ShogunML/data/data.h>
#include <plot.h>
#include "utils.h"


#include <shogun/lib/config.h>
//#include <shogun/base/init.h>
//#include <shogun/base/some.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/ensemble/CombinationRule.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/multiclass/tree/CARTree.h>
#include <shogun/util/factory.h>
#include <shogun/machine/Machine.h>
#include <shogun/multiclass/tree/ConditionalProbabilityTree.h>
//#include <shogun/evaluation/CrossValidation.h>
//#include <shogun/evaluation/SplittingStrategy.h>
//#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
//#include <shogun/modelselection/GridSearchModelSelection.h>
//#include <shogun/modelselection/ModelSelectionParameters.h>
//#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <shogun/transformer/Transformer.h>
#include <shogun/io/SGIO.h>
#include <shogun/regression/svr/LibSVR.h>
#include <shogun/preprocessor/PCA.h>

#include <experimental/filesystem>
#include <iostream>
#include <map>
#include <ShogunML/tscv.h>

// namespace fs = std::experimental::filesystem;
using namespace std;
using namespace shogun;
using namespace shogun::linalg;
using namespace shogun::io;



using MatrixSG = SGMatrix<DataType>;
using Vector = SGVector<DataType>;
using Coords =  std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<index_t, PointCoords>;


class ML_REG {
	public:
		ML_REG () {   }; //default constructor
		~ML_REG () {   };//default destructor
		static std::pair<MatrixSG, Vector> GenerateShogunData(double s, double e, int n, int dim, int seed);

		static void PlotResults(SGMatrix<double> x_coords, SGVector<double> y_coords,SGVector<double> y_pred_coords,
								const std::string& title, const std::string& file_name);

		static void PlotClusters(const Clusters& clusters, const std::string& name, const std::string& file_name);

		/* Perform GBM and plot the predicted results */
		static void GBM_Plot(	const SGMatrix<double> &features, const SGVector<double> &labels, const SGMatrix<double> &test_features,
								const SGVector<double> &test_labels, int tree_max_depth, int num_iters, double learning_rate,
								double subset_fraction, int seed);

		static SGVector<double> GBM_Plot(const SGMatrix<double> &features, const SGVector<double> &labels, int tree_max_depth, int num_iters, \
											double learning_rate, double subset_fraction, int num_rand_feats, int num_bags, int seed);

		static double GBM(	const SGMatrix<double> &features_train, const SGVector<double> &labels_train, const SGMatrix<double> &features_valid,
							const SGVector<double> &labels_valid, int tree_max_depth, int num_iters, double learning_rate, double subset_fraction,
							int seed);

		/* Perform GBM with cross validation */
		static void GBMRegression(	const SGMatrix<double> &features, /*columns are features*/
									const SGVector<double> &labels, const SGMatrix<double> &test_features, const SGVector<double> &test_labels,
									int num_subsets, int min_subset_size, SGVector<int> tree_max_depths_list, SGVector<int> num_iters_list,
									SGVector<double> learning_rates_list, SGVector<double> subset_fractions_list, int seed);

		static SGVector<double> GBMRegression(	const SGMatrix<double> &features, /*columns are features*/
												const SGVector<double> &labels,
												int num_subsets, int min_subset_size,
												SGVector<int> tree_max_depths_list,
												SGVector<int> num_iters_list,
												SGVector<double> learning_rates_list,
												SGVector<double> subset_fractions_list,
												SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
												SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
												int seed);

		/* Perform GBM CV
		OUTPUT: optimal tree maximum depth, number of iterations, learning rate, subset fraction, number of random features, number of bags */
		static std::tuple<int, int, double, double, int, int> GBM_cv(	const SGMatrix<double> &features, /*columns are features*/
																		const SGVector<double> &labels,
																		int num_subsets, int min_subset_size,
																		SGVector<int> tree_max_depths_list,
																		SGVector<int> num_iters_list,
																		SGVector<double> learning_rates_list,
																		SGVector<double> subset_fractions_list,
																		SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
																		SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
																		int seed);
		/* Do Random Forest (RF) regression and plot the results */
		static void RF_Plot(const SGMatrix<double> &features, const SGVector<double> &labels, const SGMatrix<double> &test_features,
							const SGVector<double> &test_labels, int num_rand_feats, int num_bags, int seed);

		/* Do RF regression */
		static SGVector<double> RF_Plot(const SGMatrix<double> &features,
										const SGVector<double> &labels,
										int tree_max_depth,
										int num_iters,
										double learning_rate,
										double subset_fraction,
										int num_rand_feats, /*number of random features used for bagging (for RF)*/
										int num_bags, /*number of bags (for RF)*/
										int seed);

		/* Compute MSE on test data with Random Forest */
		static double RF_MSE(	const SGMatrix<double> &features_train,
								const SGVector<double> &labels_train,
								const SGMatrix<double> &features_valid,
								const SGVector<double> &labels_valid,
								int num_rand_feats,
								int num_bags,
								int seed);

		static void RFRegression(	const SGMatrix<double> &features, /*columns are features*/
									const SGVector<double> &labels,
									const SGMatrix<double> &test_features,
									const SGVector<double> &test_labels,
									int num_subsets, int min_subset_size,
									SGVector<int> num_rand_feats_list,
									SGVector<int> num_bags_list,
									int seed);

		/* Perform RF CV with time-series splitting.
		OUTPUT: optimal tree maximum depth, number of iterations, learning rate, subset fraction, number of random features, number of bags */
		static std::tuple<int, int, double, double, int, int> RF_cv(const SGMatrix<double> &features, /*columns are features*/
																	const SGVector<double> &labels,
																	int num_subsets, int min_subset_size,
																	SGVector<int> tree_max_depths_list,
																	SGVector<int> num_iters_list,
																	SGVector<double> learning_rates_list,
																	SGVector<double> subset_fractions_list,
																	SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
																	SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
																	int seed);

		/* Select RF hyperparameters with mininum MSE.
		OUTPUT: optimal tree maximum depth, number of iterations, learning rate, subset fraction, number of random features, number of bags */
		static std::tuple<int, int, double, double, int, int> RF_cv1(const SGMatrix<double> &features, /*columns are features*/
																			const SGVector<double> &labels,
																			int num_subsets, int min_subset_size,
																			SGVector<int> tree_max_depths_list,
																			SGVector<int> num_iters_list,
																			SGVector<double> learning_rates_list,
																			SGVector<double> subset_fractions_list,
																			SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
																			SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
																			int seed);

		/* Train and test the support vector regression */
		static void SVR_Plot( 	const SGMatrix<double> &features,
								const SGVector<double> &labels,
								const SGMatrix<double> &test_features,
								const SGVector<double> &test_labels,
								double kernel_width, double C1,
								double C2, double tube_epsilon,
								int svr_solver_type);

		/* Compute the mse on the validation dataset using the support vector machine regression */
		static double SVR(	const SGMatrix<double> &features_train, const SGVector<double> &labels_train,
							const SGMatrix<double> &features_valid, const SGVector<double> &labels_valid,
							double kernel_width, double C1, double C2, double tube_epsilon, int svr_solver_type);

		/* Implement the support vector regression with cross-validation */
		static void SVRegression(	const SGMatrix<double> &features, /*columns are features*/
									const SGVector<double> &labels,
									const SGMatrix<double> &test_features,
									const SGVector<double> &test_labels,
									int num_subsets, int min_subset_size,
									SGVector<double> kernel_widths_list,
									SGVector<double> C1_list,
									SGVector<double> C2_list,
									SGVector<double> tube_epsilons_list);

		/* Train and test the least angle regression */
		static void LARS_Plot(	const SGMatrix<double> &features,
								const SGVector<double> &labels,
								const SGMatrix<double> &test_features,
								const SGVector<double> &test_labels,
								double max_l1_norm);

		/* Train and test the kernel ridge regression */
		static void KRR_Plot(	const SGMatrix<double> &features,
								const SGVector<double> &labels,
								const SGMatrix<double> &test_features,
								const SGVector<double> &test_labels,
								double kernel_width, double tau);

		/* Compute normalized principal components for a feature matrix */
		static std::pair<SGMatrix<double>, SGMatrix<double>> calcul_PCs(const SGMatrix<double> &features_train,
																		const SGMatrix<double> &features_test,
																		int target_dim);

	private:
		static bool isnotZero(double x);
		static int num_threads;
		static std::vector<std::string> colors;
};

int ML_REG::num_threads = omp_get_num_procs(); // set the number of threads for multiprocessing ( e.g., omp_get_num_procs() )

std::vector<std::string> ML_REG::colors{"black", "red", "blue", "green", "cyan"};


/* Define a predicate to select nonzero elements from a SG vector */
bool ML_REG::isnotZero(double x) {
	return ( (x > 0) || (x < 0) );
}

/*
std::pair<MatrixSG, Vector> GenerateShogunData(double s,
                                             double e,
                                             size_t n,
                                             size_t seed,
                                             bool noise) {
  Values x, y;
  std::tie(x, y) = GenerateData(s, e, n, seed, noise);
  MatrixSG x_values(1, static_cast<int>(n));
  Vector y_values(static_cast<int>(n));

  for (size_t i = 0; i < n; ++i) {
    x_values.set_element(x[i], 0, static_cast<int>(i));
    y_values.set_element(y[i], static_cast<int>(i));
  }
  return {x_values, y_values};
} */

std::pair<MatrixSG, Vector> ML_REG::GenerateShogunData(	double s,
														double e,
														int n,
														int dim,
														int seed) {
  Matrix x(dim, n), y(1, n);
  std::tie(x, y) = GenerateData(s, e, n, dim, seed);

  MatrixSG x_values(dim, n);
  Vector y_values(n);

  for (int i = 0; i < n; ++i) {
  	for (int j = 0; j < dim; ++j) {
		x_values.set_element(x(j+1, i+1),  j, i);
  	}
	y_values.set_element(y(1, i+1),  i);
  }
  return {x_values, y_values};
}


/*==================================================== Plot Data ==========================================================================*/

void ML_REG::PlotResults(	SGMatrix<double> x_coords,
							SGVector<double> y_coords,
							SGVector<double> y_pred_coords,
							const std::string& title,
							const std::string& file_name) {

  plotcpp::Plot plt;

    //plot a test feature vs the test label
  for (int i = 0; i < 1 /*x_coords.num_rows*/; i++) {
     plt.SetTerminal("png");
     plt.SetOutput(file_name +  "_feat_" + std::to_string(i) + ".png");
     plt.SetTitle(title);
     plt.SetXLabel("x");
     plt.SetYLabel("y");
     plt.SetAutoscale();
     plt.GnuplotCommand("set grid");

     SGVector<DataType> x_coord0 =  x_coords.get_column(i);
     plt.Draw2D(
        plotcpp::Points(x_coord0.begin(), x_coord0.end(), y_coords.begin(),
                      "actual", "lc rgb 'black' ps 1.5 pt 7"),
        /*plotcpp::Lines(x_coord0.begin(), x_coord0.end(), y_pred_coords.begin(),
                     "pred", "lc rgb 'red' lw 2") );*/
        plotcpp::Points(x_coord0.begin(), x_coord0.end(), y_pred_coords.begin(),
					  "predicted", "lc rgb 'red' ps 1 pt 5")
		);
  }
  plt.Flush();
}


void ML_REG::PlotClusters(	const Clusters& clusters,
							const std::string& name,
							const std::string& file_name) {
	plotcpp::Plot plt(true);
	plt.SetTerminal("png");
	// plt.SetTerminal("qt");
	plt.SetOutput(file_name);
	plt.SetTitle(name);
	plt.SetXLabel("x");
	plt.SetYLabel("y");
	// plt.SetAutoscale();
	plt.GnuplotCommand("set size square");
	plt.GnuplotCommand("set grid");

	auto draw_state = plt.StartDraw2D<Coords::const_iterator>();
	for (auto& cluster : clusters) {
	std::stringstream params;
	params << "lc rgb '" << colors[cluster.first] << "' pt 7";
	plt.AddDrawing(draw_state,
				   plotcpp::Points(
					   cluster.second.first.begin(), cluster.second.first.end(),
					   cluster.second.second.begin(),
					   std::to_string(cluster.first) + " cls", params.str()));
	}

	plt.EndDraw2D(draw_state);
	plt.Flush();
}


/*================================================== Supervised PCA ===================================================================================*/

/* Compute normalized principal components for a feature matrix */
std::pair<SGMatrix<double>, SGMatrix<double>> ML_REG::calcul_PCs(	const SGMatrix<double> &features_train,
							const SGMatrix<double> &features_test, int target_dim) {
	//![create_features]
	auto SGfeatures_train = create<Features>( transpose_matrix(features_train) );
	auto SGfeatures_test = create<Features>( transpose_matrix(features_test) );
	//![create_features]

	//![create preprocessor]
	bool do_whitening = true;
	EPCAMode mode = FIXED_NUMBER;
	double thresh = 1e-6;
	EPCAMethod method = EVD;
	EPCAMemoryMode mem_mode=MEM_REALLOCATE;

	std::unique_ptr<PCA> preproc( new PCA(do_whitening, mode, thresh, method, mem_mode) );
	preproc->set_target_dim(target_dim);
	preproc->fit(SGfeatures_train);
	auto p = preproc->get_global_parallel();
	p->set_num_threads(num_threads);

	//![transform_features]
	auto SGfeatures_train_trf = preproc->transform(SGfeatures_train);
	auto SGfeatures_test_trf = preproc->transform(SGfeatures_test);
	auto SGfeatures_train_trf_matrix = transpose_matrix( SGfeatures_train_trf->get<SGMatrix<float64_t>>("feature_matrix") );
	auto SGfeatures_test_trf_matrix = transpose_matrix( SGfeatures_test_trf->get<SGMatrix<float64_t>>("feature_matrix") );
	//![transform_features]

	SGfeatures_train_trf_matrix.display_matrix("train");
	cout << "number of train features after dimensionality reduction = " << SGfeatures_train_trf_matrix.num_cols << endl;
	SGfeatures_test_trf_matrix.display_matrix("test");
	cout << "number of test features after dimensionality reduction = " << SGfeatures_test_trf_matrix.num_cols << endl;

	ML_REG::PlotResults(SGfeatures_train_trf_matrix, SGfeatures_train_trf_matrix.get_column(1),
						SGfeatures_train_trf_matrix.get_column(1), "Shogun PCA: Train", "shogun-pca_train");

	return {SGfeatures_train_trf_matrix, SGfeatures_test_trf_matrix};
}


/*============================================== Kernel Ridge Regression ===============================================================================*/

/* Train and test the kernel ridge regression */
void ML_REG::KRR_Plot(const SGMatrix<double> &features,
						const SGVector<double> &labels,
						const SGMatrix<double> &test_features,
						const SGVector<double> &test_labels,
						double kernel_width, double tau) {

	//auto features_train = create<Features>( transpose_matrix(features) );
	auto features_train = create<Features>( transpose_matrix(features) );
	//cout << "(num_vectors, num_features) = " << features_train->get_num_vectors() << " , " <<  features_train->get_num_features() << endl;
	auto features_test = create<Features>( transpose_matrix(test_features) );
	//auto features_test = DenseFeatures<double>::obtain_from_generic( create<Features>( transpose_matrix(test_features) ) );
	auto labels_train = create<Labels>(labels);
	auto labels_test =  create<Labels>(test_labels);


	//![create_appropriate_kernel]
	auto kernel = create<Kernel>("GaussianKernel");
	kernel->put("width", kernel_width); // set a postive value for 'kernel width'
	//![create_appropriate_kernel]

	//![create_instance]
	auto krr = create<Machine>("KernelRidgeRegression");
	krr->put("labels", labels_train);
	krr->put("tau", tau); // set a small 'tau'
	krr->put("kernel", kernel);
	//![create_instance]

	//![train model]
	krr->train(features_train);

	// evaluate model on train data
	auto new_labels_train = krr->apply_regression(features_train);
	PlotResults(features, labels, new_labels_train->get_labels(), "Shogun KRR: Train", "shogun-krr-train");

	// evaluate model on test data
	auto new_labels = krr->apply_regression(features_test);

	auto eval_criterium = create<Evaluation>("MeanSquaredError");
	auto accuracy = eval_criterium->evaluate(new_labels, labels_test);
	std::cout << "KRR mse = " << accuracy << std::endl;

	PlotResults(test_features, test_labels, new_labels->get_labels(), "Shogun KRR: Test", "shogun-krr-test");
}


/*=============================== Efron, Hastie, Johnstone, and Tibshirani's (2004) Least Angle Regresssion ============================================*/

/* Train and test the least angle regression */
void ML_REG::LARS_Plot(const SGMatrix<double> &features,
						const SGVector<double> &labels,
						const SGMatrix<double> &test_features,
						const SGVector<double> &test_labels,
						double max_l1_norm) {

	//auto features_train = create<Features>( transpose_matrix(features) );
	auto features_train = create<Features>( transpose_matrix(features) );
	//cout << "(num_vectors, num_features) = " << features_train->get_num_vectors() << " , " <<  features_train->get_num_features() << endl;
	auto features_test = create<Features>( transpose_matrix(test_features) );
	//auto features_test = DenseFeatures<double>::obtain_from_generic( create<Features>( transpose_matrix(test_features) ) );
	auto labels_train = create<Labels>(labels);
	auto labels_test =  create<Labels>(test_labels);

	//![preprocess_features]
	auto SubMean = create<Transformer>("PruneVarSubMean");
	auto Normalize = create<Transformer>("NormOne");
	SubMean->fit(features_train);
	auto pruned_features_train = SubMean->transform(features_train);
	auto pruned_features_test = SubMean->transform(features_test);
	Normalize->fit(features_train);
	auto normalized_features_train = Normalize->transform(pruned_features_train);
	auto normalized_features_test = Normalize->transform(pruned_features_test);
	//![preprocess_features]


	//![create_instance]
	auto lars = create<Machine>("LeastAngleRegression");
	auto p = lars->get_global_parallel();
	p->set_num_threads(num_threads);
	lars->put("labels", labels_train);
	lars->put("lasso", false);
	lars->put("max_l1_norm", max_l1_norm); //!< max l1-norm of beta (estimator) for early stopping
	//![create_instance]

	// train model
	lars->train(normalized_features_train);

	// evaluate model on train data
	auto new_labels_train = lars->apply_regression(normalized_features_train);
	PlotResults(features, labels, new_labels_train->get_labels(), "Shogun LARS: Train", "shogun-lars-train");

	// evaluate model on test data
	auto new_labels = lars->apply_regression(normalized_features_test);

	auto eval_criterium = create<Evaluation>("MeanSquaredError");
	auto accuracy = eval_criterium->evaluate(new_labels, labels_test);
	std::cout << "LARS mse = " << accuracy << std::endl;

	PlotResults(test_features, test_labels, new_labels->get_labels(), "Shogun LARS: Test", "shogun-lars-test");
}



/*================================================ Support Vector Machine ====================================================================*/

/* Train and test the support vector regression */
void ML_REG::SVR_Plot(const SGMatrix<double> &features,
						const SGVector<double> &labels,
						const SGMatrix<double> &test_features,
						const SGVector<double> &test_labels,
						double kernel_width, double C1,
						double C2, double tube_epsilon,
						int svr_solver_type) {

	//auto features_train = create<Features>( transpose_matrix(features) );
	auto features_train = create<Features>( transpose_matrix(features) );
	//cout << "(num_vectors, num_features) = " << features_train->get_num_vectors() << " , " <<  features_train->get_num_features() << endl;
	auto features_test = create<Features>( transpose_matrix(test_features) );
	//auto features_test = DenseFeatures<double>::obtain_from_generic( create<Features>( transpose_matrix(test_features) ) );
	auto labels_train = create<Labels>(labels);
	auto labels_test =  create<Labels>(test_labels);

	//![create_appropriate_kernel]
	auto kernel = create<Kernel>("GaussianKernel");
	kernel->put("width", kernel_width); // must be strictly positive, say [0.1, ... , 10]
	//![create_appropriate_kernel]


	//![create_instance]
	auto svr = create<Machine>("LibSVR");
	svr->put("C1", C1); // must be strictly positive
	svr->put("C2", C2);
	svr->put("tube_epsilon", tube_epsilon); // must be small
	svr->put("kernel", kernel);
	// use 'LIBSVR_EPSILON_SVR == 1' or 'LIBSVR_NU_SVR == 2'
	svr->put("libsvr_solver_type", svr_solver_type);
	svr->set_solver_type(ST_AUTO);
	auto p = svr->get_global_parallel();
	p->set_num_threads(num_threads);
	svr->put("labels", labels_train);
	//![create_instance]

	svr->train(features_train);

	// evaluate model on train data
	auto new_labels_train = svr->apply_regression(features_train);
	PlotResults(features, labels, new_labels_train->get_labels(), "Shogun SVR: Train", "shogun-svr-train");

	// evaluate model on test data
	auto new_labels = svr->apply_regression(features_test);

	auto eval_criterium = create<Evaluation>("MeanSquaredError");
	auto accuracy = eval_criterium->evaluate(new_labels, labels_test);
	std::cout << "SVR mse = " << accuracy << std::endl;

	PlotResults(test_features, test_labels, new_labels->get_labels(), "Shogun SVR: Test", "shogun-svr-test");
}


/* Compute the mse on the validation dataset using the support vector machine regression */
double ML_REG::SVR(	const SGMatrix<double> &features_train, const SGVector<double> &labels_train,
					const SGMatrix<double> &features_valid, const SGVector<double> &labels_valid,
					double kernel_width, double C1, double C2, double tube_epsilon,
					int svr_solver_type) {

	auto SGfeatures_train = create<Features>( transpose_matrix(features_train) );
	auto SGfeatures_valid = create<Features>( transpose_matrix(features_valid) );
	auto SGlabels_train = create<Labels>(labels_train);
	auto SGlabels_valid = create<Labels>(labels_valid);


	//![create_appropriate_kernel]
	auto kernel = create<Kernel>("GaussianKernel");
	kernel->put("width", kernel_width); // must be strictly positive, say [0.1, ... , 10]
	//![create_appropriate_kernel]

	//![create_instance]
	auto svr = create<Machine>("LibSVR");
	svr->put("C1", C1); // must be strictly positive
	svr->put("C2", C2);
	svr->put("tube_epsilon", tube_epsilon); // must be small
	svr->put("kernel", kernel);
	// use 'LIBSVR_EPSILON_SVR == 1' or 'LIBSVR_NU_SVR == 2'
	svr->put("libsvr_solver_type", svr_solver_type);
	svr->set_solver_type(ST_AUTO);
	auto p = svr->get_global_parallel();
	p->set_num_threads(num_threads);
	svr->put("labels", SGlabels_train);
	//![create_instance]

	svr->train(SGfeatures_train);

	// evaluate model on validation data
	auto new_labels = svr->apply_regression(SGfeatures_valid);

	auto eval_criterium = create<Evaluation>("MeanSquaredError");
	auto mse = eval_criterium->evaluate(new_labels, SGlabels_valid);
	return mse;
}


void ML_REG::SVRegression(	const SGMatrix<double> &features, /*columns are features*/
							const SGVector<double> &labels,
							const SGMatrix<double> &test_features,
							const SGVector<double> &test_labels,
							int num_subsets, int min_subset_size,
							SGVector<double> kernel_widths_list,
							SGVector<double> C1_list,
							SGVector<double> C2_list,
							SGVector<double> tube_epsilons_list) {

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_(features.num_rows == labels.vlen);

	// draw subsamples to train and validate the model
	int nonzero_train_size = 0, nonzero_valid_size = 0, nonzero_counter = 0;
	SGMatrix<int> train_indices(n, num_subsets), valid_indices(n, num_subsets);
	std::tie(train_indices, valid_indices) = tscv(labels, num_subsets, min_subset_size);

	SGVector<int> train_indices_i(n), valid_indices_i(n), nonzero_train_indices, nonzero_valid_indices;

	//define SVR solver types
	SGVector<int> svr_solver_types(2);
	svr_solver_types[0] = 1;
	svr_solver_types[1] = 2;

	// define a matrix to save MSEs for all 'num_feats' and 'num_bags'
	SGMatrix<double> mse_mat(kernel_widths_list.vlen * C1_list.vlen * C2_list.vlen * tube_epsilons_list.vlen * 2, 6);
	mse_mat.zero();

	double mse_i = 0;

	/*// define pointers to training and validation data objects
	CDenseFeatures<float64_t>* SGfeatures_train_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_train_i = new CRegressionLabels();
	CDenseFeatures<float64_t>* SGfeatures_valid_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_valid_i = new CRegressionLabels();*/

	for (int i = 0; i < num_subsets; ++i) {
		train_indices_i = train_indices.get_column(i);
		valid_indices_i = valid_indices.get_column(i);

		// construct a training subsample and a validation subsample
		nonzero_train_indices = train_indices_i.find_if(ML_REG::isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(ML_REG::isnotZero);
		nonzero_valid_size = nonzero_valid_indices.size();

		SGMatrix<double> features_train_i(nonzero_train_size, dim), features_valid_i(nonzero_valid_size, dim);
		SGVector<double> labels_train_i(nonzero_train_size), labels_valid_i(nonzero_valid_size);

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_train_indices) {
			for (int k = 0; k < dim; ++k) {
				features_train_i(nonzero_counter, k) = features(j, k);
			}
			labels_train_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_valid_indices) {
			for (int k = 0; k < dim; ++k) {
				features_valid_i(nonzero_counter, k) = features(j, k);
			}
			labels_valid_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}


		// loop through 'kernel_widths_list','C1_list', 'C2_list', and 'tube_epsilons_list'
		// to obtain the hyperparameters with minimum RMSE
		int mse_counter = 0;
		for (double kernel_width : kernel_widths_list){
			for (double C1 : C1_list) {
				for (double C2 : C2_list) {
					for (double tube_epsilon : tube_epsilons_list) {
						for (int svr_solver_type : svr_solver_types) {

							mse_i = ML_REG::SVR(features_train_i, labels_train_i, features_valid_i, labels_valid_i,
																kernel_width, C1, C2, tube_epsilon, svr_solver_type);
							if (i == 0) {
								mse_mat(mse_counter, 0) = kernel_width;
								mse_mat(mse_counter, 1) = C1;
								mse_mat(mse_counter, 2) = C2;
								mse_mat(mse_counter, 3) = tube_epsilon;
								mse_mat(mse_counter, 4) = svr_solver_type;
							}
							mse_mat(mse_counter, 5) += mse_i / num_subsets;
							mse_counter += 1;
						}
					}
				}
			}
		}
	}

	mse_mat.display_matrix("mse_mat");

	// get the optimal hyperparameters
	SGVector<double> mse = mse_mat.get_column(5);
	int min_index = Math::arg_min(mse.vector, 1, mse.vlen);
	double opt_kernel_width = mse_mat(min_index, 0);
	double opt_C1 = mse_mat(min_index, 1);
	double opt_C2 = mse_mat(min_index, 2);
	double opt_tube_epsilon = mse_mat(min_index, 3);
	int opt_svr_solver_type = static_cast<int>( mse_mat(min_index, 4) );

	cout << "(optimal kernel width, optimal C1, optimal C2, optimal tube epsilon, optimal solver type index) = "
		 << opt_kernel_width << " , " << opt_C1 << " , " << opt_C2 << " , " << opt_tube_epsilon << " , " << opt_svr_solver_type << endl;

	// RE-TRAIN the model using the optimal hyperparameters
	ML_REG::SVR_Plot(features, labels, test_features, test_labels, opt_kernel_width, opt_C1, opt_C2, opt_tube_epsilon, opt_svr_solver_type);
}



/*================================================================== Gradient Boosting Machine ============================================================*/

/* Train and test the Gradient Boosting Machine */
void ML_REG::GBM_Plot(const SGMatrix<double> &features,
			  const SGVector<double> &labels,
			  const SGMatrix<double> &test_features,
			  const SGVector<double> &test_labels,
			  int tree_max_depth, int num_iters,
			  double learning_rate,
			  double subset_fraction,
			  int seed) {

	//auto features_train = create<Features>( transpose_matrix(features) );
	auto features_train = create<Features>( transpose_matrix(features) );
	//cout << "(num_vectors, num_features) = " << features_train->get_num_vectors() << " , " <<  features_train->get_num_features() << endl;
	auto features_test = create<Features>( transpose_matrix(test_features) );
	//auto features_test = DenseFeatures<double>::obtain_from_generic( create<Features>( transpose_matrix(test_features) ) );
	auto labels_train = create<Labels>(labels);
	auto labels_test =  create<Labels>(test_labels);

	int dim = features.num_cols;
	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);

	/*
	* A CART tree is a binary decision tree that is constructed by splitting a
	* node into two child nodes repeatedly, beginning with the root node that
	* contains the whole dataset.
	*/
	auto tree = create<Machine>("CARTree");
	tree->put("max_depth", tree_max_depth);
	tree->put("nominal", feature_type);
	tree->put("mode", EProblemType::PT_REGRESSION);

	auto loss = create<LossFunction>("SquaredLoss");

	//![create_instance]
	auto sgbm = create<Machine>("StochasticGBMachine");
	sgbm->put("machine", tree);
	sgbm->put("loss", loss);
	sgbm->put("num_iterations", num_iters);
	sgbm->put("learning_rate", learning_rate);
	sgbm->put("subset_frac", subset_fraction);
	sgbm->put("labels", labels_train);
	sgbm->put("seed", seed);
	sgbm->set_solver_type(ST_AUTO);
	auto p = sgbm->get_global_parallel();
	p->set_num_threads(num_threads);
	//![create_instance]

	sgbm->train(features_train);

	// evaluate model on train data
	auto new_labels_train = sgbm->apply_regression(features_train);
	PlotResults(features, labels, new_labels_train->get_labels(), "Shogun Gradient Boosting: Train", "shogun-gbm-train");

	// evaluate model on test data
	auto new_labels = sgbm->apply_regression(features_test);

	auto eval_criterium = create<Evaluation>("MeanSquaredError");
	auto accuracy = eval_criterium->evaluate(new_labels, labels_test);
	std::cout << "GBM mse = " << accuracy << std::endl;

	PlotResults(test_features, test_labels, new_labels->get_labels(), "Shogun Gradient Boosting: Test", "shogun-gbm-test");
}

SGVector<double> ML_REG::GBM_Plot(const SGMatrix<double> &features,
								  const SGVector<double> &labels,
								  int tree_max_depth,
								  int num_iters,
								  double learning_rate,
								  double subset_fraction,
								  int num_rand_feats, /*number of random features used for bagging (for RF)*/
								  int num_bags, /*number of bags (for RF)*/
								  int seed) {
	(void) num_rand_feats, num_bags;

	//auto features_train = create<Features>( transpose_matrix(features) );
	auto features_train = create<Features>( transpose_matrix(features) );
	//cout << "(num_vectors, num_features) = " << features_train->get_num_vectors() << " , " <<  features_train->get_num_features() << endl;
	auto labels_train = create<Labels>(labels);

	//![create_preprocessor]
	auto preproc = create<Transformer>("PruneVarSubMean");
	preproc->fit(features_train);
	//![create_preprocessor]

	//![transform_features]
	features_train = preproc->transform(features_train);
	//![transform_features]

	int dim = features.num_cols;
	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);

	/*
	* A CART tree is a binary decision tree that is constructed by splitting a
	* node into two child nodes repeatedly, beginning with the root node that
	* contains the whole dataset.
	*/
	auto tree = create<Machine>("CARTree");
	tree->put("max_depth", tree_max_depth);
	tree->put("nominal", feature_type);
	tree->put("mode", EProblemType::PT_REGRESSION);

	auto loss = create<LossFunction>("SquaredLoss");

	//![create_instance]
	auto sgbm = create<Machine>("StochasticGBMachine");
	sgbm->put("machine", tree);
	sgbm->put("loss", loss);
	sgbm->put("num_iterations", num_iters);
	sgbm->put("learning_rate", learning_rate);
	sgbm->put("subset_frac", subset_fraction);
	sgbm->put("labels", labels_train);
	sgbm->put("seed", seed);
	sgbm->set_solver_type(ST_AUTO);
	auto p = sgbm->get_global_parallel();
	p->set_num_threads(num_threads);
	//![create_instance]

	sgbm->train(features_train);

	// evaluate model on train data
	auto new_labels_train = sgbm->apply_regression(features_train);
	return new_labels_train->get_labels();
}


/* Compute the mse on the validation dataset using the Gradient Boosting Machine */
double ML_REG::GBM(	const SGMatrix<double> &features_train, const SGVector<double> &labels_train,
					const SGMatrix<double> &features_valid, const SGVector<double> &labels_valid,
					int tree_max_depth, int num_iters, double learning_rate, double subset_fraction,
					int seed) {
	int dim = features_train.num_cols;

	auto SGfeatures_train = create<Features>( transpose_matrix(features_train) );
	auto SGfeatures_valid = create<Features>( transpose_matrix(features_valid) );
	auto SGlabels_train = create<Labels>(labels_train);
	auto SGlabels_valid = create<Labels>(labels_valid);

	//![create_preprocessor]
	auto preproc = create<Transformer>("PruneVarSubMean");
	preproc->fit(SGfeatures_train);
	//![create_preprocessor]

	//![transform_features]
	SGfeatures_train = preproc->transform(SGfeatures_train);
	SGfeatures_valid = preproc->transform(SGfeatures_valid);
	//![transform_features]

	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);

	/*
	* A CART tree is a binary decision tree that is constructed by splitting a
	* node into two child nodes repeatedly, beginning with the root node that
	* contains the whole dataset.
	*/
	auto tree = create<Machine>("CARTree");
	tree->put("max_depth", tree_max_depth);
	tree->put("nominal", feature_type);
	tree->put("mode", EProblemType::PT_REGRESSION);

	auto loss = create<LossFunction>("SquaredLoss");

	//![create_instance]
	auto sgbm = create<Machine>("StochasticGBMachine");
	sgbm->put("machine", tree);
	sgbm->put("loss", loss);
	sgbm->put("num_iterations", num_iters);
	sgbm->put("learning_rate", learning_rate);
	sgbm->put("subset_frac", subset_fraction);
	sgbm->put("labels", SGlabels_train);
	sgbm->put("seed", seed);
	sgbm->set_solver_type(ST_AUTO);
	auto p = sgbm->get_global_parallel();
	p->set_num_threads(num_threads);
	//![create_instance]

	sgbm->train(SGfeatures_train);

	// evaluate model on validation data
	auto new_labels = sgbm->apply_regression(SGfeatures_valid);

	auto eval_criterium = create<Evaluation>("MeanSquaredError");
	auto mse = eval_criterium->evaluate(new_labels, SGlabels_valid);
	return mse;
}


/* Perform GBM with cross validation */
void ML_REG::GBMRegression(	const SGMatrix<double> &features, /*columns are features*/
							const SGVector<double> &labels,
							const SGMatrix<double> &test_features,
							const SGVector<double> &test_labels,
							int num_subsets, int min_subset_size,
							SGVector<int> tree_max_depths_list,
							SGVector<int> num_iters_list,
							SGVector<double> learning_rates_list,
							SGVector<double> subset_fractions_list,
							int seed) {

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_( (features.num_rows == labels.vlen) && (Math::max(learning_rates_list.vector, learning_rates_list.vlen) <= 1.) \
											   && (Math::max(subset_fractions_list.vector, subset_fractions_list.vlen) <= 1.) );

	// draw subsamples to train and validate the model
	int nonzero_train_size = 0, nonzero_valid_size = 0, nonzero_counter = 0;
	SGMatrix<int> train_indices(n, num_subsets), valid_indices(n, num_subsets);
	std::tie(train_indices, valid_indices) = tscv(labels, num_subsets, min_subset_size);

	SGVector<int> train_indices_i(n), valid_indices_i(n), nonzero_train_indices, nonzero_valid_indices;

	// define a matrix to save MSEs for all 'num_feats' and 'num_bags'
	SGMatrix<double> mse_mat(tree_max_depths_list.vlen * num_iters_list.vlen * learning_rates_list.vlen * subset_fractions_list.vlen, 5);
	mse_mat.zero();

	double mse_i = 0;

	/*// define pointers to training and validation data objects
	CDenseFeatures<float64_t>* SGfeatures_train_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_train_i = new CRegressionLabels();
	CDenseFeatures<float64_t>* SGfeatures_valid_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_valid_i = new CRegressionLabels();*/

	for (int i = 0; i < num_subsets; ++i) {
		train_indices_i = train_indices.get_column(i);
		valid_indices_i = valid_indices.get_column(i);

		// construct a training subsample and a validation subsample
		nonzero_train_indices = train_indices_i.find_if(ML_REG::isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(ML_REG::isnotZero);
		nonzero_valid_size = nonzero_valid_indices.size();

		SGMatrix<double> features_train_i(nonzero_train_size, dim), features_valid_i(nonzero_valid_size, dim);
		SGVector<double> labels_train_i(nonzero_train_size), labels_valid_i(nonzero_valid_size);

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_train_indices) {
			for (int k = 0; k < dim; ++k) {
				features_train_i(nonzero_counter, k) = features(j, k);
			}
			labels_train_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_valid_indices) {
			for (int k = 0; k < dim; ++k) {
				features_valid_i(nonzero_counter, k) = features(j, k);
			}
			labels_valid_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}


		// loop through 'tree_max_depths_list','num_iters_list', 'learning_rates_list', and 'subset_fractions_list'
		// to obtain the hyperparameters with minimum RMSE
		int mse_counter = 0;
		for (int tree_max_depth : tree_max_depths_list){
			for (int num_iters : num_iters_list) {
				for (double learning_rate : learning_rates_list) {
					for (double subset_fraction : subset_fractions_list) {

						mse_i = GBM(features_train_i, labels_train_i, features_valid_i, labels_valid_i,
										tree_max_depth, num_iters, learning_rate, subset_fraction, seed);
						if (i == 0) {
							mse_mat(mse_counter, 0) = tree_max_depth;
							mse_mat(mse_counter, 1) = num_iters;
							mse_mat(mse_counter, 2) = learning_rate;
							mse_mat(mse_counter, 3) = subset_fraction;
						}
						mse_mat(mse_counter, 4) += mse_i / num_subsets;
						mse_counter += 1;
					}
				}
			}
		}
	}

	mse_mat.display_matrix("mse_mat");

	// get the optimal hyperparameters
	SGVector<double> mse = mse_mat.get_column(4);
	int min_index = Math::arg_min(mse.vector, 1, mse.vlen);
	int opt_tree_max_depth = static_cast<int>( mse_mat(min_index, 0) );
	int opt_num_iters = static_cast<int>( mse_mat(min_index, 1) );
	double opt_learning_rate = mse_mat(min_index, 2);
	double opt_subset_fraction = mse_mat(min_index, 3);

	cout << "(optimal tree max depth, optimal number of iterations, optimal learning rate, optimal subset fraction) = "
		 << opt_tree_max_depth << " , " << opt_num_iters << " , " << opt_learning_rate << " , " << opt_subset_fraction << endl;

	// RE-TRAIN the model using the optimal hyperparameters
	GBM_Plot(features, labels, test_features, test_labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate,
			 opt_subset_fraction, seed);
}

SGVector<double> ML_REG::GBMRegression(	const SGMatrix<double> &features, /*columns are features*/
										const SGVector<double> &labels,
										int num_subsets, int min_subset_size,
										SGVector<int> tree_max_depths_list,
										SGVector<int> num_iters_list,
										SGVector<double> learning_rates_list,
										SGVector<double> subset_fractions_list,
										SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
										SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
										int seed) {

	(void) num_rand_feats_list, num_bags_list;

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_( (features.num_rows == labels.vlen) && (Math::max(learning_rates_list.vector, learning_rates_list.vlen) <= 1.) \
											   && (Math::max(subset_fractions_list.vector, subset_fractions_list.vlen) <= 1.) );

	// draw subsamples to train and validate the model
	int nonzero_train_size = 0, nonzero_valid_size = 0, nonzero_counter = 0;
	SGMatrix<int> train_indices(n, num_subsets), valid_indices(n, num_subsets);
	std::tie(train_indices, valid_indices) = tscv(labels, num_subsets, min_subset_size);

	SGVector<int> train_indices_i(n), valid_indices_i(n), nonzero_train_indices, nonzero_valid_indices;

	// define a matrix to save MSEs for all 'num_feats' and 'num_bags'
	SGMatrix<double> mse_mat(tree_max_depths_list.vlen * num_iters_list.vlen * learning_rates_list.vlen * subset_fractions_list.vlen, 5);
	mse_mat.zero();

	double mse_i = 0;

	/*// define pointers to training and validation data objects
	CDenseFeatures<float64_t>* SGfeatures_train_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_train_i = new CRegressionLabels();
	CDenseFeatures<float64_t>* SGfeatures_valid_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_valid_i = new CRegressionLabels();*/

	for (int i = 0; i < num_subsets; ++i) {
		train_indices_i = train_indices.get_column(i);
		valid_indices_i = valid_indices.get_column(i);

		// construct a training subsample and a validation subsample
		nonzero_train_indices = train_indices_i.find_if(ML_REG::isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(ML_REG::isnotZero);
		nonzero_valid_size = nonzero_valid_indices.size();

		SGMatrix<double> features_train_i(nonzero_train_size, dim), features_valid_i(nonzero_valid_size, dim);
		SGVector<double> labels_train_i(nonzero_train_size), labels_valid_i(nonzero_valid_size);

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_train_indices) {
			for (int k = 0; k < dim; ++k) {
				features_train_i(nonzero_counter, k) = features(j, k);
			}
			labels_train_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_valid_indices) {
			for (int k = 0; k < dim; ++k) {
				features_valid_i(nonzero_counter, k) = features(j, k);
			}
			labels_valid_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}


		// loop through 'tree_max_depths_list','num_iters_list', 'learning_rates_list', and 'subset_fractions_list'
		// to obtain the hyperparameters with minimum RMSE
		int mse_counter = 0;
		for (int tree_max_depth : tree_max_depths_list){
			for (int num_iters : num_iters_list) {
				for (double learning_rate : learning_rates_list) {
					for (double subset_fraction : subset_fractions_list) {

						mse_i = GBM(features_train_i, labels_train_i, features_valid_i, labels_valid_i,
										tree_max_depth, num_iters, learning_rate, subset_fraction, seed);
						if (i == 0) {
							mse_mat(mse_counter, 0) = tree_max_depth;
							mse_mat(mse_counter, 1) = num_iters;
							mse_mat(mse_counter, 2) = learning_rate;
							mse_mat(mse_counter, 3) = subset_fraction;
						}
						mse_mat(mse_counter, 4) += mse_i / num_subsets;
						mse_counter += 1;
					}
				}
			}
		}
	}

	//mse_mat.display_matrix("mse_mat");

	// get the optimal hyperparameters
	SGVector<double> mse = mse_mat.get_column(4);
	int min_index = Math::arg_min(mse.vector, 1, mse.vlen);
	int opt_tree_max_depth = static_cast<int>( mse_mat(min_index, 0) );
	int opt_num_iters = static_cast<int>( mse_mat(min_index, 1) );
	double opt_learning_rate = mse_mat(min_index, 2);
	double opt_subset_fraction = mse_mat(min_index, 3);

	cout << "(optimal tree max depth, optimal number of iterations, optimal learning rate, optimal subset fraction) = "
		 << opt_tree_max_depth << " , " << opt_num_iters << " , " << opt_learning_rate << " , " << opt_subset_fraction << endl;

	// RE-TRAIN the model using the optimal hyperparameters
	SGVector<double> new_labels = GBM_Plot(features, labels, opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, 0, 0, seed);
	return new_labels;
}

/* Perform GBM CV
OUTPUT: optimal tree maximum depth, number of iterations, learning rate, subset fraction, number of random features, number of bags */
std::tuple<int, int, double, double, int, int> ML_REG::GBM_cv(	const SGMatrix<double> &features, /*columns are features*/
																const SGVector<double> &labels,
																int num_subsets, int min_subset_size,
																SGVector<int> tree_max_depths_list,
																SGVector<int> num_iters_list,
																SGVector<double> learning_rates_list,
																SGVector<double> subset_fractions_list,
																SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
																SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
																int seed) {

	(void) num_rand_feats_list, num_bags_list;

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_( (features.num_rows == labels.vlen) && (Math::max(learning_rates_list.vector, learning_rates_list.vlen) <= 1.) \
											   && (Math::max(subset_fractions_list.vector, subset_fractions_list.vlen) <= 1.) );

	// draw subsamples to train and validate the model
	int nonzero_train_size = 0, nonzero_valid_size = 0, nonzero_counter = 0;
	SGMatrix<int> train_indices(n, num_subsets), valid_indices(n, num_subsets);
	std::tie(train_indices, valid_indices) = tscv(labels, num_subsets, min_subset_size);

	SGVector<int> train_indices_i(n), valid_indices_i(n), nonzero_train_indices, nonzero_valid_indices;

	// define a matrix to save MSEs for all 'num_feats' and 'num_bags'
	SGMatrix<double> mse_mat(tree_max_depths_list.vlen * num_iters_list.vlen * learning_rates_list.vlen * subset_fractions_list.vlen, 5);
	mse_mat.zero();

	double mse_i = 0;

	/*// define pointers to training and validation data objects
	CDenseFeatures<float64_t>* SGfeatures_train_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_train_i = new CRegressionLabels();
	CDenseFeatures<float64_t>* SGfeatures_valid_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_valid_i = new CRegressionLabels();*/

	for (int i = 0; i < num_subsets; ++i) {
		train_indices_i = train_indices.get_column(i);
		valid_indices_i = valid_indices.get_column(i);

		// construct a training subsample and a validation subsample
		nonzero_train_indices = train_indices_i.find_if(ML_REG::isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(ML_REG::isnotZero);
		nonzero_valid_size = nonzero_valid_indices.size();

		SGMatrix<double> features_train_i(nonzero_train_size, dim), features_valid_i(nonzero_valid_size, dim);
		SGVector<double> labels_train_i(nonzero_train_size), labels_valid_i(nonzero_valid_size);

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_train_indices) {
			for (int k = 0; k < dim; ++k) {
				features_train_i(nonzero_counter, k) = features(j, k);
			}
			labels_train_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_valid_indices) {
			for (int k = 0; k < dim; ++k) {
				features_valid_i(nonzero_counter, k) = features(j, k);
			}
			labels_valid_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}


		// loop through 'tree_max_depths_list','num_iters_list', 'learning_rates_list', and 'subset_fractions_list'
		// to obtain the hyperparameters with minimum RMSE
		int mse_counter = 0;
		for (int tree_max_depth : tree_max_depths_list){
			for (int num_iters : num_iters_list) {
				for (double learning_rate : learning_rates_list) {
					for (double subset_fraction : subset_fractions_list) {

						mse_i = GBM(features_train_i, labels_train_i, features_valid_i, labels_valid_i,
										tree_max_depth, num_iters, learning_rate, subset_fraction, seed);
						if (i == 0) {
							mse_mat(mse_counter, 0) = tree_max_depth;
							mse_mat(mse_counter, 1) = num_iters;
							mse_mat(mse_counter, 2) = learning_rate;
							mse_mat(mse_counter, 3) = subset_fraction;
						}
						mse_mat(mse_counter, 4) += mse_i / num_subsets;
						mse_counter += 1;
					}
				}
			}
		}
	}

	//mse_mat.display_matrix("mse_mat");

	// get the optimal hyperparameters
	SGVector<double> mse = mse_mat.get_column(4);
	int min_index = Math::arg_min(mse.vector, 1, mse.vlen);
	int opt_tree_max_depth = static_cast<int>( mse_mat(min_index, 0) );
	int opt_num_iters = static_cast<int>( mse_mat(min_index, 1) );
	double opt_learning_rate = mse_mat(min_index, 2);
	double opt_subset_fraction = mse_mat(min_index, 3);

	cout << "(optimal tree max depth, optimal number of iterations, optimal learning rate, optimal subset fraction) = "
		 << opt_tree_max_depth << " , " << opt_num_iters << " , " << opt_learning_rate << " , " << opt_subset_fraction << endl;

	return {opt_tree_max_depth, opt_num_iters, opt_learning_rate, opt_subset_fraction, 0 , 0};
}


#if 0
void GBMRegression(const SGMatrix<double> &features, /*columns are features*/
				   const SGVector<double> &labels,
				   const SGMatrix<double> &test_features,
				   const SGVector<double> &test_labels,
				   int num_subsets, int min_subset_size,
				   SGVector<int> tree_max_depths_list,
				   SGVector<int> num_iters_list,
				   SGVector<double> learning_rates_list,
				   SGVector<double> subset_fractions_list,
				   int seed) {

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_( (features.num_rows == labels.vlen) && (CMath::max(learning_rates_list.vector, learning_rates_list.vlen) <= 1.) \
											   && (CMath::max(subset_fractions_list.vector, subset_fractions_list.vlen) <= 1.) );

	// draw subsamples to train and validate the model
	int nonzero_train_size = 0, nonzero_valid_size = 0, nonzero_counter = 0;
	SGMatrix<int> train_indices(n, num_subsets), valid_indices(n, num_subsets);
	std::tie(train_indices, valid_indices) = tscv(labels, num_subsets, min_subset_size);

	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);

	/*
	   * A CART tree is a binary decision tree that is constructed by splitting a
	   * node into two child nodes repeatedly, beginning with the root node that
	   * contains the whole dataset.
	*/
	auto tree = new CCARTree(feature_type, PT_REGRESSION);
	auto loss = some<CSquaredLoss>();
	auto eval_criterium = some<CMeanSquaredError>();

	// define the gradient boosting machine
	//CStochasticGBMachine *sgbm = new CStochasticGBMachine();
	auto sgbm = new CStochasticGBMachine();
	sgbm->set_loss_function(loss);
	sgbm->set_machine(tree);
	sgbm->put("seed", seed);
	sgbm->set_solver_type(ST_AUTO);
	auto p = wrap( sgbm->get_global_parallel() );
	p->set_num_threads(num_threads);


	SGVector<int> train_indices_i(n), valid_indices_i(n), nonzero_train_indices, nonzero_valid_indices;

	// define a matrix to save MSEs for all 'num_feats' and 'num_bags'
	SGMatrix<double> mse_mat(tree_max_depths_list.vlen * num_iters_list.vlen * learning_rates_list.vlen * subset_fractions_list.vlen, 5);
	mse_mat.zero();

	/*// define pointers to training and validation data objects
	CDenseFeatures<float64_t>* SGfeatures_train_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_train_i = new CRegressionLabels();
	CDenseFeatures<float64_t>* SGfeatures_valid_i = new CDenseFeatures<float64_t>();
	CRegressionLabels* SGlabels_valid_i = new CRegressionLabels();*/

	for (int i = 0; i < num_subsets; ++i) {
		train_indices_i = train_indices.get_column(i);
		valid_indices_i = valid_indices.get_column(i);

		// construct a training subsample and a validation subsample
		nonzero_train_indices = train_indices_i.find_if(ML_REG::isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(ML_REG::isnotZero);
		nonzero_valid_size = nonzero_valid_indices.size();

		SGMatrix<double> features_train_i(nonzero_train_size, dim), features_valid_i(nonzero_valid_size, dim);
		SGVector<double> labels_train_i(nonzero_train_size), labels_valid_i(nonzero_valid_size);

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_train_indices) {
			for (int k = 0; k < dim; ++k) {
				features_train_i(nonzero_counter, k) = features(j, k);
			}
			labels_train_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_valid_indices) {
			for (int k = 0; k < dim; ++k) {
				features_valid_i(nonzero_counter, k) = features(j, k);
			}
			labels_valid_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		// construct training and validation data objects
		auto SGfeatures_train_i = some<CDenseFeatures<double>>( transpose_matrix(features_train_i) );
		auto SGlabels_train_i = some<CRegressionLabels>(labels_train_i);
		auto SGfeatures_valid_i = some<CDenseFeatures<double>>( transpose_matrix(features_valid_i) );
		auto SGlabels_valid_i = some<CRegressionLabels>(labels_valid_i);

		// loop through 'tree_max_depths_list','num_iters_list', 'learning_rates_list', and 'subset_fractions_list'
		// to obtain the hyperparameters with minimum RMSE
		int mse_counter = 0;
		for (int tree_max_depth : tree_max_depths_list){
			auto tree = some<CCARTree>(feature_type, PT_REGRESSION);
			//SG_REF(tree);
			// try to change tree depth to see its influence on accuracy
			tree->ref();
			tree->set_max_depth(tree_max_depth);
			for (int num_iters : num_iters_list) {
				for (double learning_rate : learning_rates_list) {
					for (double subset_fraction : subset_fractions_list) {
						//auto sgbm = some<CStochasticGBMachine>(tree, loss, num_iters, learning_rate, subset_fraction);
						sgbm->ref();
						//SG_REF(sgbm)
						sgbm->set_num_iterations(num_iters);
						sgbm->set_learning_rate(learning_rate);
						sgbm->set_subset_fraction(subset_fraction);
						sgbm->set_labels(SGlabels_train_i);

						// train the model
						sgbm->train(SGfeatures_train_i);

						// evaluate the model on the validation data
						auto new_SGlabels_valid_i = wrap( sgbm->apply_regression(SGfeatures_valid_i) );
						if (i == 0) {
							mse_mat(mse_counter, 0) = tree_max_depth;
							mse_mat(mse_counter, 1) = num_iters;
							mse_mat(mse_counter, 2) = learning_rate;
							mse_mat(mse_counter, 3) = subset_fraction;
						}
						mse_mat(mse_counter, 4) += eval_criterium->evaluate(new_SGlabels_valid_i, SGlabels_valid_i) / num_subsets;
						sgbm->unref();
						mse_counter += 1;
					}
				}
			}
			tree->unref();
		}
	}
	mse_mat.display_matrix("mse_mat");

	// get the optimal hyperparameters
	SGVector<double> mse = mse_mat.get_column(4);
	int min_index = CMath::arg_min(mse.vector, 1, mse.vlen);
	int opt_tree_max_depth = static_cast<int>( mse_mat(min_index, 0) );
	int opt_num_iters = static_cast<int>( mse_mat(min_index, 1) );
	double opt_learning_rate = mse_mat(min_index, 2);
	double opt_subset_fraction = mse_mat(min_index, 3);

	cout << "(optimal tree max depth, optimal number of iterations, optimal learning rate, optimal subset fraction) = "
		 << opt_tree_max_depth << " , " << opt_num_iters << " , " << opt_learning_rate << " , " << opt_subset_fraction << endl;

	// RE-TRAIN the model using the optimal hyperparameters
	auto SGfeatures_train = some<CDenseFeatures<double>>( transpose_matrix(features) );
	auto SGlabels_train = some<CRegressionLabels>(labels);
	auto SGfeatures_test = some<CDenseFeatures<double>>( transpose_matrix(test_features) );
	auto SGlabels_test = some<CRegressionLabels>(test_labels);

	//tree = some<CCARTree>(feature_type, PT_REGRESSION);
	//SG_REF(tree);
	tree->set_max_depth(opt_tree_max_depth);
	//auto sgbm = some<CStochasticGBMachine>(tree, loss, opt_num_iters, opt_learning_rate, opt_subset_fraction);
	//SG_REF(sgbm);
	sgbm->set_num_iterations(opt_num_iters);
	sgbm->set_learning_rate(opt_learning_rate);
	sgbm->set_subset_fraction(opt_subset_fraction);
	sgbm->set_labels(SGlabels_train);

	// train the model
	sgbm->train(SGfeatures_train);
	// evaluate the trained model
	auto new_SGlabels_train = wrap( sgbm->apply_regression(SGfeatures_train) );
	auto accuracy = eval_criterium->evaluate(new_SGlabels_train, SGlabels_train);
	cout << "The MSE = " << accuracy << endl;

	PlotResults(SGfeatures_train, SGlabels_train, new_SGlabels_train, "Shogun GBM: Train", "shogun-gbm-train");

	// evaluate the model on the test data
	auto new_SGlabels_test = wrap( sgbm->apply_regression(SGfeatures_test) );
	accuracy = eval_criterium->evaluate(new_SGlabels_test, SGlabels_test);
	cout << "The Forecast MSE = " << accuracy << endl;
	SG_FREE(tree);
	SG_FREE(sgbm);

	PlotResults(SGfeatures_test, SGlabels_test, new_SGlabels_test, "Shogun GBM: Test", "shogun-gbm-test");
}



void GBMRegressionCV(const SGMatrix<double> &features, /*columns are features*/
				   const SGVector<double> &labels0,
				   const SGMatrix<double> &test_features,
				   const SGVector<double> &test_labels,
				   int num_subsets, int min_subset_size,
				   int num_iters_min, int num_iters_max,
				   int tree_max_depth,
				   double learning_rate_min, double learning_rate_max,
				   double subset_fraction_min, double subset_fraction_max,
				   int seed) {

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_(features.num_rows == labels0.vlen);

	// define Shogun data objects
	auto SGfeatures = some<CDenseFeatures<double>>( transpose_matrix(features) );
	auto SGlabels = some<CRegressionLabels>(labels0);


	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);

	/*
	   * A CART tree is a binary decision tree that is constructed by splitting a
	   * node into two child nodes repeatedly, beginning with the root node that
	   * contains the whole dataset.
	*/
	CCARTree *tree = new CCARTree(feature_type, PT_REGRESSION);
	tree->set_max_depth(tree_max_depth);
	auto loss = some<CSquaredLoss>();
	auto eval_criterium = some<CMeanSquaredError>();

	// define the gradient boosting machine
	int num_iterations = 100;
	double learning_rate = 0.01;
	double subset_fraction = 0.6;
	//CStochasticGBMachine *sgbm = new CStochasticGBMachine();
	auto sgbm = some<CStochasticGBMachine>(nullptr, nullptr , num_iterations, learning_rate, subset_fraction);
	sgbm->set_loss_function(loss);
	sgbm->set_machine(tree);
	sgbm->put("seed", seed);
	sgbm->set_solver_type(ST_AUTO);
	auto p = wrap( sgbm->get_global_parallel() );
	p->set_num_threads(num_threads);

	 // configure dataset splitting
	auto splitting_method = wrap(splitting_strategy("TimeSeriesSplitting"));
	auto reg_labels = wrap( labels(labels0) );
	splitting_method->put("labels", reg_labels);
	splitting_method->put("num_subsets", num_subsets);
	splitting_method->put("min_subset_size", min_subset_size);
	auto cross_validation = some<CCrossValidation>(sgbm, SGfeatures, SGlabels, splitting_method, eval_criterium);
	cross_validation->set_num_runs(2);
	cross_validation->print_modsel_params();


	// define parameters grid
	auto params_root = some<CModelSelectionParameters>();

//	auto param_tree = some<CModelSelectionParameters>("machine", tree);
//	params_root->append_child(param_tree);
//
//	auto param_loss = some<CModelSelectionParameters>("loss", loss);
//	params_root->append_child(param_loss);

	auto param_num_iters = some<CModelSelectionParameters>("num_iterations");
	params_root->append_child(param_num_iters);
	param_num_iters->build_values(num_iters_min, num_iters_max, ERangeType::R_LINEAR, 100);

	auto param_learning_rate = some<CModelSelectionParameters>("learning_rate");
	params_root->append_child(param_learning_rate);
	param_learning_rate->build_values(learning_rate_min, learning_rate_max, ERangeType::R_LINEAR, 0.05);


	auto param_subset_fraction = some<CModelSelectionParameters>("subset_fraction");
	params_root->append_child(param_subset_fraction);
	param_subset_fraction->build_values(subset_fraction_min, subset_fraction_max, ERangeType::R_LINEAR, 0.1);
	params_root->print_tree();



	// model selection
	auto model_selection = some<CGridSearchModelSelection>(cross_validation, params_root);
	auto best_parameters = model_selection->select_model(/*print_state*/ true);
	best_parameters->apply_to_machine(sgbm);
	best_parameters->print_tree();
	cout << "finishing CV..." << endl;

	 // train with best parameters
	if ( !sgbm->train(SGfeatures) ) {
		std::cerr << "training failed\n";
	}

	// evaluate the trained model
	auto new_SGlabels = wrap( sgbm->apply_regression(SGfeatures) );
	auto accuracy = eval_criterium->evaluate(new_SGlabels, SGlabels);
	cout << "The MSE = " << accuracy << endl;

	PlotResults(SGfeatures, SGlabels, new_SGlabels, "Shogun GBM: Train", "shogun-gbm-train");

	// evaluate the model on the test data
	auto SGfeatures_test = some<CDenseFeatures<double>>( transpose_matrix(test_features) );
	auto SGlabels_test = some<CRegressionLabels>(test_labels);
	auto new_SGlabels_test = wrap( sgbm->apply_regression(SGfeatures_test) );
	accuracy = eval_criterium->evaluate(new_SGlabels_test, SGlabels_test);
	cout << "The Forecast MSE = " << accuracy << endl;

	PlotResults(SGfeatures_test, SGlabels_test, new_SGlabels_test, "Shogun GBM: Test", "shogun-gbm-test");
}

#endif

/*================================================================ Random Forest ==================================================================*/

void ML_REG::RF_Plot(	const SGMatrix<double> &features,
						const SGVector<double> &labels,
						const SGMatrix<double> &test_features,
						const SGVector<double> &test_labels,
						int num_rand_feats, int num_bags,
						int seed) {

	int dim = features.num_cols;
	ASSERT_(features.num_rows == labels.vlen && num_rand_feats <= dim);

	//auto features_train = create<Features>( transpose_matrix(features) );
	auto features_train = create<Features>( transpose_matrix(features) );
	//cout << "(num_vectors, num_features) = " << features_train->get_num_vectors() << " , " <<  features_train->get_num_features() << endl;
	auto features_test = create<Features>( transpose_matrix(test_features) );
	//auto features_test = DenseFeatures<double>::obtain_from_generic( create<Features>( transpose_matrix(test_features) ) );
	auto labels_train = create<Labels>(labels);
	auto labels_test =  create<Labels>(test_labels);

	//![create_combination_rule]
	auto comb_rule = create<CombinationRule>("MeanRule");
	//![create_combination_rule]

	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);


	//![create_instance]
	//auto rand_forest = create<Machine>("RandomForest");
	std::unique_ptr<RandomForest> rand_forest( new RandomForest(features_train, labels_train, num_bags, num_rand_feats) );
	rand_forest->set_feature_types(feature_type);
	rand_forest->set_machine_problem_type(EProblemType::PT_REGRESSION);
	//rand_forest->set_num_random_features(num_rand_feats);
	//rand_forest->put("num_bags", num_bags);
	rand_forest->set_combination_rule(comb_rule);
	rand_forest->put("seed", seed);
	rand_forest->set_solver_type(ST_AUTO);

	auto p = rand_forest->get_global_parallel(); // parallelize RF
	p->set_num_threads(num_threads);
	//set label data
	//rand_forest->put("labels", labels_train);
	//![create_instance]

	//![train the model]
	rand_forest->train(features_train);
	//![train the model]

	// evaluate model on train data
	auto new_labels_train = rand_forest->apply_regression(features_train);
	PlotResults(features, labels, new_labels_train->get_labels(), "Random Forest: Train", "shogun-rf-train");

	// evaluate model on test data
	auto new_labels = rand_forest->apply_regression(features_test);

	auto eval_criterium = create<Evaluation>("MeanSquaredError");
	//rand_forest->put("oob_evaluation_metric", eval_criterium);
	//auto oob = rand_forest->get<float64_t>("oob_error");
	auto mse_train = eval_criterium->evaluate(new_labels_train, labels_train);
	auto mse_test = eval_criterium->evaluate(new_labels, labels_test);

	std::cout << "RF mse on the train data = " << mse_train << std::endl;
	std::cout << "RF mse on the test data = " << mse_test << std::endl;

	PlotResults(test_features, test_labels, new_labels->get_labels(), "Random Forest: Test", "shogun-rf-test");
}

/* Do RF regression */
SGVector<double> ML_REG::RF_Plot(const SGMatrix<double> &features,
								  const SGVector<double> &labels,
								  int tree_max_depth,
								  int num_iters,
								  double learning_rate,
								  double subset_fraction,
								  int num_rand_feats, /*number of random features used for bagging (for RF)*/
								  int num_bags, /*number of bags (for RF)*/
								  int seed) {

	(void) tree_max_depth, num_iters, learning_rate, subset_fraction;

	int dim = features.num_cols;
	ASSERT_(features.num_rows == labels.vlen);
	if (num_rand_feats > dim)
		num_rand_feats = dim;

	//auto features_train = create<Features>( transpose_matrix(features) );
	auto features_train = create<Features>( transpose_matrix(features) );
	//cout << "(num_vectors, num_features) = " << features_train->get_num_vectors() << " , " <<  features_train->get_num_features() << endl;
	auto labels_train = create<Labels>(labels);

	//![create_preprocessor]
	auto preproc = create<Transformer>("RescaleFeatures"); // to standardize data, just use 'PruneVarSubMean'
	preproc->fit(features_train);
	//![create_preprocessor]

	//![transform_features]
	features_train = preproc->transform(features_train);
	//![transform_features]

	//![create_combination_rule]
	auto comb_rule = create<CombinationRule>("MeanRule");
	//![create_combination_rule]

	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);


	//![create_instance]
	//auto rand_forest = create<Machine>("RandomForest");
	std::unique_ptr<RandomForest> rand_forest( new RandomForest(features_train, labels_train, num_bags, num_rand_feats) );
	rand_forest->set_feature_types(feature_type);
	rand_forest->set_machine_problem_type(EProblemType::PT_REGRESSION);
	//rand_forest->set_num_random_features(num_rand_feats);
	//rand_forest->put("num_bags", num_bags);
	rand_forest->set_combination_rule(comb_rule);
	rand_forest->put("seed", seed);
	rand_forest->set_solver_type(ST_AUTO);

	auto p = rand_forest->get_global_parallel(); // parallelize RF
	p->set_num_threads(num_threads);
	//set label data
	//rand_forest->put("labels", labels_train);
	//![create_instance]

	//![train the model]
	rand_forest->train(features_train);
	//![train the model]

	// evaluate model on train data
	auto new_labels_train = rand_forest->apply_regression(features_train);

	return new_labels_train->get_labels();
}

/* Compute MSE on test data with Random Forest */
double ML_REG::RF_MSE(	const SGMatrix<double> &features_train,
						const SGVector<double> &labels_train,
						const SGMatrix<double> &features_valid,
						const SGVector<double> &labels_valid,
						int num_rand_feats,
						int num_bags,
						int seed) {

	ASSERT_(features_train.num_rows == labels_train.vlen && features_valid.num_rows == labels_valid.vlen);
	int dim = features_train.num_cols;
	if (num_rand_feats > dim)
		num_rand_feats = dim;

	auto SGfeatures_train = create<Features>( transpose_matrix(features_train) );
	auto SGfeatures_valid = create<Features>( transpose_matrix(features_valid) );
	auto SGlabels_train = create<Labels>(labels_train);
	auto SGlabels_valid = create<Labels>(labels_valid);

	//![create_preprocessor]
	auto preproc = create<Transformer>("RescaleFeatures"); // to standardize data, just use 'PruneVarSubMean'
	preproc->fit(SGfeatures_train);
	//![create_preprocessor]

	//![transform_features]
	SGfeatures_train = preproc->transform(SGfeatures_train);
	SGfeatures_valid = preproc->transform(SGfeatures_valid);
	//![transform_features]


	//![create_combination_rule]
	auto comb_rule = create<CombinationRule>("MeanRule");
	//![create_combination_rule]

	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);


	//![create_instance]
	//auto rand_forest = create<Machine>("RandomForest");
	std::unique_ptr<RandomForest> rand_forest( new RandomForest(SGfeatures_train, SGlabels_train, num_bags, num_rand_feats) );
	rand_forest->set_feature_types(feature_type);
	rand_forest->set_machine_problem_type(EProblemType::PT_REGRESSION);
	//rand_forest->set_num_random_features(num_rand_feats);
	//rand_forest->put("num_bags", num_bags);
	rand_forest->set_combination_rule(comb_rule);
	rand_forest->put("seed", seed);
	rand_forest->set_solver_type(ST_AUTO);

	auto p = rand_forest->get_global_parallel(); // parallelize RF
	p->set_num_threads(num_threads);
	//set label data
	//rand_forest->put("labels", labels_train);
	//![create_instance]

	//![train the model]
	rand_forest->train(SGfeatures_train);
	//![train the model]

	// evaluate model on validation data
	auto new_labels = rand_forest->apply_regression(SGfeatures_valid);

	auto eval_criterium = create<Evaluation>("MeanSquaredError");
	auto mse = eval_criterium->evaluate(new_labels, SGlabels_valid);
	return mse;
}

/* Implement Random Forest with Cross Validation */
void ML_REG::RFRegression(	const SGMatrix<double> &features, /*columns are features*/
							const SGVector<double> &labels,
							const SGMatrix<double> &test_features,
							const SGVector<double> &test_labels,
							int num_subsets, int min_subset_size,
							SGVector<int> num_rand_feats_list,
							SGVector<int> num_bags_list,
							int seed) {

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_(features.num_rows == labels.vlen && Math::max<int>(num_rand_feats_list.vector, num_rand_feats_list.vlen) <= dim);

	// draw subsamples to train and validate the model
	int nonzero_train_size = 0, nonzero_valid_size = 0, nonzero_counter = 0;
	SGMatrix<int> train_indices(n, num_subsets), valid_indices(n, num_subsets);
	std::tie(train_indices, valid_indices) = tscv(labels, num_subsets, min_subset_size);

	SGVector<int> train_indices_i(n), valid_indices_i(n), nonzero_train_indices, nonzero_valid_indices;

	// define a matrix to save MSEs for all 'num_rand_feats' and 'num_bags'
	SGMatrix<double> mse_mat(num_rand_feats_list.vlen * num_bags_list.vlen, 3); // define a matrix to save MSEs for all 'num_rand_feats' and 'num_bags'
	mse_mat.zero();

	double mse_i = 0;
	for (int i = 0; i < num_subsets; ++i) {
		train_indices_i = train_indices.get_column(i);
		valid_indices_i = valid_indices.get_column(i);

		// construct a training subsample and a validation subsample
		nonzero_train_indices = train_indices_i.find_if(ML_REG::isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(ML_REG::isnotZero);
		nonzero_valid_size = nonzero_valid_indices.size();

		SGMatrix<double> features_train_i(nonzero_train_size, dim), features_valid_i(nonzero_valid_size, dim);
		SGVector<double> labels_train_i(nonzero_train_size), labels_valid_i(nonzero_valid_size);

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_train_indices) {
			for (int k = 0; k < dim; ++k) {
				features_train_i(nonzero_counter, k) = features(j, k);
			}
			labels_train_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_valid_indices) {
			for (int k = 0; k < dim; ++k) {
				features_valid_i(nonzero_counter, k) = features(j, k);
			}
			labels_valid_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}


		// loop through 'tree_max_depths_list','num_iters_list', 'learning_rates_list', and 'subset_fractions_list'
		// to obtain the hyperparameters with minimum RMSE
		int mse_counter = 0;
		for (int num_rand_feats : num_rand_feats_list){
			for (int num_bags : num_bags_list) {
				mse_i = ML_REG::RF_MSE(features_train_i, labels_train_i, features_valid_i, labels_valid_i, num_rand_feats, num_bags, seed);
				if (i == 0) {
					mse_mat(mse_counter, 0) = num_rand_feats;
					mse_mat(mse_counter, 1) = num_bags;
				}
				mse_mat(mse_counter, 2) += mse_i / num_subsets;
				mse_counter += 1;
			}
		}
	}

	mse_mat.display_matrix("mse_mat");

	// get the optimal 'num_feats' and 'num_bags'
	SGVector<double> mse = mse_mat.get_column(2);
	int min_index = Math::arg_min(mse.vector, 1, mse.vlen);
	int opt_num_rand_feats = static_cast<int>( mse_mat(min_index, 0) );
	int opt_num_bags = static_cast<int>( mse_mat(min_index, 1) );

	cout << "(optimal number of random features, optimal number of bags) = " << opt_num_rand_feats << " , " << opt_num_bags << endl;

	// RE-TRAIN the model using the optimal hyperparameters
	RF_Plot(features, labels, test_features, test_labels, opt_num_rand_feats, opt_num_bags, seed);
}


/* Perform RF CV with time-series splitting.
OUTPUT: optimal tree maximum depth, number of iterations, learning rate, subset fraction, number of random features, number of bags */
std::tuple<int, int, double, double, int, int> ML_REG::RF_cv(	const SGMatrix<double> &features, /*columns are features*/
																const SGVector<double> &labels,
																int num_subsets, int min_subset_size,
																SGVector<int> tree_max_depths_list,
																SGVector<int> num_iters_list,
																SGVector<double> learning_rates_list,
																SGVector<double> subset_fractions_list,
																SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
																SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
																int seed) {

	(void) tree_max_depths_list, num_iters_list, learning_rates_list, subset_fractions_list;

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_(features.num_rows == labels.vlen);

	// draw subsamples to train and validate the model
	int nonzero_train_size = 0, nonzero_valid_size = 0, nonzero_counter = 0;
	SGMatrix<int> train_indices(n, num_subsets), valid_indices(n, num_subsets);
	std::tie(train_indices, valid_indices) = tscv(labels, num_subsets, min_subset_size);

	SGVector<int> train_indices_i(n), valid_indices_i(n), nonzero_train_indices, nonzero_valid_indices;

	// define a matrix to save MSEs for all 'num_rand_feats' and 'num_bags'
	SGMatrix<double> mse_mat(num_rand_feats_list.vlen * num_bags_list.vlen, 3); // define a matrix to save MSEs for all 'num_rand_feats' and 'num_bags'
	mse_mat.zero();

	double mse_i = 0;
	for (int i = 0; i < num_subsets; ++i) {
		train_indices_i = train_indices.get_column(i);
		valid_indices_i = valid_indices.get_column(i);

		// construct a training subsample and a validation subsample
		nonzero_train_indices = train_indices_i.find_if(ML_REG::isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(ML_REG::isnotZero);
		nonzero_valid_size = nonzero_valid_indices.size();

		SGMatrix<double> features_train_i(nonzero_train_size, dim), features_valid_i(nonzero_valid_size, dim);
		SGVector<double> labels_train_i(nonzero_train_size), labels_valid_i(nonzero_valid_size);

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_train_indices) {
			for (int k = 0; k < dim; ++k) {
				features_train_i(nonzero_counter, k) = features(j, k);
			}
			labels_train_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_valid_indices) {
			for (int k = 0; k < dim; ++k) {
				features_valid_i(nonzero_counter, k) = features(j, k);
			}
			labels_valid_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}


		// loop through 'num_bags_list' and 'num_rand_feats_list' to obtain the hyperparameters with minimum RMSE
		int mse_counter = 0;
		for (int num_rand_feats : num_rand_feats_list){
			for (int num_bags : num_bags_list) {
				mse_i = ML_REG::RF_MSE(features_train_i, labels_train_i, features_valid_i, labels_valid_i, num_rand_feats, num_bags, seed);
				if (i == 0) {
					mse_mat(mse_counter, 0) = num_rand_feats;
					mse_mat(mse_counter, 1) = num_bags;
				}
				mse_mat(mse_counter, 2) += mse_i / num_subsets;
				mse_counter += 1;
			}
		}
	}

	//mse_mat.display_matrix("mse_mat");

	// get the optimal 'num_feats' and 'num_bags'
	SGVector<double> mse = mse_mat.get_column(2);
	int min_index = Math::arg_min(mse.vector, 1, mse.vlen);
	int opt_num_rand_feats = static_cast<int>( mse_mat(min_index, 0) );
	int opt_num_bags = static_cast<int>( mse_mat(min_index, 1) );

	//cout << "(optimal number of random features, optimal number of bags) = " << opt_num_rand_feats << " , " << opt_num_bags << endl;

	return {0, 0, 0., 0., opt_num_rand_feats, opt_num_bags};
}


/* Select RF hyperparameters with mininum MSE.
OUTPUT: optimal tree maximum depth, number of iterations, learning rate, subset fraction, number of random features, number of bags */
std::tuple<int, int, double, double, int, int> ML_REG::RF_cv1(	const SGMatrix<double> &features, /*columns are features*/
																const SGVector<double> &labels,
																int num_subsets, int min_subset_size,
																SGVector<int> tree_max_depths_list,
																SGVector<int> num_iters_list,
																SGVector<double> learning_rates_list,
																SGVector<double> subset_fractions_list,
																SGVector<int> num_rand_feats_list, /*list of numbers of random features used for bagging (for RF)*/
																SGVector<int> num_bags_list, /*list of number of bags (for RF)*/
																int seed) {

	(void) num_subsets, min_subset_size, tree_max_depths_list, num_iters_list, learning_rates_list, subset_fractions_list;

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_(features.num_rows == labels.vlen);


	// define a matrix to save MSEs for all 'num_rand_feats' and 'num_bags'
	SGMatrix<double> mse_mat(num_rand_feats_list.vlen * num_bags_list.vlen, 3); // define a matrix to save MSEs for all 'num_rand_feats' and 'num_bags'
	mse_mat.zero();

	// loop through 'num_bags_list' and 'num_rand_feats_list' to obtain the hyperparameters with minimum RMSE
	int mse_counter = 0;
	for (int num_rand_feats : num_rand_feats_list) {
		for (int num_bags : num_bags_list) {
			mse_mat(mse_counter, 0) = num_rand_feats;
			mse_mat(mse_counter, 1) = num_bags;
			mse_mat(mse_counter, 2) = ML_REG::RF_MSE(features, labels, features, labels, num_rand_feats, num_bags, seed);
			mse_counter += 1;
		}
	}


	//mse_mat.display_matrix("mse_mat");

	// get the optimal 'num_feats' and 'num_bags'
	SGVector<double> mse = mse_mat.get_column(2);
	int min_index = Math::arg_min(mse.vector, 1, mse.vlen);
	int opt_num_rand_feats = static_cast<int>( mse_mat(min_index, 0) );
	int opt_num_bags = static_cast<int>( mse_mat(min_index, 1) );

	cout << "(optimal number of random features, optimal number of bags) = " << opt_num_rand_feats << " , " << opt_num_bags << endl;

	return {0, 0, 0., 0., opt_num_rand_feats, opt_num_bags};
}


#if 0
void RFRegression(const SGMatrix<double> &features, /*columns are features*/
				  const SGVector<double> &labels,
				  const SGMatrix<double> &test_features,
				  const SGVector<double> &test_labels,
				  int num_subsets, int min_subset_size,
				  SGVector<int> num_rand_feats_list,
				  SGVector<int> num_bags_list,
				  int seed) {

	int n = features.num_rows, dim = features.num_cols;
	ASSERT_(num_rand_feats_list.vlen <= dim);

	// draw subsamples to train and validate the model
	int nonzero_train_size = 0, nonzero_valid_size = 0, nonzero_counter = 0;
	SGMatrix<int> train_indices(n, num_subsets), valid_indices(n, num_subsets);
	std::tie(train_indices, valid_indices) = tscv(labels, num_subsets, min_subset_size);

	auto vote = some<CMajorityVote>();
	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);

	// evaluate the model by using the RMSE
	auto eval_criterium = some<CMeanSquaredError>();

	// define a random forest object
	auto rand_forest = new CRandomForest();
	auto p = wrap(rand_forest->get_global_parallel() );
	p->set_num_threads(num_threads);
	rand_forest->put("seed", seed);
	rand_forest->set_combination_rule(vote);
	rand_forest->set_feature_types(feature_type);
	rand_forest->set_machine_problem_type(PT_REGRESSION);
	rand_forest->set_solver_type(ST_ELASTICNET);
	if (dim > 1)
		rand_forest->set_bag_size( static_cast<int>(dim/2) );

	SGVector<int> train_indices_i(n), valid_indices_i(n), nonzero_train_indices, nonzero_valid_indices;
	SGMatrix<double> mse_mat(num_rand_feats_list.vlen * num_bags_list.vlen, 3); // define a matrix to save MSEs for all 'num_feats' and 'num_bags'
	mse_mat.zero();

	for (int i = 0; i < num_subsets; ++i) {
		train_indices_i = train_indices.get_column(i);
		valid_indices_i = valid_indices.get_column(i);

		// construct a training subsample and a validation subsample
		nonzero_train_indices = train_indices_i.find_if(ML_REG::isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(ML_REG::isnotZero);
		nonzero_valid_size = nonzero_valid_indices.size();

		SGMatrix<double> features_train_i(nonzero_train_size, dim), features_valid_i(nonzero_valid_size, dim);
		SGVector<double> labels_train_i(nonzero_train_size), labels_valid_i(nonzero_valid_size);

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_train_indices) {
			for (int k = 0; k < dim; ++k) {
				features_train_i(nonzero_counter, k) = features(j, k);
			}
			labels_train_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		nonzero_counter = 0; // reset the counter
		for (int j : nonzero_valid_indices) {
			for (int k = 0; k < dim; ++k) {
				features_valid_i(nonzero_counter, k) = features(j, k);
			}
			labels_valid_i[nonzero_counter] = labels[j];
			nonzero_counter += 1;
		}

		// construct training and validation data objects
		auto SGfeatures_train_i = some<CDenseFeatures<double>>( transpose_matrix(features_train_i) );
		auto SGlabels_train_i = some<CRegressionLabels>(labels_train_i);
		auto SGfeatures_valid_i = some<CDenseFeatures<double>>( transpose_matrix(features_valid_i) );
		auto SGlabels_valid_i = some<CRegressionLabels>(labels_valid_i);

		// loop through 'num_rand_feats_list' and 'num_bags_list' to obtain the hyperparameters with minimum RMSE
		int mse_counter = 0;
		for (int num_rand_feats : num_rand_feats_list){
			for (int num_bags : num_bags_list) {
				rand_forest->set_num_random_features(num_rand_feats);
				rand_forest->set_num_bags(num_bags);
				rand_forest->set_labels(SGlabels_train_i);

				// train the model
				rand_forest->train(SGfeatures_train_i);

				// evaluate the model on the validation data
				auto new_SGlabels_valid_i = wrap( rand_forest->apply_regression(SGfeatures_valid_i) );
				mse_mat(mse_counter, 0) = num_rand_feats;
				mse_mat(mse_counter, 1) = num_bags;
				mse_mat(mse_counter, 2) += eval_criterium->evaluate(new_SGlabels_valid_i, SGlabels_valid_i) / num_subsets;
				mse_counter += 1;
			}
		}
	}
	mse_mat.display_matrix("mse_mat");

	// get the optimal 'num_feats' and 'num_bags'
	SGVector<double> mse = mse_mat.get_column(2);
	int min_index = CMath::arg_min(mse.vector, 1, mse.vlen);
	int opt_num_rand_feats = static_cast<int>( mse_mat(min_index, 0) );
	int opt_num_bags = static_cast<int>( mse_mat(min_index, 1) );

	cout << "(optimal number of random features, optimal number of bags) = " << opt_num_rand_feats << " , " << opt_num_bags << endl;

	// re-train the model on the optimal hyperparameters
	rand_forest->set_num_random_features(opt_num_rand_feats);
	rand_forest->set_num_bags(opt_num_bags);

	auto SGfeatures_train = some<CDenseFeatures<double>>( transpose_matrix(features) );
	auto SGlabels_train = some<CRegressionLabels>(labels);
	auto SGfeatures_test = some<CDenseFeatures<double>>( transpose_matrix(test_features) );
	auto SGlabels_test = some<CRegressionLabels>(test_labels);

	rand_forest->set_labels(SGlabels_train);

	// train the model
	rand_forest->train(SGfeatures_train);

	// evaluate the trained model
	auto new_SGlabels_train = wrap( rand_forest->apply_regression(SGfeatures_train) );
	auto accuracy = eval_criterium->evaluate(new_SGlabels_train, SGlabels_train);
	cout << "The MSE = " << accuracy << endl;

	PlotResults(SGfeatures_train, SGlabels_train, new_SGlabels_train, "Shogun Random Forest: Train", "shogun-rf-train");

	// evaluate the model on the test data
	auto new_SGlabels_test = wrap( rand_forest->apply_regression(SGfeatures_test) );
	accuracy = eval_criterium->evaluate(new_SGlabels_test, SGlabels_test);
	cout << "The Forecast MSE = " << accuracy << endl;

	PlotResults(SGfeatures_test, SGlabels_test, new_SGlabels_test, "Shogun Random Forest: Test", "shogun-rf-test");
}

#endif


























#endif //ml_reg.h
