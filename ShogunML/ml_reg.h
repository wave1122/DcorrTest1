#ifndef ML_REG_H
#define ML_REG_H

#include <ShogunML/data/data.h>
#include <plot.h>
#include "utils.h"


#include <shogun/lib/config.h>
//#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/multiclass/tree/CARTree.h>
#include <shogun/util/factory.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
//#include <shogun/modelselection/GridSearchModelSelection.h>
//#include <shogun/modelselection/ModelSelectionParameters.h>
//#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <shogun/transformer/Transformer.h>

#include <experimental/filesystem>
#include <iostream>
#include <map>
#include <ShogunML/tscv.h>

// namespace fs = std::experimental::filesystem;
using namespace std;
using namespace shogun;
using namespace shogun::linalg;

using MatrixSG = SGMatrix<DataType>;
using Vector = SGVector<DataType>;

/* Define a predicate to select nonzero elements from a SG vector */
bool isnotZero(double x) {
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

std::pair<MatrixSG, Vector> GenerateShogunData(double s,
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

void PlotResults(Some<CDenseFeatures<DataType>> test_features,
                 Some<CRegressionLabels> test_labels,
                 Some<CRegressionLabels> pred_labels,
                 const std::string& title,
                 const std::string& file_name) {
  auto x_coords = test_features->get_feature_matrix();
  auto y_coords = test_labels->get_labels();
  auto y_pred_coords = pred_labels->get_labels();

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

     SGVector<DataType> x_coord0 =  x_coords.get_row_vector(i);
     plt.Draw2D(
        plotcpp::Points(x_coord0.begin(), x_coord0.end(), y_coords.begin(),
                      "actual", "lc rgb 'black' pt 7"),
        /*plotcpp::Lines(x_coord0.begin(), x_coord0.end(), y_pred_coords.begin(),
                     "pred", "lc rgb 'red' lw 2") );*/
        plotcpp::Points(x_coord0.begin(), x_coord0.end(), y_pred_coords.begin(),
					  "predicted", "lc rgb 'red' pt 9")
		);
  }
  plt.Flush();
}

void GBMClassification(Some<CDenseFeatures<DataType>> features,
                       Some<CRegressionLabels> labels,
                       Some<CDenseFeatures<DataType>> test_features,
                       Some<CRegressionLabels> test_labels, int dim) {

  // mark feature type as continuous
  SGVector<bool> feature_type(dim);
  feature_type.set_const(false);
  /*
   * A CART tree is a binary decision tree that is constructed by splitting a
   * node into two child nodes repeatedly, beginning with the root node that
   * contains the whole dataset.
   */
  auto tree = some<CCARTree>(feature_type, PT_REGRESSION);
  // try to change tree depth to see its influence on accuracy
  tree->set_max_depth(3);
  auto loss = some<CSquaredLoss>();

  // GBM supports only regression
  // try to change learning rate to see its influence on accuracy
  auto sgbm = some<CStochasticGBMachine>(tree, loss, /*iterations*/ 100,
                                         /*learning rate*/ 0.1, 1.0);
  sgbm->set_labels(labels);
  sgbm->train(features);

  // evaluate model on test data
  auto new_labels = wrap(sgbm->apply_regression(test_features));

  auto eval_criterium = some<CMeanSquaredError>();
  auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
  std::cout << "GBM classification accuracy = " << accuracy << std::endl;

  PlotResults(test_features, test_labels, new_labels, "Shogun Gradient Boosting", "shogun-gbm");
}

double GBM (CDenseFeatures<double> *SGfeatures_train, CRegressionLabels *SGlabels_train,
												CDenseFeatures<double> *SGfeatures_valid, CRegressionLabels *SGlabels_valid,
												int tree_max_depth, int num_iters, double learning_rate,
												double subset_fraction, int seed) {
	int dim = SGfeatures_train->get_num_features();

	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);

	/*
	   * A CART tree is a binary decision tree that is constructed by splitting a
	   * node into two child nodes repeatedly, beginning with the root node that
	   * contains the whole dataset.
	*/
	auto tree = some<CCARTree>(feature_type, PT_REGRESSION);
	tree->set_max_depth(tree_max_depth);
	auto loss = some<CSquaredLoss>();
	auto eval_criterium = some<CMeanSquaredError>();

	// set up the machine
	CStochasticGBMachine *sgbm = new CStochasticGBMachine(tree, loss, num_iters, learning_rate, subset_fraction);
	SG_REF(sgbm);
	sgbm->put("seed", seed);
	sgbm->set_solver_type(ST_AUTO);
	auto p = wrap( sgbm->get_global_parallel() );
	p->set_num_threads(1);
	sgbm->set_labels(SGlabels_train);

	// train the model
	sgbm->train(SGfeatures_train);

	// evaluate the model on the validation data
	auto new_SGlabels_valid = wrap( sgbm->apply_regression(SGfeatures_valid) );
	double mse = eval_criterium->evaluate(new_SGlabels_valid, SGlabels_valid);
	SG_UNREF(sgbm);
	return mse;
}

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
	ASSERT( (features.num_rows == labels.vlen) && (CMath::max(learning_rates_list.vector, learning_rates_list.vlen) <= 1.) \
											   && (CMath::max(subset_fractions_list.vector, subset_fractions_list.vlen) <= 1.) );

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
		nonzero_train_indices = train_indices_i.find_if(isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(isnotZero);
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
			for (int num_iters : num_iters_list) {
				for (double learning_rate : learning_rates_list) {
					for (double subset_fraction : subset_fractions_list) {

						mse_i = GBM(SGfeatures_train_i, SGlabels_train_i,SGfeatures_valid_i, SGlabels_valid_i,
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

	// mark feature type as continuous
	SGVector<bool> feature_type(dim);
	feature_type.set_const(false);

	auto tree = some<CCARTree>(feature_type, PT_REGRESSION);
	tree->set_max_depth(opt_tree_max_depth);
	auto loss = some<CSquaredLoss>();

	auto sgbm = some<CStochasticGBMachine>(tree, loss, opt_num_iters, opt_learning_rate, opt_subset_fraction);
	sgbm->put("seed", seed);
	sgbm->set_solver_type(ST_AUTO);
	auto p = wrap( sgbm->get_global_parallel() );
	p->set_num_threads(1);
	sgbm->set_labels(SGlabels_train);


	// train the model
	sgbm->train(SGfeatures_train);

	// evaluate the trained model
	auto eval_criterium = some<CMeanSquaredError>();
	auto new_SGlabels_train = wrap( sgbm->apply_regression(SGfeatures_train) );
	auto accuracy = eval_criterium->evaluate(new_SGlabels_train, SGlabels_train);
	cout << "The MSE = " << accuracy << endl;

	PlotResults(SGfeatures_train, SGlabels_train, new_SGlabels_train, "Shogun GBM: Train", "shogun-gbm-train");

	// evaluate the model on the test data
	auto new_SGlabels_test = wrap( sgbm->apply_regression(SGfeatures_test) );
	accuracy = eval_criterium->evaluate(new_SGlabels_test, SGlabels_test);
	cout << "The Forecast MSE = " << accuracy << endl;

	PlotResults(SGfeatures_test, SGlabels_test, new_SGlabels_test, "Shogun GBM: Test", "shogun-gbm-test");
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
	ASSERT( (features.num_rows == labels.vlen) && (CMath::max(learning_rates_list.vector, learning_rates_list.vlen) <= 1.) \
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
	p->set_num_threads(1);


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
		nonzero_train_indices = train_indices_i.find_if(isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(isnotZero);
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
#endif


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
	ASSERT(features.num_rows == labels0.vlen);

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
	p->set_num_threads(1);

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


void RFClassification(Some<CDenseFeatures<DataType>> features,
					  Some<CRegressionLabels> labels,
                      Some<CDenseFeatures<DataType>> test_features,
                      Some<CRegressionLabels> test_labels, int dim) {
  // number of attributes chosen randomly during node split in candidate trees
  int32_t num_rand_feats = 1;
  // number of trees in forest
  int32_t num_bags = 10;

  auto rand_forest = shogun::some<shogun::CRandomForest>(num_rand_feats, num_bags);
  auto p = wrap(rand_forest->get_global_parallel() );
  p->set_num_threads(1);
  rand_forest->put("seed", 1);

  auto vote = shogun::some<shogun::CMajorityVote>();
  rand_forest->set_combination_rule(vote);

  // mark feature type as continuous
  SGVector<bool> feature_type(dim);
  feature_type.set_const(false);
  rand_forest->set_feature_types(feature_type);

  rand_forest->set_labels(labels);
  rand_forest->set_machine_problem_type(PT_REGRESSION);
  rand_forest->train(features);

  // evaluate model on test data
  auto new_labels = wrap(rand_forest->apply_regression(test_features) );

  auto eval_criterium = some<CMeanSquaredError>();
  auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
  std::cout << "RF classification accuracy = " << accuracy << std::endl;

  PlotResults(test_features, test_labels, new_labels, "Shogun Random Forest", "shogun-rf");
}



void RFRegression(const SGMatrix<double> &features, /*columns are features*/
				  const SGVector<double> &labels,
				  const SGMatrix<double> &test_features,
				  const SGVector<double> &test_labels,
				  int num_subsets, int min_subset_size,
				  SGVector<int> num_rand_feats_list,
				  SGVector<int> num_bags_list,
				  int seed) {

	int n = features.num_rows, dim = features.num_cols;
	ASSERT(num_rand_feats_list.vlen <= dim);

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
	p->set_num_threads(1);
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
		nonzero_train_indices = train_indices_i.find_if(isnotZero);
		nonzero_train_size = nonzero_train_indices.size();
		nonzero_valid_indices = valid_indices_i.find_if(isnotZero);
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


























#endif //ml_reg.h
