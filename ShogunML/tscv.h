/* This script is used to construct vectors of indices for training and validation samples  */

#ifndef TSCV_H
#define TSCV_H

#include <typeinfo>
//#include <shogun/base/some.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/io/File.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/serialization/Serializer.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/SGVector.h>
#include <shogun/util/factory.h>


using namespace std;
using namespace shogun;
using namespace shogun::io;

pair< SGMatrix<int>, SGMatrix<int> > tscv(SGVector<double> y_values, int num_subsets, int min_subset_size) {

	int n = y_values.size();

	//![create labels]
	auto reg_labels = create<Labels>(y_values); //wrap( Labels(y_values) );
	//![create labels]

//		Some<CRegressionLabels> creg_labels =  reg_labels;
//		auto value_labels = creg_labels ->get_labels();
//		cout << value_labels[20] << endl;

	//![set parameters]
	auto splitting = create<SplittingStrategy>("TimeSeriesSplitting");
	splitting->put("labels", reg_labels);
	splitting->put("num_subsets", num_subsets);
	splitting->put("min_subset_size", min_subset_size);
	//![set parameters]

	//![build subsets]
	splitting->build_subsets();
	//![build subsets]

	//![generate subsets and inverse (aka test labels and train labels)]
	SGVector<int> train_labels_indices, test_labels_indices;
	SGMatrix<int> train_indices(n, num_subsets), test_indices(n, num_subsets);
	train_indices.zero();
	test_indices.zero();
	for (int i = 0 ; i < num_subsets; ++i) {
		test_labels_indices = splitting->generate_subset_indices(i);
		train_labels_indices = splitting->generate_subset_inverse(i);
		for (int j = 0; j < train_labels_indices.size(); ++j) {
			train_indices(j, i) =  train_labels_indices[j];
		}
		for (int j = 0; j < test_labels_indices.size(); ++j) {
			test_indices(j, i) = test_labels_indices[j];
		}
	}
	//![generate subsets and inverse (aka test labels and train labels)]

	return {train_indices, test_indices};
}



#endif // tscv.h
