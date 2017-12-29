// MNISTCNNToy.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "NeuralNet.h"
#include "ConvolutionLayer.h"
#include "BiasLayer.h"
#include "DenseLayer.h"
#include "TanHLayer.h"
#include "SigmoidLayer.h"
#include "SoftmaxLayer.h"
#include "TrainingFunctions.h"
#include "PaddedConvolutionLayer.h"

int IntEndSwap(int input)
{
	char* ptr = ((char*)&input);
	ptr[0] ^= ptr[3];
	ptr[3] ^= ptr[0];
	ptr[0] ^= ptr[3];

	ptr[1] ^= ptr[2];
	ptr[2] ^= ptr[1];
	ptr[1] ^= ptr[2];

	return input;
}

int main()
{
	af::setDevice(0);
	af::info();

	using namespace UAFML;
	int num_train, num_test, width, height;
	af::array train_labels, train_images, test_labels, test_images;

	try
	{
		train_images = af::readArray("mnist_set.arr", "mnist_train_img");
		num_train = (int)train_images.dims(0);
	}
	catch (...)
	{
		std::ifstream mnist_file("train-images-idx3-ubyte\\train-images.idx3-ubyte", std::ios::binary);
		std::stringstream buffer;
		unsigned char item;
		int magic_number;

		if (mnist_file.is_open())
		{
			buffer << mnist_file.rdbuf();
			mnist_file.close();
			buffer.read((char*)&magic_number, 4);
			magic_number = IntEndSwap(magic_number);
			buffer.read((char*)&num_train, 4);
			num_train = IntEndSwap(num_train);
			buffer.read((char*)&height, 4);
			height = IntEndSwap(height);
			buffer.read((char*)&width, 4);
			width = IntEndSwap(width);

			train_images = af::constant(0, num_train, 1, height, width);

			for (int item_idx = 0; item_idx < num_train; item_idx++)
			{
				for (int row = 0; row < height; row++)
				{
					for (int col = 0; col < width; col++)
					{
						buffer.read((char*)&item, 1);
						train_images(item_idx, 0, row, col) = item;
					}
				}
				if (item_idx % (num_train / 10) == (num_train / 10) - 1)
					std::cout << '|';
				else if (item_idx % (num_train / 100) == 0)
					std::cout << '.';
			}
			std::cout << '\n';

			af::saveArray("mnist_train_img", train_images, "mnist_set.arr", true);
		}
	}

	try
	{
		test_images = af::readArray("mnist_set.arr", "mnist_test_img");
		num_test = (int)test_images.dims(0);
	}
	catch (...)
	{
		std::ifstream mnist_file("t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte", std::ios::binary);
		std::stringstream buffer;
		unsigned char item;
		int magic_number;

		if (mnist_file.is_open())
		{
			buffer << mnist_file.rdbuf();
			mnist_file.close();
			buffer.read((char*)&magic_number, 4);
			magic_number = IntEndSwap(magic_number);
			buffer.read((char*)&num_test, 4);
			num_test = IntEndSwap(num_test);
			buffer.read((char*)&height, 4);
			height = IntEndSwap(height);
			buffer.read((char*)&width, 4);
			width = IntEndSwap(width);

			test_images = af::constant(0, num_test, 1, height, width);

			for (int item_idx = 0; item_idx < num_test; item_idx++)
			{
				for (int row = 0; row < height; row++)
				{
					for (int col = 0; col < width; col++)
					{
						buffer.read((char*)&item, 1);
						test_images(item_idx, 0, row, col) = item;
					}
				}
				if (item_idx % (num_test / 10) == (num_test / 10) - 1)
					std::cout << '|';
				else if (item_idx % (num_test / 100) == 0)
					std::cout << '.';
			}
			std::cout << '\n';

			af::saveArray("mnist_test_img", test_images, "mnist_set.arr", true);
		}
	}

	try
	{
		train_labels = af::readArray("mnist_set.arr", "mnist_train_lbl");
		if (train_labels.dims(0) != num_train)
			throw;
	}
	catch (...)
	{
		std::ifstream mnist_file("train-labels-idx1-ubyte\\train-labels.idx1-ubyte", std::ios::binary);
		std::stringstream buffer;
		unsigned char item;
		int magic_number;

		if (mnist_file.is_open())
		{
			buffer << mnist_file.rdbuf();
			mnist_file.close();
			buffer.read((char*)&magic_number, 4);
			magic_number = IntEndSwap(magic_number);
			buffer.read((char*)&num_train, 4);
			num_train = IntEndSwap(num_train);

			train_labels = af::constant(0, num_train, 10, 1, 1);

			for (int item_idx = 0; item_idx < num_train; item_idx++)
			{
				buffer.read((char*)&item, 1);
				train_labels(item_idx, item) = 1;
			}

			af::saveArray("mnist_train_lbl", train_labels, "mnist_set.arr", true);
		}
	}

	try
	{
		test_labels = af::readArray("mnist_set.arr", "mnist_test_lbl");
		if (test_labels.dims(0) != num_test)
			throw;
	}
	catch (...)
	{
		std::ifstream mnist_file("t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte", std::ios::binary);
		std::stringstream buffer;
		unsigned char item;
		int magic_number;

		if (mnist_file.is_open())
		{
			buffer << mnist_file.rdbuf();
			mnist_file.close();
			buffer.read((char*)&magic_number, 4);
			magic_number = IntEndSwap(magic_number);
			buffer.read((char*)&num_test, 4);
			num_test = IntEndSwap(num_test);

			test_labels = af::constant(0, num_test, 10, 1, 1);

			for (int item_idx = 0; item_idx < num_test; item_idx++)
			{
				buffer.read((char*)&item, 1);
				test_labels(item_idx, item) = 1;
			}

			af::saveArray("mnist_test_lbl", test_labels, "mnist_set.arr", true);
		}
	}

	height = (int)train_images.dims(2);
	width = (int)train_images.dims(3);

	//af::Window window(height * 10, width * 10, "Training Example");
	//window.image(1 - af::reorder(train_images(rand() % num_train, 0, af::span, af::span), 2, 3, 0, 1) / 255.0f);

	UAFML::NeuralNet mnist_cnn(0.01);
	if(0)
	{
		mnist_cnn.AddLayer<ConvolutionLayer>(af::dim4(20, 1, 3, 3));
		mnist_cnn.AddLayer<BiasLayer>(af::dim4(1, 20, height - 2, width - 2));
		mnist_cnn.AddLayer<TanHLayer>();

		mnist_cnn.AddLayer<ConvolutionLayer>(af::dim4(15, 20, 3, 3));
		mnist_cnn.AddLayer<BiasLayer>(af::dim4(1, 15, height - 4, width - 4));
		mnist_cnn.AddLayer<TanHLayer>();

		mnist_cnn.AddLayer<ConvolutionLayer>(af::dim4(10, 15, 3, 3));
		mnist_cnn.AddLayer<BiasLayer>(af::dim4(1, 10, height - 6, width - 6));
		mnist_cnn.AddLayer<TanHLayer>();

		mnist_cnn.AddLayer<DenseLayer>(af::dim4(10 * (height - 6) * (width - 6), 10));
		mnist_cnn.AddLayer<BiasLayer>(af::dim4(1, 10));
		//mnist_cnn.AddLayer<TanHLayer>();

		mnist_cnn.AddLayer<SoftmaxLayer>();
	}
	else if (1)
	{
		mnist_cnn.AddLayer<DenseLayer>(af::dim4(height * width, 10));
		mnist_cnn.AddLayer<BiasLayer>(af::dim4(1, 10));
		mnist_cnn.AddLayer<SigmoidLayer>();

		mnist_cnn.AddLayer<DenseLayer>(af::dim4(10, 10));
		mnist_cnn.AddLayer<BiasLayer>(af::dim4(1, 10));
		mnist_cnn.AddLayer<SigmoidLayer>();
	}

	if (0 && !CheckGradient(mnist_cnn, af::dim4(1, 1, height, width), 10, 7960, f64))
		std::cout << "Gradient check FAILED.\n";
	else
	{
		std::cout << "Gradient check successful.\n";
		af::array train_points = af::constant(0.0, 1, 2);
		af::array test_points = af::constant(100.0, 1, 2);
		test_points(0, 0) = 0.0;
		bool stop = false;
		std::mutex plot_lock;

		std::thread training_thread([&]()->void{
			for (int num = 1; num <= 80; ++num)
			{
				af::array weights, new_weights, best_weights = weights = InitializeWeights(mnist_cnn.GetWeightsSize());
				double cost, new_cost, best_cost = DBL_MAX;
				auto set = RandomPermutation(num, num_train);
				for (size_t i = 0; i < 3; i++)
				{
					new_weights = InitializeWeights(mnist_cnn.GetWeightsSize());

					ConjugateGradientDescent(mnist_cnn, train_images(set, af::span), train_labels(set, af::span), weights);
					ConjugateGradientDescent(mnist_cnn, train_images(set, af::span), train_labels(set, af::span), new_weights);

					cost = mnist_cnn.CalculateCost(mnist_cnn.ForwardPropagate(train_images(set, af::span), weights), train_labels(set, af::span), weights);
					new_cost = mnist_cnn.CalculateCost(mnist_cnn.ForwardPropagate(train_images(set, af::span), new_weights), train_labels(set, af::span), new_weights);

					if (new_cost < cost && new_cost < best_cost)
					{
						best_cost = new_cost;
						best_weights = new_weights;
					}
					else if (cost < best_cost)
					{
						best_cost = cost;
						best_weights = weights;
					}

					weights = ((1 / new_cost) * new_weights + (1 / cost) * weights) / ((cost + new_cost) / (cost * new_cost));
				}

				weights = best_weights;
				cost = best_cost;
				af::array train_point(1, 2), test_point(1, 2);
				train_point(0, 0) = num;
				train_point(0, 1) = cost;
				std::cout << "***TRAINING COST FOR " << num << " EXAMPLES:\t"
					<< std::setw(10) << std::setfill(' ') << std::setprecision(6) << std::fixed
					<< cost << '\n';

				cost = mnist_cnn.CalculateCost(mnist_cnn.ForwardPropagate(test_images, weights), test_labels, weights);
				test_point(0, 0) = num;
				test_point(0, 1) = cost;
				std::cout << "***TESTING COST FOR  " << num << " EXAMPLES:\t"
					<< std::setw(10) << std::setfill(' ') << std::setprecision(6) << std::fixed
					<< cost << '\n';

				plot_lock.lock();
				train_points = af::join(0, train_points, train_point);
				test_points = af::join(0, test_points, test_point);
				plot_lock.unlock();
			}
			system("PAUSE");
			stop = true;
		});

		af::Window plot(750, 750, "Learning Curve");
		plot.grid(2, 1);
		plot(0, 0).setTitle("Training");
		plot(1, 0).setTitle("Testing");
		plot(0, 0).setAxesTitles("Cost", "Training Examples", "");
		plot(1, 0).setAxesTitles("Cost", "Training Examples", "");
		plot(0, 0).setAxesLimits(0.f, 80.f, 0.f, 16.f, true);
		plot(1, 0).setAxesLimits(0.f, 80.f, 0.f, 16.f, true);

		while (!stop)
		{
			if (std::cout.bad())
			{
				std::cout.clear();
				std::cout << "\nClearing Bad Console State..." << std::endl;
			}

			if (plot_lock.try_lock())
			{
				plot(0, 0).plot(train_points);
				plot(1, 0).plot(test_points);
				plot.show();

				plot_lock.unlock();
			}
		}
		training_thread.join();
	}
	return 0;
}
