/* Author: Sudnya Padalikar
 * Date  : 11/23/2013
 * A unit test that implements a neural network to perform face detection on a set of images
*/

#include <minerva/classifiers/interface/ClassifierFactory.h>
#include <minerva/classifiers/interface/ClassifierEngine.h>
#include <minerva/model/interface/ClassificationModel.h>
#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/ArgumentParser.h>

#include <random>
#include <cstdlib>
#include <memory>

namespace minerva
{
namespace classifiers
{

typedef neuralnetwork::Layer Layer;
typedef matrix::Matrix Matrix;

neuralnetwork::NeuralNetwork createAndInitializeNeuralNetwork(unsigned networkSize, float epsilon)
{
    neuralnetwork::NeuralNetwork ann;
    // Layer 1
    ann.addLayer(Layer(1,networkSize,networkSize));
    ann.addLayer(Layer(1,networkSize,networkSize));
    ann.addLayer(Layer(1,networkSize,1));

    ann.initializeRandomly(epsilon);

    return ann;
}

void trainNeuralNetwork(neuralnetwork::NeuralNetwork& ann, unsigned trainingIter, std::default_random_engine& generator)
{
	
    /*unsigned samplesPerIter = ann.getInputCount() * 10;

    util::log("TestClassifier") << "Starting training\n";

    for(unsigned i = 0; i < trainingIter; ++i)
    {
        // matrix is (samples) rows x (features) columns
        Matrix input = generateRandomMatrix(samplesPerIter, ann.getInputCount(), generator);
        Matrix referenceMatrix = matrixXor(input);

        util::log("TestClassifier") << " Input is:     " << input.toString();
        util::log("TestClassifier") << " Output is:    " << threshold(ann.runInputs(input)).toString();
        util::log("TestClassifier") << "  Output entropy is " << computeEntropy(threshold(ann.runInputs(input))) << "\n";
        util::log("TestClassifier") << " Reference is: " << referenceMatrix.toString();

        ann.train(input, referenceMatrix);
        util::log("TestClassifier") << " After BackProp, output is:    " << threshold(ann.runInputs(input)).toString();
    } */   
}

void runTest(unsigned iterations, bool seed, unsigned networkSize, float epsilon)
{
	// create a neural network - simple, 3 layer - done in runTest and passed here
    neuralnetwork::NeuralNetwork ann = createAndInitializeNeuralNetwork(networkSize, epsilon); 
	
	// add it to the model, hardcode the resolution for these images
	model::ClassificationModel faceModel("temp");
	faceModel.setNeuralNetwork("face-detector", ann);

	// the images of faces and not faces are now stored as image vector - no need to worry about this anymore
	
	// engine will now be a Learner
	std::unique_ptr<classifiers::ClassifierEngine> learnerEngine(classifiers::ClassifierFactory::create("LearnerEngine"));

	learnerEngine->setModel(&faceModel);

	// read from database and use model to train
	// read from test-database and use model to classify
    learnerEngine->runOnDatabaseFile("examples/faces.txt");

	

    // Run classifier and record accuracy

    //TODO float accuracy = classify(ann, std::max(1U, iterations/10), generator);

	float accuracy = 0.1; //TODO write a function to set this
    // Test if accuracy is greater than threshold
	// if > 90% accurate - say good
	// if < 90% accurate - say bad

    float threshold = 0.90;
    if (accuracy > threshold)
    {
        std::cout << "Test passed with accuracy " << accuracy 
            << " which is more than expected threshold " << threshold << "\n";
    }
    else
    {
        std::cout << "Test FAILED with accuracy " << accuracy 
            << " which is less than expected threshold " << threshold << "\n";
    }
}



static void enableSpecificLogs(const std::string& modules)
{
	auto individualModules = util::split(modules, ",");
	
	for(auto& module : individualModules)
	{
		util::enableLog(module);
	}
}


}
}

int main(int argc, char** argv)
{
    minerva::util::ArgumentParser parser(argc, argv);
    
    bool verbose = false;
    bool seed = false;
    std::string loggingEnabledModules;
	
	unsigned iterations = 0;
	unsigned networkSize = 0;
	float epsilon = 1.0f;

    parser.description("The Minerva face detection classifier.");

    parser.parse("-i", "--iterations", iterations, 1000,
        "The number of iterations to train for");
    parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-s", "--seed", seed, false,
        "Seed with time.");
    parser.parse("-n", "--network-size", networkSize, 100,
        "The number of inputs to the network.");
    parser.parse("-e", "--epsilon", epsilon, 1.0f,
        "Range to intiialize the network with.");
    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

	parser.parse();

    if(verbose)
	{
		minerva::util::enableAllLogs();
	}
	else
	{
		minerva::classifiers::enableSpecificLogs(loggingEnabledModules);
	}
    
    minerva::util::log("TestClassifier") << "Test begins\n";
    
    try
    {
        minerva::classifiers::runTest(iterations, seed, networkSize, epsilon);
    }
    catch(const std::exception& e)
    {
        std::cout << "Minerva Face Detection Classifier Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}


