/*	\file   minerva-make-dataset.cpp
	\date   Saturday August 10, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  A tool for splitting a dataset into training/validation sets.
*/

// Minerva Includes
#include <minerva/database/interface/SampleDatabase.h>
#include <minerva/database/interface/Sample.h>

#include <minerva/video/interface/Image.h>

#include <minerva/util/interface/ArgumentParser.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>
#include <fstream>
#include <memory>
#include <algorithm>

namespace minerva
{

static void createDirectories(const std::string& outputPath, const std::string& setName, const util::StringVector& labels)
{
    for(auto& label : labels)
    {
        util::makeDirectory(util::joinPaths(outputPath, util::joinPaths(setName, label)));
    }
}

static void copySampleToDatabase(database::SampleDatabase& outputDatabase, const database::Sample& sample)
{
    auto directory = util::getDirectory(outputDatabase.path());

    auto path = util::joinPaths(directory, util::joinPaths(sample.label(), util::getFile(sample.path())));

    util::log("MinervaMakeDataset") << "Copying sample '" + sample.path() + "' to '" + path + "'\n";

    util::copyFile(path, sample.path());

    outputDatabase.addSample(database::Sample(util::joinPaths(sample.label(), util::getFile(sample.path())), sample.label()));
}

static bool validateSample(const database::Sample& sample)
{
    if(sample.isImageSample() && video::Image::isPathAnImage(sample.path()))
    {
        video::Image image(sample.path());

        try
        {
            image.loadHeader();

            if(!image.headerLoaded())
            {
                return false;
            }
        }
        catch(const std::exception& e)
        {
            return false;
        }

        return true;
    }

    return false;
}

static void randomlyShuffleSamples(database::SampleDatabase& trainingDatabase,
    database::SampleDatabase& validationDatabase,
    const database::SampleDatabase& inputDatabase, size_t validationSamples)
{
    database::SampleDatabase::SampleVector samples;

    for(auto& sample : inputDatabase)
    {
        if(!validateSample(sample))
        {
            continue;
        }

        samples.push_back(sample);
    }

    std::srand(377);

    std::random_shuffle(samples.begin(), samples.end());

    validationSamples = std::min(samples.size() - 1, validationSamples);

    size_t sampleIndex = 0;

    for(; sampleIndex < validationSamples; ++sampleIndex)
    {
        copySampleToDatabase(validationDatabase, samples[sampleIndex]);
    }

    for(; sampleIndex < samples.size(); ++sampleIndex)
    {
        copySampleToDatabase(trainingDatabase, samples[sampleIndex]);
    }
}

static void splitDatabase(const std::string& outputPath, const std::string& inputFileName, size_t validationSamples)
{
    database::SampleDatabase inputDatabase(inputFileName);

    inputDatabase.load();

    createDirectories(outputPath, "training", inputDatabase.getAllPossibleLabels());
    createDirectories(outputPath, "validation", inputDatabase.getAllPossibleLabels());

    database::SampleDatabase trainingDatabase(util::joinPaths(outputPath, util::joinPaths("training", "database.txt")));
    database::SampleDatabase validationDatabase(util::joinPaths(outputPath, util::joinPaths("validation", "database.txt")));

    randomlyShuffleSamples(trainingDatabase, validationDatabase, inputDatabase, validationSamples);

    trainingDatabase.save();
    validationDatabase.save();
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

int main(int argc, char** argv)
{
	minerva::util::ArgumentParser parser(argc, argv);

	std::string inputFileName;
	std::string outputPath;
    size_t validationSamples;

	std::string loggingEnabledModules;

	bool verbose = false;

	parser.description("A tool for splitting a dataset into training/validation sets.");

	parser.parse("-i", "--input",  inputFileName,
		"", "The input database path.");
	parser.parse("-o", "--output",  outputPath,
		"", "The output path to store generated files "
			"(for visualization or feature extraction).");
	parser.parse("-S", "--validation-samples",  validationSamples,
		1000, "The number of samples to withold for validation.");

	parser.parse("-v", "--verbose", verbose, false,
		"Print out log messages during execution");
	parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
	parser.parse();

	if(verbose)
	{
		minerva::util::enableAllLogs();
	}
	else
	{
		minerva::enableSpecificLogs(loggingEnabledModules);
	}

	try
	{
        minerva::splitDatabase(outputPath, inputFileName, validationSamples);
	}
	catch(const std::exception& e)
	{
		std::cout << "Minerva Data Set Creation Failed:\n";
		std::cout << "Message: " << e.what() << "\n\n";
	}

	return 0;

}



