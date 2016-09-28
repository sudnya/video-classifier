/*  \file   lucius-make-dataset.cpp
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  A tool for splitting a dataset into training/validation sets.
*/

// Lucius Includes
#include <lucius/database/interface/SampleDatabase.h>
#include <lucius/database/interface/Sample.h>

#include <lucius/audio/interface/Audio.h>

#include <lucius/video/interface/Image.h>

#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/paths.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>
#include <fstream>
#include <memory>
#include <algorithm>

namespace lucius
{

static void createDirectories(const std::string& outputPath, const std::string& setName,
    const util::StringVector& labels)
{
    for(auto& label : labels)
    {
        util::makeDirectory(util::joinPaths(outputPath, util::joinPaths(setName, label)));
    }
}

static void copySampleToDatabase(database::SampleDatabase& outputDatabase,
    const database::Sample& sample, bool groupByLabel)
{
    auto directory = util::getDirectory(outputDatabase.path());

    std::string path;

    if(groupByLabel)
    {
        path = sample.label();
    }

    path = util::joinPaths(path, util::getFile(sample.path()));

    auto completePath = util::joinPaths(directory, path);

    util::log("LuciusMakeDataset") << "Copying sample '" + sample.path() + "' to '" +
        completePath + "'\n";

    util::copyFile(completePath, sample.path());

    outputDatabase.addSample(database::Sample(path, sample.label()));
}

static bool validateSample(const database::Sample& sample)
{
    if(sample.isImageSample() && video::Image::isPathAnImageFile(sample.path()))
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
    else if(sample.isAudioSample())
    {
        audio::Audio waveform(sample.path());

        try
        {
            waveform.cache();
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
    const database::SampleDatabase& inputDatabase,
    size_t trainingSamples, size_t validationSamples,
    bool groupByLabel)
{
    database::SampleDatabase::SampleVector samples;

    for(auto& sample : inputDatabase)
    {
        samples.push_back(sample);
    }

    std::srand(377);

    std::random_shuffle(samples.begin(), samples.end());

    validationSamples = std::min(samples.size() - 1, validationSamples);

    size_t sampleIndex = 0;
    size_t nextSample  = 0;

    for(; sampleIndex < validationSamples && nextSample < samples.size(); ++nextSample)
    {
        if(!validateSample(samples[nextSample]))
        {
            continue;
        }

        copySampleToDatabase(validationDatabase, samples[nextSample], groupByLabel);
        ++sampleIndex;
    }

    trainingSamples = std::min(trainingSamples, samples.size() - validationSamples);

    size_t totalSamples = validationSamples + trainingSamples;

    for(; sampleIndex < totalSamples && nextSample < samples.size(); ++nextSample)
    {
        if(!validateSample(samples[nextSample]))
        {
            continue;
        }

        copySampleToDatabase(trainingDatabase, samples[nextSample], groupByLabel);

        ++sampleIndex;
    }
}

static void splitDatabase(const std::string& outputPath, const std::string& inputFileName,
    size_t trainingSamples, size_t validationSamples, bool groupByLabel)
{
    database::SampleDatabase inputDatabase(inputFileName);

    inputDatabase.load();

    if(groupByLabel)
    {
        createDirectories(outputPath, "training", inputDatabase.getAllPossibleLabels());
        createDirectories(outputPath, "validation", inputDatabase.getAllPossibleLabels());
    }
    else
    {
        util::makeDirectory(util::joinPaths(outputPath, "training"));
        util::makeDirectory(util::joinPaths(outputPath, "validation"));
    }

    database::SampleDatabase trainingDatabase(util::joinPaths(outputPath,
        util::joinPaths("training", "database.txt")));
    database::SampleDatabase validationDatabase(util::joinPaths(outputPath,
        util::joinPaths("validation", "database.txt")));

    randomlyShuffleSamples(trainingDatabase, validationDatabase, inputDatabase,
        trainingSamples, validationSamples, groupByLabel);

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
    lucius::util::ArgumentParser parser(argc, argv);

    std::string inputFileName;
    std::string outputPath;
    size_t validationSamples = 0;
    size_t trainingSamples = 0;
    bool groupByLabel = false;

    std::string loggingEnabledModules;

    bool verbose = false;

    parser.description("A tool for splitting a dataset into training/validation sets.");

    parser.parse("-i", "--input", inputFileName, "", "The input database path.");
    parser.parse("-o", "--output", outputPath, "", "The output path to store generated files "
            "(the training/validation datasets).");
    parser.parse("-S", "--validation-samples", validationSamples,
        1000, "The number of samples to withold for validation.");
    parser.parse("-T", "--training-samples", trainingSamples, 1e9,
        "The maximum number of samples to use for validation.");
    parser.parse("-g", "--group-by-label", groupByLabel,
        false, "Group samples together by labels.");

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");
    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse();

    if(verbose)
    {
        lucius::util::enableAllLogs();
    }
    else
    {
        lucius::enableSpecificLogs(loggingEnabledModules);
    }

    try
    {
        lucius::splitDatabase(outputPath, inputFileName, trainingSamples, validationSamples,
            groupByLabel);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Data Set Creation Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;

}



