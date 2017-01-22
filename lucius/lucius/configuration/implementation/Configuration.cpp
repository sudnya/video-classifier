/*  \file   Configuration.cpp
    \date   Saturday December 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Configuration class.
*/

// Lucius Includes
#include <lucius/configuration/interface/Configuration.h>

#include <lucius/util/interface/PropertyTree.h>
#include <lucius/util/interface/paths.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <fstream>

namespace lucius
{

namespace configuration
{

Configuration::Configuration(std::unique_ptr<PropertyTree>&& properties)
: _properties(std::move(properties))
{

}

Configuration::Configuration(const Configuration& c)
: _properties(std::make_unique<PropertyTree>(*c._properties))
{

}

Configuration::~Configuration()
{

}

Configuration& Configuration::operator=(const Configuration& c)
{
    if(&c == this)
    {
        return *this;
    }

    *_properties = *c._properties;

    return *this;
}

std::string Configuration::getSampleStatisticsEngineName() const
{
    return (*_properties)["model"].get<std::string>("sample-statistics-engine",
        "SampleStatisticsEngine");
}

std::string Configuration::getLearnerEngineName() const
{
    return (*_properties)["model"].get<std::string>("learner-engine", "LearnerEngine");
}

std::string Configuration::getClassifierEngineName() const
{
    return (*_properties)["model"].get<std::string>("classifier-engine", "ClassifierEngine");
}

size_t Configuration::getBatchSize() const
{
    if(!(*_properties)["optimization"].exists("batch-size"))
    {
        throw std::runtime_error("Configuration must include an optimization.batch-size field.");
    }

    return (*_properties)["optimization"].get<size_t>("batch-size");
}

size_t Configuration::getMaximumSamples() const
{
    if(!(*_properties)["optimization"].exists("maximum-samples"))
    {
        throw std::runtime_error(
            "Configuration must include an optimization.maximum-samples field.");
    }

    return (*_properties)["optimization"].get<double>("maximum-samples");
}

size_t Configuration::getMaximumStandardizationSamples() const
{
    return _properties->get<double>("optimization.standardization-samples", 1024.0);
}

size_t Configuration::getMaximumValidationSamples() const
{
    return _properties->get<double>("optimization.validation-samples", 1024.0);
}

size_t Configuration::getEpochs() const
{
    if(!(*_properties)["optimization"].exists("epochs"))
    {
        throw std::runtime_error("Configuration must include an optimization.epochs field.");
    }

    return (*_properties)["optimization"].get<size_t>("epochs");
}

size_t Configuration::getPassesPerEpoch() const
{
    return (*_properties)["optimization"].get<size_t>("passes-per-epoch", 1);
}

bool Configuration::getShouldSeed() const
{
    return (*_properties)["optimization"].get<size_t>("should-seed", 0);
}

double Configuration::getRequiredAccuracy() const
{
    if(!(*_properties)["dataset"].exists("epochs"))
    {
        throw std::runtime_error("Configuration must include a dataset.required-accuracy field.");
    }

    return (*_properties)["dataset"].get<double>("required-accuracy");
}

std::string Configuration::getTrainingPath() const
{
    if(!(*_properties)["dataset"].exists("training-dataset-path"))
    {
        throw std::runtime_error("Configuration must include a "
            "dataset.training-dataset-path field.");
    }

    return (*_properties)["dataset"].get<std::string>("training-dataset-path");
}

std::string Configuration::getValidationPath() const
{
    if(!(*_properties)["dataset"].exists("validation-dataset-path"))
    {
        throw std::runtime_error("Configuration must include a "
            "dataset.validation-dataset-path field.");
    }

    return (*_properties)["dataset"].get<std::string>("validation-dataset-path");
}

std::string Configuration::getOutputPath() const
{
    if(!(*_properties)["checkpointing"].exists("base-directory"))
    {
        throw std::runtime_error("Configuration must include a "
            "checkpointing.base-directory field.");
    }

    return _properties->get<std::string>("checkpointing.base-directory");
}

bool Configuration::getIsLogFileEnabled() const
{
    return _properties->get<bool>("checkpointing.use-log-file", false);
}

std::string Configuration::getLoggingEnabledModules() const
{
    std::stringstream modules;

    if(!_properties->exists("logging.enabled-loggers"))
    {
        return modules.str();
    }

    bool first = true;

    for(auto& log : _properties->get("logging.enabled-loggers"))
    {
        if(!first) modules << ",";
        first = false;

        modules << log.key();
    }

    return modules.str();
}

std::string Configuration::getLogPath() const
{
    if(!_properties->exists("checkpointing.use-log-file"))
    {
        return "";
    }

    return _properties->get<std::string>("checkpointing.log-path",
        _properties->get<std::string>("checkpointing.base-directory") + "/log");
}

std::string Configuration::getTrainingReportPath() const
{
    return _properties->get<std::string>("checkpointing.training-report-path",
        _properties->get<std::string>("checkpointing.base-directory") + "/training-error.csv");
}

std::string Configuration::getValidationReportPath() const
{
    return _properties->get<std::string>("checkpointing.validation-report-path",
        _properties->get<std::string>("checkpointing.base-directory") + "/validation-error.csv");
}

std::string Configuration::getModelSavePath() const
{
    return (*_properties)["checkpointing"].get<std::string>("base-directory") + "/model.tar";
}

std::string Configuration::getModelSpecification() const
{
    auto model = (*_properties)["model"];

    model.setKey("");

    return model.jsonString();
}

Configuration::AttributeList Configuration::getAllAttributes() const
{
    AttributeList attributes;

    for (auto& parameter : (*_properties)["optimization"]["parameters"])
    {
        attributes.push_back(std::make_pair(parameter.key(), parameter.value()));
    }

    for (auto& parameter : (*_properties)["attributes"])
    {
        attributes.push_back(std::make_pair(parameter.key(), parameter.value()));
    }

    return attributes;
}

bool Configuration::isCudaEnabled() const
{
    return _properties->get<bool>("system.use-cuda");
}

std::string Configuration::getCudaDevice() const
{
    return _properties->get("system.cuda-device");
}

std::string Configuration::getPrecision() const
{
    return _properties->get("system.precision");
}

Configuration Configuration::create(const std::string& path)
{
    std::ifstream file(path);

    if(!file.is_open())
    {
        throw std::runtime_error("Failed to open path '" + path + "'.");
    }

    auto properties = util::PropertyTree::loadJson(file);

    util::log("Configuration") << "Loading configuration data from the following properties:\n"
        << properties.jsonString() << "\n";

    return Configuration(std::make_unique<PropertyTree>(properties));
}

}

}




