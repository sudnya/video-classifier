/*  \file   Configuration.h
    \date   Saturday December 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Configuration class.
*/

#pragma once

// Lucius Includes
#include <lucius/util/interface/memory.h>

// Standard Library Includes
#include <string>
#include <memory>
#include <vector>

// Forward Declarations
namespace lucius { namespace util { class PropertyTree; } }

namespace lucius
{

namespace configuration
{

/*! \brief A representation of experiment configuration data */
class Configuration
{
public:
    typedef util::PropertyTree PropertyTree;
    typedef std::vector<std::pair<std::string, std::string>> AttributeList;

public:
    Configuration(std::unique_ptr<PropertyTree>&& properties);
    Configuration(const Configuration& c);
    ~Configuration();

public:
    Configuration& operator=(const Configuration& );

public:
    std::string getSampleStatisticsEngineName() const;
    std::string getLearnerEngineName() const;
    std::string getClassifierEngineName() const;

public:
    size_t getBatchSize() const;
    size_t getMaximumSamples() const;
    size_t getMaximumStandardizationSamples() const;
    size_t getMaximumValidationSamples() const;
    size_t getEpochs() const;
    size_t getPassesPerEpoch() const;

public:
    bool getShouldSeed() const;

public:
    double getRequiredAccuracy() const;

public:
    std::string getTrainingPath() const;
    std::string getValidationPath() const;
    std::string getOutputPath() const;

public:
    bool getIsLogFileEnabled() const;
    std::string getLogPath() const;
    std::string getLoggingEnabledModules() const;

public:
    std::string getTrainingReportPath() const;
    std::string getValidationReportPath() const;

public:
    std::string getModelSpecification() const;
    std::string getModelSavePath() const;

public:
    AttributeList getAllAttributes() const;

public:
    bool isCudaEnabled() const;
    std::string getCudaDevice() const;
    std::string getPrecision() const;

public:
    static Configuration create(const std::string& path);

private:
    std::unique_ptr<PropertyTree> _properties;

};

}

}



