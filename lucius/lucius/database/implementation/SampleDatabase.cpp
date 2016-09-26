/*! \file   SampleDatabase.cpp
    \date   Saturday December 6, 2013
    \author Gregory Diamos <solusstutus@gmail.com>
    \brief  The source file for the SampleDatabase class.
*/

// Lucius Includes
#include <lucius/database/interface/SampleDatabase.h>
#include <lucius/database/interface/SampleDatabaseParser.h>
#include <lucius/database/interface/Sample.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>

namespace lucius
{

namespace database
{

SampleDatabase::SampleDatabase()
{
}

SampleDatabase::SampleDatabase(const std::string& path)
: _path(path)
{

}

SampleDatabase::~SampleDatabase()
{

}

SampleDatabase::StringVector SampleDatabase::getAllPossibleLabels() const
{
    if(_graphemes.empty())
    {
        return StringVector(_labels.begin(), _labels.end());
    }
    else
    {
        return getGraphemes();
    }
}

SampleDatabase::StringVector SampleDatabase::getGraphemes() const
{
    StringVector graphemes(_graphemes.begin(), _graphemes.end());

    graphemes.push_back(getDefaultGrapheme());

    return graphemes;
}

const std::string& SampleDatabase::getDefaultGrapheme() const
{
    return _defaultGrapheme;
}

size_t SampleDatabase::getTotalLabelCount() const
{
    return _labels.size();
}

SampleDatabase::iterator SampleDatabase::begin()
{
    return _samples.begin();
}

SampleDatabase::const_iterator SampleDatabase::begin() const
{
    return _samples.begin();
}

SampleDatabase::iterator SampleDatabase::end()
{
    return _samples.end();
}

SampleDatabase::const_iterator SampleDatabase::end() const
{
    return _samples.end();
}

size_t SampleDatabase::size() const
{
    return _samples.size();
}

bool SampleDatabase::empty() const
{
    return _samples.empty();
}

const std::string& SampleDatabase::path() const
{
    return _path;
}

bool SampleDatabase::containsVideoSamples() const
{
    for(auto& sample : _samples)
    {
        if(sample.isVideoSample())
        {
            return true;
        }
    }

    return false;
}

bool SampleDatabase::containsImageSamples() const
{
    for(auto& sample : _samples)
    {
        if(sample.isImageSample())
        {
            return true;
        }
    }

    return false;
}

bool SampleDatabase::containsAudioSamples() const
{
    for(auto& sample : _samples)
    {
        if(sample.isAudioSample())
        {
            return true;
        }
    }

    return false;
}

bool SampleDatabase::containsTextSamples() const
{
    return false;
}

void SampleDatabase::addSample(const Sample& sample)
{
    if(sample.hasLabel())
    {
        _labels.insert(sample.label());
    }

    return _samples.push_back(sample);
}

void SampleDatabase::addLabel(const std::string& label)
{
    _labels.insert(label);
}

void SampleDatabase::addGrapheme(const std::string& grapheme)
{
    _graphemes.insert(grapheme);
}

void SampleDatabase::setDefaultGrapheme(const std::string& grapheme)
{
    _defaultGrapheme = grapheme;
}

void SampleDatabase::save() const
{
    std::ofstream output(path());

    if(!output.is_open())
    {
        throw std::runtime_error("Failed to open database file '" + path() + "' for writing.");
    }

    for(auto& sample : *this)
    {
        auto entry = sample.path() + ", \"" + sample.label() + "\"\n";

        output.write(entry.c_str(), entry.size());

        if(!output.good())
        {
            throw std::runtime_error("Failed to save sample '" + sample.path() +
                "' to database file '" + path() + "'");
        }
    }
}

void SampleDatabase::load()
{
    SampleDatabaseParser parser(this);

    parser.parse();
}

}

}

