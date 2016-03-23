/*! \file   SampleDatabase.h
    \date   Saturday December 6, 2013
    \author Gregory Diamos <solusstutus@gmail.com>
    \brief  The header file for the SampleDatabase class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>
#include <set>

// Forward Declarations
namespace lucius { namespace database { class Sample; } }

namespace lucius
{

namespace database
{

/*! \brief A class for representing a database of sample data */
class SampleDatabase
{
public:
    typedef std::vector<std::string> StringVector;
    typedef std::vector<Sample> SampleVector;

    typedef SampleVector::iterator iterator;
    typedef SampleVector::const_iterator const_iterator;

public:
    SampleDatabase();
    SampleDatabase(const std::string& path);
    ~SampleDatabase();

public:
    StringVector getAllPossibleLabels() const;
    StringVector getGraphemes() const;
    const std::string& getDefaultGrapheme() const;
    const std::string& getDelimiterGrapheme() const;
    size_t getTotalLabelCount() const;

public:
    iterator begin();
    const_iterator begin() const;

    iterator end();
    const_iterator end() const;

public:
    size_t size()  const;
    bool   empty() const;

public:
    const std::string& path() const;

public:
    bool containsVideoSamples() const;
    bool containsImageSamples() const;
    bool containsAudioSamples() const;
    bool containsTextSamples() const;

public:
    void addSample(const Sample& sample);
    void addLabel(const std::string& label);
    void addGrapheme(const std::string& grapheme);
    void setDefaultGrapheme(const std::string& grapheme);
    void setDelimiterGrapheme(const std::string& grapheme);

public:
    void save() const;

public:
    void load();

private:
    typedef std::set<std::string> StringSet;

private:
    std::string _path;

private:
    SampleVector _samples;
    StringSet    _labels;
    StringSet    _graphemes;
    std::string  _defaultGrapheme;
    std::string  _delimiterGrapheme;

};


}

}




