/*! \file   Sample.h
    \date   Saturday December 6, 2013
    \author Gregory Diamos <solusstutus@gmail.com>
    \brief  The header file for the Sample class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

namespace lucius
{

namespace database
{

/*! \brief A class for a sample */
class Sample
{
public:
    typedef std::vector<std::string> StringVector;

public:
    Sample(const std::string& path = "",
        const std::string& label = "",
        size_t beginFrame = 0,
        size_t endFrame   = 0);

public:
    const std::string& path()  const;
    const std::string& label() const;

public:
    size_t beginFrame() const;
    size_t endFrame() const;

public:
    bool isVideoSample() const;
    bool isImageSample() const;
    bool isAudioSample() const;
    bool isTextSample() const;

    bool hasLabel() const;

private:
    std::string _path;
    std::string _label;

private:
    size_t _beginFrame;
    size_t _endFrame;

};

}

}

