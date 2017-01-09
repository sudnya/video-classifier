/*! \file string.cpp
    \date Friday February 13, 2009
    \author Gregory Diamos <gregory.diamos@gatech.edu>
    \brief Function sources for common C string manipulations
*/

// Lucius Includes
#include <lucius/util/interface/string.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

namespace lucius
{

namespace util
{

void strlcpy(char* dest, const char* src, unsigned int length)
{
    const char* end = src + (length - 1);
    for(; src != end; ++src, ++dest)
    {
        *dest = *src;
        if(*src == '\0')
        {
            return;
        }
    }
    *dest = '\0';
}

StringVector split(const std::string& string,
    const std::string& delimiter)
{
    size_t begin = 0;
    size_t end = 0;
    StringVector strings;

    while(end != std::string::npos)
    {
        end = string.find(delimiter, begin);

        if(end > begin)
        {
            std::string substring = string.substr(begin, end - begin);

            if(!substring.empty()) strings.push_back(substring);
        }

        begin = end + delimiter.size();
    }

    return strings;
}

static bool isWhitespace(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string removeWhitespace(const std::string& string)
{
    std::string result;

    auto begin = string.begin();

    for( ; begin != string.end(); ++begin)
    {
        if(!isWhitespace(*begin)) break;
    }

    auto end = string.end();

    if(end != begin)
    {
        --end;

        for( ; end != begin; --end)
        {
            if(!isWhitespace(*end))
            {
                break;
            }
        }

        ++end;
    }

    return std::string(begin, end);
}

std::string strip(const std::string& string, const std::string& delimiter)
{
    std::string result;
    size_t begin = 0;
    size_t end = 0;

    while(end != std::string::npos)
    {
        end = string.find(delimiter, begin);
        result += string.substr(begin, end - begin);
        begin = end + delimiter.size();
    }

    return result;
}

std::string format(const std::string& input,
    const std::string& firstPrefix, const std::string& prefix,
    unsigned int width)
{
    std::string word;
    std::string result = firstPrefix;
    unsigned int currentIndex = firstPrefix.size();

    for(std::string::const_iterator fi = input.begin();
        fi != input.end(); ++fi)
    {
        if(*fi == ' ' || *fi == '\t' || *fi == '\n'
            || *fi == '\r' || *fi == '\f')
        {
            if(currentIndex + word.size() > width)
            {
                currentIndex = prefix.size();
                result += "\n";
                result += prefix;
            }

            if(!word.empty())
            {
                result += word + " ";
                ++currentIndex;
                word.clear();
            }
        }
        else
        {
            word.push_back(*fi);
            ++currentIndex;
        }
    }

    if(currentIndex + word.size() > width)
    {
        result += "\n";
        result += prefix;
    }

    result += word + "\n";
    return result;
}

long long unsigned int binaryToUint(const std::string& string)
{
    long long unsigned int result = 0;
    assert(string.size() > 2);

    std::string::const_iterator ci = string.begin();
    assert(*ci == '0');
    ++ci;
    assert(*ci == 'b');
    ++ci;

    long long unsigned int mask = 1;

    for(; ci != string.end(); ++ci)
    {
        assert(*ci == '0' || *ci == '1');

        result |= mask & (*ci == '1');
        mask <<= 1;
    }

    return result;
}

std::string toGraphVizParsableLabel(const std::string& string)
{
    std::string result;
    for(std::string::const_iterator fi = string.begin();
        fi != string.end(); ++fi)
    {
        if(*fi == '{')
        {
            result.push_back('[');
        }
        else if(*fi == '}')
        {
            result.push_back(']');
        }
        else if(*fi == '|')
        {
            result.push_back('/');
        }
        else
        {
            result.push_back(*fi);
        }
    }
    return result;
}

std::string addLineNumbers(const std::string& string, unsigned int line)
{
    std::stringstream result;

    result << line++ << " ";

    for(std::string::const_iterator s = string.begin();
        s != string.end(); ++s)
    {
        if(*s == '\n')
        {
            result << "\n" << line++ << " ";
        }
        else
        {
            result << *s;
        }
    }
    return result.str();
}

std::string dataToString(const void* data, unsigned int size)
{
    std::stringstream stream;

    while(size > 0)
    {
        stream << "0x";
        stream.width(2);
        stream.fill('0');
        stream << std::hex << (int)*((unsigned char*) data) << " ";
        size--;
        data = ((char*)data + 1);
    }

    return stream.str();
}

std::string join(const StringVector& strings, const std::string& delimiter)
{
    auto string = strings.begin();

    std::string result;

    if(string != strings.end())
    {
        result += *string;
        ++string;
    }

    for(; string != strings.end(); ++string)
    {
        result += delimiter + *string;
    }

    return result;
}

std::string toString(const std::vector<size_t>& indices)
{
    std::stringstream stream;

    stream << "[";

    for(auto& index : indices)
    {
        stream << " " << index;
    }

    stream << " ]";

    return stream.str();
}

StringVector toGraphemes(const std::string& label, const StringSet& graphemeDictionary,
    bool ignoreMissingGraphemes)
{
    auto remainingLabel = label;

    StringVector graphemes;

    while(!remainingLabel.empty())
    {
        auto insertPosition = graphemeDictionary.lower_bound(remainingLabel);

        // exact match
        if(insertPosition != graphemeDictionary.end() && *insertPosition == remainingLabel)
        {
            graphemes.push_back(remainingLabel);
            break;
        }

        // ordered before first grapheme
        if(insertPosition == graphemeDictionary.begin())
        {
            if(ignoreMissingGraphemes)
            {
                remainingLabel = remainingLabel.substr(1);
                continue;
            }
            else
            {
                throw std::runtime_error("Could not match remaining label '" + remainingLabel +
                    "' against a grapheme.");
            }
        }

        --insertPosition;

        auto grapheme = remainingLabel.substr(0, insertPosition->size());

        if(grapheme != *insertPosition)
        {
            if(ignoreMissingGraphemes)
            {
                remainingLabel = remainingLabel.substr(1);
                continue;
            }
            else
            {
                throw std::runtime_error("Could not match remaining label '" + remainingLabel +
                    "' against best grapheme '" + *insertPosition + "'.");
            }
        }

        graphemes.push_back(grapheme);

        remainingLabel = remainingLabel.substr(grapheme.size());
    }

    return graphemes;

}

}

}

