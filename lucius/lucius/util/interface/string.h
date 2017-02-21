/*! \file string.h
    \date Friday February 13, 2009
    \author Gregory Diamos <gregory.diamos@gatech.edu>
    \brief Function headers for common string manipulations
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>
#include <set>

namespace lucius
{
namespace util
{

/*! \brief A vector of strings */
typedef std::vector<std::string> StringVector;

/*! \brief A set of strings */
typedef std::set<std::string> StringSet;

/*! \brief Safe string copy

    \param destination The target string
    \param source The source string
    \param max The max number of characters to copy
*/
void strlcpy(char* destination, const char* source, unsigned int max);

/*! \brief Split a string into substrings divided on a delimiter */
StringVector split(const std::string& string,
    const std::string& delimiter = " ");

/*! \brief Removing leading and trailing whitespace */
std::string removeWhitespace(const std::string& string);

/*! \brief Strip out substrings in a string */
std::string strip(const std::string& string,
    const std::string& delimiter = " ");

/*! \brief Format a string to fit a specific character width */
std::string format(const std::string& input,
    const std::string& firstPrefix = "", const std::string& prefix = "",
    unsigned int width = 80);

/*! \brief Parse a string specifying a binary number, return the number */
long long unsigned int binaryToUint(const std::string&);

/*! \brief Convert a string to a label that can be parsed by graphviz */
std::string toGraphVizParsableLabel(const std::string&);

/*! \brief Add line numbers to a very large string */
std::string addLineNumbers(const std::string&, unsigned int begin = 1);

/*! \brief Convert a raw data stream into a hex representation */
std::string dataToString(const void* data, unsigned int bytes);

/*! \brief Join a set of strings together. */
std::string join(const StringVector& strings, const std::string& delimiter);

/*! \brief Convert an array of integers into a string representation. */
std::string toString(const std::vector<size_t>& indices);

/*! \brief Convert a string into graphemes from a dictionary */
StringVector toGraphemes(const std::string& label, const StringSet& graphemeDictionary,
    bool ignoreMissing);

} // namespace util
} // namespace lucius

