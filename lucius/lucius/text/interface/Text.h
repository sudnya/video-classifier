/*! \file   Text.h
    \date   September 13, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Text class.
*/

#pragma once

// Standard Library Includes
#include <string>

namespace lucius
{

namespace text
{

class Text
{
public:
    static bool isPathATextFile(const std::string& path);
};

}

}

