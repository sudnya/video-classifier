/*! \file   Text.cpp
    \date   September 13, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Text class.
*/

// Lucius Includes
#include <lucius/text/interface/Text.h>

#include <lucius/util/interface/paths.h>


namespace lucius
{

namespace text
{

bool Text::isPathATextFile(const std::string& path)
{
    auto extension = util::getExtension(path);

    return extension == ".txt";
}

}

}

