/*! \file Units.h
    \date Friday February 13, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief Function headers for common unit to text conversions.
*/

// Lucius Includes
#include <lucius/util/interface/Units.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{

namespace util
{

static void getStandardUnitValueAndExtension(double& value, std::string& extension)
{
    if(value < 1.0e3)
    {
        extension = "";
    }
    else if(value < 1.0e6)
    {
        value /= 1.0e3;
        extension = "K";
    }
    else if(value < 1.0e9)
    {
        value /= 1.0e6;
        extension = "K";
    }
    else if(value < 1.0e12)
    {
        value /= 1.0e9;
        extension = "G";
    }
    else if(value < 1.0e15)
    {
        value /= 1.0e12;
        extension = "T";
    }
    else if(value < 1.0e18)
    {
        value /= 1.0e15;
        extension = "P";
    }
    else if(value < 1.0e21)
    {
        value /= 1.0e18;
        extension = "E";
    }
    else
    {
        value /= 1.0e21;
        extension = "Z";
    }

}

std::string flopsString(double flops)
{
    std::string extension;

    getStandardUnitValueAndExtension(flops, extension);

    std::stringstream stream;

    stream << flops << " " << extension << "FLOPS";

    return stream.str();
}

std::string byteString(double bytes)
{
    std::string extension;

    getStandardUnitValueAndExtension(bytes, extension);

    std::stringstream stream;

    stream << bytes << " " << extension << "B";

    return stream.str();

}

}

}



