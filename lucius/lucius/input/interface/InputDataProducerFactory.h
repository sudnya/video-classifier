/*    \file   InputDataProducer.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the InputDataProducer class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace input{ class InputDataProducer; } }

namespace lucius
{

namespace input
{

/*! \brief A factory for classifier engines */
class InputDataProducerFactory
{
public:
    static InputDataProducer* create(const std::string& producerName, const std::string& databaseName);
    static InputDataProducer* create();
    static InputDataProducer* createForDatabase(const std::string& databaseName);

};

}

}

