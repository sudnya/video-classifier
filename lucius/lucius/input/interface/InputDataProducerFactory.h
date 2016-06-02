/*  \file   InputDataProducer.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the InputDataProducer class.
*/

#pragma once

// Standard Library Includes
#include <memory>
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
    static std::unique_ptr<InputDataProducer> create(const std::string& producerName,
        const std::string& databaseName);
    static std::unique_ptr<InputDataProducer> create();
    static std::unique_ptr<InputDataProducer> createForDatabase(const std::string& databaseName);

};

}

}

