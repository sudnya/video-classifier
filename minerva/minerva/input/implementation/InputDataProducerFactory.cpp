/*	\file   InputDataProducer.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the InputDataProducer class.
*/

// Minerva Includes
#include <minerva/input/interface/InputDataProducerFactory.h>

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace input{ class InputDataProducer; } }

namespace minerva
{

namespace input
{

/*! \brief A factory for classifier engines */
class InputDataProducerFactory
{
public:
	static InputDataProducer* create(const std::string& classifierName);
	static InputDataProducer* createForDatabase(const std::string& classifierName);

};

}

}


