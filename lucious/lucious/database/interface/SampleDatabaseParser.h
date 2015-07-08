/*! \file   SampleDatabaseParser.h
	\date   Saturday December 6, 2013
	\author Gregory Diamos <solusstutus@gmail.com>
	\brief  The header file for the SampleDatabaseParser class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace lucious { namespace database { class SampleDatabase; } }

namespace lucious
{

namespace database
{

/*! \brief A class for representing a database of sample data */
class SampleDatabaseParser
{
public:
	SampleDatabaseParser(SampleDatabase* database);

public:
	void parse();

private:
	SampleDatabase* _database;

};


}

}

