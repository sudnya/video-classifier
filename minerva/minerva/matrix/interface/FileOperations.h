#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{
namespace matrix
{

void save(const std::string& path, const Matrix& input);

Matrix load(const std::string& path);

}
}



