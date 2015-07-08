#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucious { namespace matrix { class Matrix; } }

namespace lucious
{
namespace matrix
{

void save(std::ostream& file, const Matrix& input);
void save(const std::string& path, const Matrix& input);

Matrix load(std::istream& file);
Matrix load(const std::string& path);

}
}



