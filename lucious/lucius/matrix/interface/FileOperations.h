#pragma once

// Standard Library Includes
#include <string>
#include <istream>
#include <ostream>

// Forward Declarations
namespace lucious { namespace matrix { class Matrix;           } }
namespace lucious { namespace util   { class InputTarArchive;  } }
namespace lucious { namespace util   { class OutputTarArchive; } }

namespace lucious
{
namespace matrix
{

void save(std::ostream& file, const Matrix& input);
void save(const std::string& path, const Matrix& input);

Matrix load(std::istream& file);
Matrix load(const std::string& path);

void saveToArchive(util::OutputTarArchive& archive, const std::string& path, const Matrix& input);
Matrix loadFromArchive(util::InputTarArchive& archive, const std::string& path);

}
}



