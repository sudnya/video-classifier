#pragma once

// Forward Declarations
namespace lucious { namespace matrix { class Matrix; } }

namespace lucious
{
namespace matrix
{

Matrix transpose(const Matrix& input);
void transpose(Matrix& output, const Matrix& input);

}
}



