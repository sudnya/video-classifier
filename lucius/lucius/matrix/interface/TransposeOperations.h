#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{
namespace matrix
{

Matrix transpose(const Matrix& input);
void transpose(Matrix& output, const Matrix& input);

}
}



