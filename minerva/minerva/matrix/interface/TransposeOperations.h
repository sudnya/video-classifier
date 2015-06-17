#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{
namespace matrix
{

Matrix transpose(const Matrix& input);
void transpose(Matrix& output, const Matrix& input);

}
}



