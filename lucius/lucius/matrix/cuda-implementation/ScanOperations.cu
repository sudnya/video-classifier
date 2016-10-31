
// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace matrix
{

void inclusiveScan(Matrix& output, const Matrix& input, const Dimension& dimensionsToReduce,
    const Operation& op);
Matrix inclusiveScan(const Matrix& input, const Dimension& dimensionsToReduce,
    const Operation& op);

void exclusiveScan(Matrix& output, const Matrix& input, const Dimension& dimensionsToReduce,
    const Operation& op);
Matrix exclusiveScan(const Matrix& input, const Dimension& dimensionsToReduce,
    const Operation& op);

}
}


