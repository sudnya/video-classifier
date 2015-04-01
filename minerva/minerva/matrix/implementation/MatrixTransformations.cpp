
// Minerva Includes
#include <minerva/matrix/interface/MatrixTransformations.h>

namespace minerva
{
namespace matrix
{

Dimension linearStride(const Dimension& size)
{
	Dimension stride;
	
	size_t step = 1;
	
	for (auto sizeStep : size)
	{
		stride.push_back(step);
		step *= sizeStep;
	}
	
	return stride;
}

}
}	


