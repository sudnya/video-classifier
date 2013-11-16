/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The implementation of the layer class 
 */

#include <minerva/neuralnetwork/interface/Layer.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

#include <random>
#include <cstdlib>

namespace minerva
{
namespace neuralnetwork
{

typedef minerva::matrix::Matrix::FloatVector FloatVector;

Layer::Layer(unsigned totalBlocks, size_t blockInput, size_t blockOutput)
{
    m_sparseMatrix.resize(totalBlocks, blockInput + 1, blockOutput);
}

void Layer::initializeRandomly(float e)
{
	float epsilon = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", e);

	epsilon = (e / std::sqrt(getInputCount() + 1));
	
	m_sparseMatrix.assignUniformRandomValues(-epsilon, epsilon);
}

Layer::BlockSparseMatrix Layer::runInputs(const BlockSparseMatrix& m) const
{
    util::log("Layer") << " Running forward propagation on matrix (" << m.rows()
            << " rows, " << m.columns() << " columns).\n";
	
	auto output = m_sparseMatrix.multiply(m).sigmoid();
    util::log("Layer") << "  output: " << output.toString() << "\n";
    
    util::log("Layer") << "  layer output is a matrix (" << output.rows()
            << " rows, " << output.columns() << " columns).\n";
    
    return output;
}

Layer::BlockSparseMatrix Layer::runReverse(const BlockSparseMatrix& m) const
{
    util::log("Layer") << " Running reverse propagation on matrix (" << m.rows()
            << " rows, " << m.columns() << " columns).\n";
   
	auto result = m.multiply(m_sparseMatrix.transpose());

    // drop the final row from each slice corresponding to the bias
	for(auto& block : result)
	{	
		block = block.slice(0, 0, block.rows(), block.columns() - 1); 
	}

    util::log("Layer") << "  layer output is a matrix (" << result.rows()
            << " rows, " << result.columns() << " columns).\n";

    return result;
}

void Layer::transpose()
{
	m_sparseMatrix.transposeSelf();
}

unsigned Layer::getInputCount() const
{
    if(empty())
        return 0;

	return m_sparseMatrix.rows() - 1;
}

unsigned Layer::getOutputCount() const
{
	return m_sparseMatrix.columns();
}

Layer::BlockSparseMatrix Layer::getWeightsWithoutBias() const
{
	BlockSparseMatrix result;

	for(auto& matrix : *this)
	{
		result.push_back(matrix.slice(0, 0, matrix.rows(), matrix.columns() - 1));
	}
	
	return result;
}

size_t Layer::totalWeights() const
{
	return m_sparseMatrix.size() - m_sparseMatrix.columns();
}

Layer::Matrix Layer::getFlattenedWeights() const
{
	Matrix weights;
	
	auto matrix = begin();
	
	// Discard the bias weights
	weights = weights.appendColumns(
		Matrix(1, matrix->size() - matrix->columns(),
		matrix->slice(0, 0, matrix->rows() - 1,
		matrix->columns()).data()));
	
	for(++matrix; matrix != end(); ++matrix)
	{
		weights = weights.appendColumns(Matrix(1, matrix->size(),
			matrix->data()));
	}
	
	return weights;
}

void Layer::setFlattenedWeights(const Matrix& m)
{
	size_t position = 0;
	
	auto matrix = begin();

	// Add the bias weights back in
	auto updatedWeights =
		m.slice(0, 0, 1, matrix->size() - matrix->columns()).appendColumns(
		matrix->slice(0, 0, 1, matrix->columns()));
	
	matrix->data() = updatedWeights.data();
	
	position += matrix->size() - matrix->columns();
	
	for(++matrix; matrix != end(); ++matrix)
	{
		matrix->data() = m.slice(0, position, 1,
			position + matrix->size()).data();
	
		position += matrix->size();
	}
}

Layer::iterator Layer::begin()
{
	return m_sparseMatrix.begin();
}

Layer::const_iterator Layer::begin() const
{
	return m_sparseMatrix.begin();
}

Layer::iterator Layer::end()
{
	return m_sparseMatrix.end();
}

Layer::const_iterator Layer::end() const
{
	return m_sparseMatrix.end();
}

Layer::Matrix& Layer::operator[](size_t index)
{
	return m_sparseMatrix[index];
}

const Layer::Matrix& Layer::operator[](size_t index) const
{
	return m_sparseMatrix[index];
}

Layer::Matrix& Layer::back()
{
	return m_sparseMatrix.back();
}

const Layer::Matrix& Layer::back() const
{
	return m_sparseMatrix.back();
}

void Layer::push_back(const Matrix& m)
{
	m_sparseMatrix.push_back(m);
}

size_t Layer::size() const
{
	return m_sparseMatrix.size();
}

size_t Layer::blocks() const
{
	return m_sparseMatrix.blocks();
}

bool Layer::empty() const
{
	return m_sparseMatrix.empty();
}

}

}


