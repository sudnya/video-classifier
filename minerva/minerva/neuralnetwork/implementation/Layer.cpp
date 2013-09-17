/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The implementation of the layer class 
 */

#include <minerva/neuralnetwork/interface/Layer.h>

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
    m_sparseMatrix.resize(totalBlocks);
    for (auto i = m_sparseMatrix.begin(); i != m_sparseMatrix.end(); ++i)
    {
		// The +1 is for the bias layer
        (*i).resize(blockInput + 1, blockOutput);
    }
}

void Layer::initializeRandomly(float e)
{
	float epsilon = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", e);

	epsilon = (e / std::sqrt(getInputCount() + 1));

    std::default_random_engine generator(std::time(0));
    std::uniform_real_distribution<float> distribution(-epsilon, epsilon);
    
    for (auto i = begin(); i != end(); ++i)
    {
        for (auto j = i->begin(); j != i->end(); ++j)
        {
            float randomNumber = distribution(generator);
            *j = randomNumber;
        }
    }
}

Layer::Matrix Layer::runInputs(const Matrix& m) const
{
    util::log("Layer") << " Running forward propagation on matrix (" << m.rows()
            << " rows, " << m.columns() << " columns).\n";
    
    unsigned int inputPixPos = 0;
    Matrix finalOutput;
    
    // sparse multiply
    // slice input Matrix into chunks multipliable to matrix blocks
    for (auto i = m_sparseMatrix.begin(); i != m_sparseMatrix.end(); ++i)
    {
    	// Extract the input
        Matrix temp = m.slice(inputPixPos, 0, m.rows(), (*i).rows() - 1);

		// add the bias
		temp = temp.appendColumns(Matrix(temp.rows(), 1,
			FloatVector(temp.columns(), 1.0f)));

		// The -1 corrects for the bias, which does not exist in the input
        inputPixPos += (*i).rows() - 1;

        Matrix output = temp.multiply((*i)).sigmoid();
        util::log("Layer") << "  output: " << output.toString() << "\n";
       
        finalOutput = finalOutput.appendColumns(output);
    }
    
    util::log("Layer") << "  layer output is a matrix (" << finalOutput.rows()
            << " rows, " << finalOutput.columns() << " columns).\n";
    
    return finalOutput;
}

Layer::Matrix Layer::runReverse(const Matrix& m) const
{
    util::log("Layer") << " Running reverse propagation on matrix (" << m.rows()
            << " rows, " << m.columns() << " columns).\n";
    
    unsigned int inputPixPos = 0;
    Matrix finalOutput;
    // sparse multiply
    // slice input Matrix into chunks multipliable to matrix blocks
    for (auto i = m_sparseMatrix.begin(); i != m_sparseMatrix.end(); ++i)
    {
    	auto sparseMatrixT = i->transpose();
    
        Matrix temp = m.slice(inputPixPos, 0, m.rows(), sparseMatrixT.rows());
        inputPixPos += sparseMatrixT.rows();
        Matrix output = temp.multiply(sparseMatrixT);
        
        // drop the final row corresponding to the bias
        output = output.slice(0, 0, output.rows(), output.columns() - 1);
        
        util::log("Layer") << "  output: " << output.toString() << "\n";
        finalOutput = finalOutput.appendColumns(output);
    }
    
    util::log("Layer") << "  layer output is a matrix (" << finalOutput.rows()
            << " rows, " << finalOutput.columns() << " columns).\n";
    
    return finalOutput;
}

void Layer::transpose()
{
	for(auto& matrix : *this)
	{
		matrix.transposeSelf();
	}
}

unsigned Layer::getInputCount() const
{
    if(empty())
        return 0;

    unsigned count = 0;
    
    for(auto& matrix : *this)
    {
        count += matrix.rows();
    }

    return count - 1;
}

unsigned Layer::getOutputCount() const
{
    if(empty())
        return 0;

    unsigned count = 0;
    
    for(auto& matrix : *this)
    {
        count += matrix.columns();
    }

    return count;
}

size_t Layer::totalWeights() const
{
	size_t weights = 0;

	for(auto& matrix : *this)
	{
		weights += matrix.size();
	}
	
	return weights - back().columns();
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
	
	matrix->setDataRowMajor(updatedWeights.data());
	
	position += matrix->size() - matrix->columns();
	
	for(++matrix; matrix != end(); ++matrix)
	{
		matrix->setDataRowMajor(m.slice(0, position, 1,
			position + matrix->size()).data());
	
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

unsigned int Layer::size() const
{
	return m_sparseMatrix.size();
}

bool Layer::empty() const
{
	return m_sparseMatrix.empty();
}

}

}


