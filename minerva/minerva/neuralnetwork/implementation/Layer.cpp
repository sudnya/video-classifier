/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The implementation of the layer class 
 */
#include <random>

#include <minerva/util/interface/debug.h>
#include <minerva/neuralnetwork/interface/Layer.h>

namespace minerva
{
namespace neuralnetwork
{

void Layer::initializeRandomly()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    
    for (auto i = begin(); i != end(); ++i)
    {
        for (auto j = i->begin(); j != i->end(); ++j)
        {
            float randomNumber = distribution(generator);
            *j = randomNumber;
        }
    }
}

Layer::Matrix Layer::runInputs(const Matrix& m)
{
    util::log("Layer") << " Running forward propagation on matrix (" << m.rows()
            << " rows, " << m.columns() << " columns).\n";
    
    unsigned int inputPixPos = 0;
    Matrix finalOutput;
    // sparse multiply
    // slice input Matrix into chunks multipliable to matrix blocks
    for (auto i = m_sparseMatrix.begin(); i != m_sparseMatrix.end(); ++i)
    {
        Matrix temp = m.slice(inputPixPos, 0, m.rows(), (*i).rows());
        inputPixPos += (*i).rows();
        Matrix output = temp.multiply((*i)).sigmoid();
        util::log("Layer") << "  output: " << output.toString() << "\n";
        finalOutput = finalOutput.append(output);
    }
    
    util::log("Layer") << "  layer output is a matrix (" << finalOutput.rows()
            << " rows, " << finalOutput.columns() << " columns).\n";
    
    return finalOutput;
}

Layer::Matrix Layer::runReverse(const Matrix& m)
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
        Matrix output = temp.multiply(sparseMatrixT).sigmoid();
        finalOutput = finalOutput.append(output);
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

    return count;
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


