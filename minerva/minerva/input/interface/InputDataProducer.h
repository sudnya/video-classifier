/*	\file   InputDataProducer.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the InputDataProducer class.
*/

#pragma once

// Minerva Includes
#include <minerva/util/interface/string.h>

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }
namespace minerva { namespace model  { class Model;  } }

// Standard Library Includes
#include <utility>

namespace minerva
{

namespace input
{

/*! \brief A class for accessing a stream of data */
class InputDataProducer
{
public:
	typedef matrix::Matrix Matrix;
	typedef std::pair<Matrix, Matrix> InputAndReferencePair;

public:
	InputDataProducer();
	virtual ~InputDataProducer();

public:
	/*! \brief Deque a set of samples from the producer.

		Note: the caller must ensure that the producer is not empty.

	*/
	virtual InputAndReferencePair pop() = 0;
	
	/*! \brief Return true if there are no more samples. */
	virtual bool empty() const = 0;

	/*! \brief Reset the producer to its original state, all previously
		popped samples should now be available again. */
	virtual void reset() = 0;
	
	/*! \brief Get the total number of unique samples that can be produced. */
	virtual size_t getUniqueSampleCount() const = 0;
	
public:
	void   setEpochs(size_t epochs);
	size_t getEpochs() const;

public:
	void   setBatchSize(size_t batchSize);
	size_t getBatchSize() const;

public:
	void   setMaximumSamplesToRun(size_t batchSize);
	size_t getMaximumSamplesToRun() const;

public:
	void setRequiresLabeledData(bool requiresLabeledData);
	bool getRequiresLabeledData() const; 

public:
	void setModel(const model::Model* model);

protected:
	size_t getInputCount() const;
	size_t getInputBlockingFactor() const;

protected:
	util::StringVector getOutputLabels() const;
	
private:
	size_t _epochs;
	bool   _requiresLabeledData;
	size_t _batchSize;
	size_t _maximumSamplesToRun;

private:
	const model::Model* _model;

};

}

}

