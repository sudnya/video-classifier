/*	\file   InputDataProducer.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the InputDataProducer class.
*/

#pragma once

namespace minerva
{

namespace input
{

/*! \brief A class for accessing a stream of data */
class InputDataProducer
{
public:
	typedef matrix::Matrix Matrix;

public:
	virtual ~InputDataProducer();

public:
	/*! \brief Deque a set of samples from the producer.

		Note: the caller must ensure that the producer is not empty.

	*/
	virtual Matrix pop() = 0;
	
	/*! \brief Return true if there are no more samples. */
	virtual bool empty() const = 0;

	/*! \brief Reset the producer to its original state, all previously
		popped samples should now be available again. */
	virtual void reset() = 0;
	
public:
	void setAllowSamplingWithReplacement(bool allowReplacement);
	bool getAllowSamplingWithReplacement() const;

public:
	void   setBatchSize(size_t batchSize);
	size_t getBatchSize() const;

public:
	void setRequiresLabelledData(bool requiresLabelledData);
	bool getRequiresLabelledData() const; 
	
private:
	bool   _allowSamplingWithReplacement;
	size_t _batchSize;

};

}

}

