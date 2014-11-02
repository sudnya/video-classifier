/*	\file   ResultVector.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ResultVector class.
*/

#include <minerva/results/interface/ResultVector.h>

namespace minerva
{

namespace results
{

ResultVector::ResultVector()
{

}

ResultVector::~ResultVector()
{
	clear();
}

void ResultVector::push_back(Result* r)
{
	_results.push_back(r);
}

ResultVector::iterator ResultVector::begin()
{
	return _results.begin();
}

ResultVector::const_iterator ResultVector::begin() const
{
	return _results.begin();
}

ResultVector::iterator ResultVector::end()
{
	return _results.end();
}

ResultVector::const_iterator ResultVector::end() const
{
	return _results.end();
}

size_t ResultVector::size() const
{
	return _results.size();
}

bool ResultVector::empty() const
{
	return _results.empty();
}

void ResultVector::clear()
{
	for(auto result : *this)
	{
		delete result;
	}
}

}

}

