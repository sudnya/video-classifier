/*	\file   test-matrix.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the matrix unit test class.
*/

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <iostream>

// Global Typedefs
typedef minerva::matrix::Matrix Matrix;

/*
	A simple matrix test
	
	[ 1 2 ] X [ 1 2 3 ] = [  9 12 15 ]
	[ 3 4 ]   [ 4 5 6 ]   [ 19 26 33 ]
	[ 5 6 ]               [ 29 40 51 ]
	
*/
bool testMultiply()
{
	Matrix a(3, 2);
	
	Matrix b(2, 3);
	
	Matrix c(3, 3);
	
	
	a(0, 0) = 1;
	a(0, 1) = 2;
	a(1, 0) = 3;
	a(1, 1) = 4;
	a(2, 0) = 5;
	a(2, 1) = 6;

	b(0, 0) = 1;
	b(0, 1) = 2;
	b(0, 2) = 3;
	b(1, 0) = 4;
	b(1, 1) = 5;
	b(1, 2) = 6;

	c(0, 0) = 9;
	c(0, 1) = 12;
	c(0, 2) = 15;
	c(1, 0) = 19;
	c(1, 1) = 26;
	c(1, 2) = 33;
	c(2, 0) = 29;
	c(2, 1) = 40;
	c(2, 2) = 51;
	
	Matrix computed = a.multiply(b);
	
	if(computed != c)
	{
		std::cout << " Matrix Multiply Test Failed:\n";
		std::cout << "  result matrix " << computed.toString();
		std::cout << "  does not match reference matrix " << c.toString();
	}
	else
	{
		std::cout << " Matrix Multiply Test Passed\n";
	}
	
	return computed == c;
}

int main(int argc, char** argv)
{
	minerva::util::enableAllLogs();
	
	std::cout << "Running matrix test unit tests\n";
	
    bool passed = true;
    
    passed &= testMultiply();

	if(not passed)
	{
		std::cout << "Test Failed\n";
	}
	else
	{
		std::cout << "Test Passed\n";
	}

	return 0;
}





