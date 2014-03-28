/*	\file   test-matrix.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the matrix unit test class.
*/

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <iostream>

// Global Typedefs
typedef minerva::matrix::Matrix Matrix;
typedef minerva::matrix::BlockSparseMatrix BlockSparseMatrix;

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

bool testTranspose()
{
	Matrix a(3, 2);
	
	a(0, 0) = 1;
	a(0, 1) = 2;
	a(1, 0) = 3;
	a(1, 1) = 4;
	a(2, 0) = 5;
	a(2, 1) = 6;
	
	Matrix t(2, 3);
	
	t(0, 0) = 1;
	t(0, 1) = 3;
	t(0, 2) = 5;
	t(1, 0) = 2;
	t(1, 1) = 4;
	t(1, 2) = 6;


	Matrix computed = a.transpose();
	
	if(computed != t)
	{
		std::cout << " Matrix Transpose Test Failed:\n";
		std::cout << "  result matrix " << computed.toString();
		std::cout << "  does not match reference matrix " << t.toString();
	}
	else
	{
		std::cout << " Matrix Transpose Test Passed\n";
	}
	
	return computed == t;

}

bool testSparseMultiply()
{
	BlockSparseMatrix a(1, 2, 3);
	
	BlockSparseMatrix b(1, 2, 3);
	
	BlockSparseMatrix c(1, 3, 3);
	
	
	a[0](0, 0) = 1;
	a[0](1, 0) = 2;
	a[0](0, 1) = 3;
	a[0](1, 1) = 4;
	a[0](0, 2) = 5;
	a[0](1, 2) = 6;

	b[0](0, 0) = 1;
	b[0](0, 1) = 2;
	b[0](0, 2) = 3;
	b[0](1, 0) = 4;
	b[0](1, 1) = 5;
	b[0](1, 2) = 6;

	c[0](0, 0) = 9;
	c[0](0, 1) = 12;
	c[0](0, 2) = 15;
	c[0](1, 0) = 19;
	c[0](1, 1) = 26;
	c[0](1, 2) = 33;
	c[0](2, 0) = 29;
	c[0](2, 1) = 40;
	c[0](2, 2) = 51;
	
	BlockSparseMatrix computed = a.transpose().multiply(b);
	
	if(computed[0] != c[0])
	{
		std::cout << " Block Sparse Matrix Multiply Test Failed:\n";
		std::cout << "  result matrix " << computed.toString();
		std::cout << "  does not match reference matrix " << c.toString();
	}
	else
	{
		std::cout << " Block Sparse Matrix Multiply Test Passed\n";
	}
	
	return computed[0] == c[0];

}

bool testSparseMultiply2()
{
	BlockSparseMatrix a(1, 3, 2);
	
	BlockSparseMatrix b(1, 3, 2);
	
	BlockSparseMatrix c(1, 3, 3);
	
	a[0](0, 0) = 1;
	a[0](0, 1) = 2;
	a[0](1, 0) = 3;
	a[0](1, 1) = 4;
	a[0](2, 0) = 5;
	a[0](2, 1) = 6;

	b[0](0, 0) = 1;
	b[0](1, 0) = 2;
	b[0](2, 0) = 3;
	b[0](0, 1) = 4;
	b[0](1, 1) = 5;
	b[0](2, 1) = 6;

	c[0](0, 0) = 9;
	c[0](0, 1) = 12;
	c[0](0, 2) = 15;
	c[0](1, 0) = 19;
	c[0](1, 1) = 26;
	c[0](1, 2) = 33;
	c[0](2, 0) = 29;
	c[0](2, 1) = 40;
	c[0](2, 2) = 51;
	
	BlockSparseMatrix computed = a.multiply(b.transpose());
	
	if(computed[0] != c[0])
	{
		std::cout << " Block Sparse Matrix Multiply Test Failed:\n";
		std::cout << "  result matrix " << computed.toString();
		std::cout << "  does not match reference matrix " << c.toString();
	}
	else
	{
		std::cout << " Block Sparse Matrix Multiply Test Passed\n";
	}
	
	return computed[0] == c[0];

}

int main(int argc, char** argv)
{
	minerva::util::enableAllLogs();
	
	std::cout << "Running matrix test unit tests\n";
	
    bool passed = true;
    
    passed &= testMultiply();
	passed &= testTranspose();
    passed &= testSparseMultiply();
    passed &= testSparseMultiply2();

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




