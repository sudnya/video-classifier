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
		std::cout << " Block Sparse Matrix Multiply Test 2 Failed:\n";
		std::cout << "  result matrix " << computed.toString();
		std::cout << "  does not match reference matrix " << c.toString();
	}
	else
	{
		std::cout << " Block Sparse Matrix Multiply Test 2 Passed\n";
	}
	
	return computed[0] == c[0];
}

bool testSparseConvolutionalMultiply()
{
	BlockSparseMatrix a(2, 3, 2, false);
	
	BlockSparseMatrix b(1, 2, 3, true);
	
	BlockSparseMatrix c(3, 3, 3, false);
	
	
	a[0](0, 0) = 1;
	a[0](0, 1) = 2;
	a[0](1, 0) = 3;
	a[0](1, 1) = 4;
	a[0](2, 0) = 5;
	a[0](2, 1) = 6;
	
	a[1](0, 0) = 7;
	a[1](0, 1) = 8;
	a[1](1, 0) = 9;
	a[1](1, 1) = 10;
	a[1](2, 0) = 11;
	a[1](2, 1) = 12;

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

	c[1](0, 0) = 30;
	c[1](0, 1) = 39;
	c[1](0, 2) = 48;

	c[1](1, 0) = 40;
	c[1](1, 1) = 53;
	c[1](1, 2) = 66;

	c[1](2, 0) = 50;
	c[1](2, 1) = 67;
	c[1](2, 2) = 84;

	c[2](0, 0) = 39;
	c[2](0, 1) = 54;
	c[2](0, 2) = 69;

	c[2](1, 0) = 49;
	c[2](1, 1) = 68;
	c[2](1, 2) = 87;

	c[2](2, 0) = 59;
	c[2](2, 1) = 82;
	c[2](2, 2) = 105;

	BlockSparseMatrix computed = a.convolutionalMultiply(b, 1);
	
	if(computed != c)
	{
		std::cout << " Block Sparse Convolutional Matrix Multiply Test Failed:\n";
		std::cout << "  result matrix " << computed.toMatrix().toString();
		std::cout << "  does not match reference matrix " << c.toMatrix().toString();
	}
	else
	{
		std::cout << " Block Sparse Convolutional Matrix Multiply Test Passed\n";
	}
	
	return computed == c;
}

bool testSparseConvolutionalMultiply2()
{
	BlockSparseMatrix a(2, 3, 2, false);
	
	BlockSparseMatrix b(1, 4, 3, true);
	
	BlockSparseMatrix c(1, 3, 3, false);
	
	a[0](0, 0) = 1;
	a[0](0, 1) = 2;
	a[0](1, 0) = 3;
	a[0](1, 1) = 4;
	a[0](2, 0) = 5;
	a[0](2, 1) = 6;
	
	a[1](0, 0) = 7;
	a[1](0, 1) = 8;
	a[1](1, 0) = 9;
	a[1](1, 1) = 10;
	a[1](2, 0) = 11;
	a[1](2, 1) = 12;

	b[0](0, 0) = 1;
	b[0](0, 1) = 2;
	b[0](0, 2) = 3;
	b[0](1, 0) = 4;
	b[0](1, 1) = 5;
	b[0](1, 2) = 6;
	b[0](2, 0) = 7;
	b[0](2, 1) = 8;
	b[0](2, 2) = 9;
	b[0](3, 0) = 10;
	b[0](3, 1) = 11;
	b[0](3, 2) = 12;

	c[0](0, 0) = 138;
	c[0](0, 1) = 156;
	c[0](0, 2) = 174;

	c[0](1, 0) = 182;
	c[0](1, 1) = 208;
	c[0](1, 2) = 234;

	c[0](2, 0) = 226;
	c[0](2, 1) = 260;
	c[0](2, 2) = 294;

	BlockSparseMatrix computed = a.convolutionalMultiply(b, 4);
	
	if(computed != c)
	{
		std::cout << " Block Sparse Convolutional Matrix Multiply Test 2 Failed:\n";
		std::cout << "  result matrix " << computed.toMatrix().toString();
		std::cout << "  does not match reference matrix " << c.toMatrix().toString();
	}
	else
	{
		std::cout << " Block Sparse Convolutional Matrix Multiply Test 2 Passed\n";
	}
	
	return computed == c;
}

bool testSparseConvolutionalAddBroadcastRow()
{
	BlockSparseMatrix a(2, 3, 2, false);
	
	BlockSparseMatrix b(1, 1, 2, false);
	
	BlockSparseMatrix c(2, 3, 2, false);
	
	a[0](0, 0) = 1;
	a[0](0, 1) = 2;
	a[0](1, 0) = 3;
	a[0](1, 1) = 4;
	a[0](2, 0) = 5;
	a[0](2, 1) = 6;
	
	a[1](0, 0) = 7;
	a[1](0, 1) = 8;
	a[1](1, 0) = 9;
	a[1](1, 1) = 10;
	a[1](2, 0) = 11;
	a[1](2, 1) = 12;

	b[0](0, 0) = 1;
	b[0](0, 1) = 2;
	
	c[0](0, 0) = 2;
	c[0](0, 1) = 4;
	c[0](1, 0) = 4;
	c[0](1, 1) = 6;
	c[0](2, 0) = 6;
	c[0](2, 1) = 8;
	
	c[1](0, 0) = 8;
	c[1](0, 1) = 10;
	c[1](1, 0) = 10;
	c[1](1, 1) = 12;
	c[1](2, 0) = 12;
	c[1](2, 1) = 14;
	
	BlockSparseMatrix computed = a.convolutionalAddBroadcastRow(b);
	
	if(computed != c)
	{
		std::cout << " Block Sparse Matrix Convolutional Add Broadcast Row Test Failed:\n";
		std::cout << "  result matrix " << computed.toMatrix().toString();
		std::cout << "  does not match reference matrix " << c.toMatrix().toString();
	}
	else
	{
		std::cout << " Block Sparse Matrix Convolutional Add Broadcast Row Test Passed\n";
	}
	
	return computed == c;
}

bool testSparseReduceTileSumAlongRows()
{
	BlockSparseMatrix a(4, 2, 2, true);
	
	BlockSparseMatrix b(2, 2, 2, true);
	
	BlockSparseMatrix c(2, 2, 2, true);
	
	a[0](0, 0) = 1;
	a[0](0, 1) = 2;
	a[0](1, 0) = 3;
	a[0](1, 1) = 4;
	
	a[1](0, 0) = 5;
	a[1](0, 1) = 6;
	a[1](1, 0) = 7;
	a[1](1, 1) = 8;
	
	a[2](0, 0) = 9;
	a[2](0, 1) = 10;
	a[2](1, 0) = 11;
	a[2](1, 1) = 12;
	
	a[3](0, 0) = 13;
	a[3](0, 1) = 14;
	a[3](1, 0) = 15;
	a[3](1, 1) = 16;

	b[0](0, 0) = 1;
	b[0](0, 1) = 2;
	b[0](1, 0) = 3;
	b[0](1, 1) = 4;
	
	b[1](0, 0) = 5;
	b[1](0, 1) = 6;
	b[1](1, 0) = 7;
	b[1](1, 1) = 8;
	
	c[0](0, 0) = 10;
	c[0](0, 1) = 12;
	c[0](1, 0) = 14;
	c[0](1, 1) = 16;
	
	c[1](0, 0) = 18;
	c[1](0, 1) = 20;
	c[1](1, 0) = 22;
	c[1](1, 1) = 24;
	
	BlockSparseMatrix computed = a.reduceTileSumAlongRows(b.rowsPerBlock(), c.blocks());
	
	if(computed != c)
	{
		std::cout << " Block Sparse Matrix Reduce Tiled Sum Along Rows Test Failed:\n";
		std::cout << "  result matrix " << computed.toMatrix().toString();
		std::cout << "  does not match reference matrix " << c.toMatrix().toString();
	}
	else
	{
		std::cout << " Block Sparse Matrix Reduce Tiled Sum Along Rows Test Passed\n";
	}
	
	return computed == c;
}

bool testSparseReduceSumAlongRows()
{
	BlockSparseMatrix a(2, 2, 3, false);
	
	BlockSparseMatrix c(2, 1, 3, false);
	
	a[0](0, 0) = 1;
	a[0](1, 0) = 2;
	a[0](0, 1) = 3;
	a[0](1, 1) = 4;
	a[0](0, 2) = 5;
	a[0](1, 2) = 6;

	a[1](0, 0) = 7;
	a[1](1, 0) = 8;
	a[1](0, 1) = 9;
	a[1](1, 1) = 10;
	a[1](0, 2) = 11;
	a[1](1, 2) = 12;
	
	c[0](0, 0) = 3;
	c[0](0, 1) = 7;
	c[0](0, 2) = 11;
	
	c[1](0, 0) = 15;
	c[1](0, 1) = 19;
	c[1](0, 2) = 23;
	
	BlockSparseMatrix computed = a.reduceSumAlongRows();
	
	if(computed != c)
	{
		std::cout << " Block Sparse Reduce Sum Along Rows Test Failed:\n";
		std::cout << "  result matrix " << computed.toMatrix().toString();
		std::cout << "  does not match reference matrix " << c.toMatrix().toString();
	}
	else
	{
		std::cout << " Block Sparse Reduce Sum Along Rows Test Passed\n";
	}
	
	return computed == c;
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
    passed &= testSparseConvolutionalMultiply();
    passed &= testSparseConvolutionalMultiply2();
    passed &= testSparseConvolutionalAddBroadcastRow();
    passed &= testSparseReduceTileSumAlongRows();
    passed &= testSparseReduceSumAlongRows();

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





