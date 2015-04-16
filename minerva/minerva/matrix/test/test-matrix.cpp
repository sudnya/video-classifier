/*    \file   test-matrix.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the matrix unit test class.
*/

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlasOperations.h>
#include <minerva/matrix/interface/CopyOperations.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/RandomOperations.h>
#include <minerva/matrix/interface/MatrixTransformations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <iostream>

// Global Typedefs
typedef minerva::matrix::Matrix Matrix;

/*
    A simple matrix multiply test
    
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
    
    Matrix computed = gemm(a, b);
    
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

/*
    A sliced matrix multiply test
    
    [ [ 1 2 ] 7 ] X [ [ 1 2 3 ] ] = [  9 12 15 ]
    [ [ 3 4 ] 8 ]   [ [ 4 5 6 ] ]   [ 19 26 33 ]
    [ [ 5 6 ] 9 ]   [   7 8 9   ]   [ 29 40 51 ]
    
*/
bool testMultiplySlice()
{
    Matrix a(3, 3);
    
    Matrix b(3, 3);
    
    Matrix c(3, 3);
    
    
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(0, 2) = 7;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(1, 2) = 8;
    a(2, 0) = 5;
    a(2, 1) = 6;
    a(2, 2) = 9;

    b(0, 0) = 1;
    b(0, 1) = 2;
    b(0, 2) = 3;
    b(1, 0) = 4;
    b(1, 1) = 5;
    b(1, 2) = 6;
    b(2, 0) = 7;
    b(2, 1) = 8;
    b(2, 2) = 9;

    c(0, 0) = 9;
    c(0, 1) = 12;
    c(0, 2) = 15;
    c(1, 0) = 19;
    c(1, 1) = 26;
    c(1, 2) = 33;
    c(2, 0) = 29;
    c(2, 1) = 40;
    c(2, 2) = 51;
    
    Matrix computed = gemm(slice(a, {0, 0}, {3, 2}), slice(b, {0, 0}, {2, 3}));
    
    if(computed != c)
    {
        std::cout << " Matrix Multiply Slice Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix Multiply Slice Test Passed\n";
    }
    
    return computed == c;
}

/*
    Test matrix addition
    
    [ 1 2 ] + [ 1 2 ] = [  2  4 ]
    [ 3 4 ]   [ 3 4 ]   [  6  8 ]
    [ 5 6 ]   [ 5 6 ]   [ 10 12 ]
    
*/
bool testAddition()
{
    Matrix a(3, 2);
    
    Matrix b(3, 2);
    
    Matrix c(3, 2);
    
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;
    
    copy(b, a);

    c(0, 0) = 2;
    c(0, 1) = 4;
    c(1, 0) = 6;
    c(1, 1) = 8;
    c(2, 0) = 10;
    c(2, 1) = 12;
    
    Matrix computed = apply(Matrix(a), b, minerva::matrix::Add());
    
    if(computed != c)
    {
        std::cout << " Matrix Addition Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix Addition Test Passed\n";
    }
    
    return computed == c;
}
/*
    Test matrix scalar addition
    
    [ 1 2 ] +  7 = [  8  9 ]
    [ 3 4 ]        [ 10 11 ]
    [ 5 6 ]        [ 12 13 ]
    
*/
bool testScalarAddition()
{
    Matrix a(3, 2);
    
    Matrix c(3, 2);
    
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;
    
    c(0, 0) = 8;
    c(0, 1) = 9;
    c(1, 0) = 10;
    c(1, 1) = 11;
    c(2, 0) = 12;
    c(2, 1) = 13;
    
    Matrix computed = apply(Matrix(a), minerva::matrix::Add(7));
    
    if(computed != c)
    {
        std::cout << " Matrix Scalar Addition Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix Scalar Addition Test Passed\n";
    }
    
    return computed == c;
}

/*
    Test reduce
    
    [ 1 2 ] = [ 21 ]
    [ 3 4 ]   
    [ 5 6 ]   
    
*/
bool testReduce()
{
    Matrix a(3, 2);
    
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;
    
    Matrix computed = reduce(a, {}, minerva::matrix::Add());
    
    if(computed[0] != 21.0)
    {
        std::cout << " Matrix Reduction Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << 21.0;
    }
    else
    {
        std::cout << " Matrix Reduction Test Passed\n";
    }
    
    return computed[0] == 21.0;
}

/*
    Test 2D reduce
    
    [ 1 2 ] = [  3 ]
    [ 3 4 ]   [  7 ]
    [ 5 6 ]   [ 11 ]
    
*/
bool test2dReduce()
{
    Matrix a(3, 2);
    
    Matrix c(3);
    
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;
    
    c(0) = 3;
    c(1) = 7;
    c(2) = 11;
    
    Matrix computed = reduce(a, {1}, minerva::matrix::Add());
    
    if(computed != c)
    {
        std::cout << " Matrix 2D Reduction 0th Dimension Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix 2D Reduction 0th Dimension Test Passed\n";
    }
    
    return computed == c;
}

/*
    Test 2D reduce
    
    [ 1 2 ] = [  9 ]
    [ 3 4 ]   [ 12 ]
    [ 5 6 ]   
    
*/
bool test2dReduce2()
{
    Matrix a(3, 2);
    
    Matrix c(2);
    
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;
    
    c(0) = 9;
    c(1) = 12;
    
    Matrix computed = reduce(a, {0}, minerva::matrix::Add());
    
    if(computed != c)
    {
        std::cout << " Matrix 2D Reduction 1st Dimension Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix 2D Reduction 1st Dimension Test Passed\n";
    }
    
    return computed == c;
}
/*
	Broadcast vector over matrix
	[ 1 3 5 7]     				[7]   	[  8 10 12 14]
	[ 7 1 3 5] broadcast(add) 	[8]  =	[ 15  9 11 13]
	[ 0 4 8 2]					[9]		[  9 13 17 11]

*/
bool testBroadcast()
{
	Matrix a(3, 4);
	Matrix b(3);
	Matrix ref(3,4);

	a(0, 0) = 1;
	a(0, 1) = 3;
	a(0, 2) = 5;
	a(0, 3) = 7;
	a(1, 0) = 7;
	a(1, 1) = 1;
	a(1, 2) = 3;
	a(1, 3) = 5;
	a(2, 0) = 0;
	a(2, 1) = 4;
	a(2, 2) = 8;
	a(2, 3) = 2;

	b(0) = 7;
	b(1) = 8;
	b(2) = 9;

	ref(0, 0) = 8;
	ref(0, 1) = 10;
	ref(0, 2) = 12;
	ref(0, 3) = 14;
	ref(1, 0) = 15;
	ref(1, 1) = 9;
	ref(1, 2) = 11;
	ref(1, 3) = 13;
	ref(2, 0) = 9;
	ref(2, 1) = 13;
	ref(2, 2) = 17;
	ref(2, 3) = 11;

	Matrix computed = broadcast(a, b, minerva::matrix::Add());
	
    if(computed != ref)
    {
        std::cout << " Matrix Broadcast Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << ref.toString();
    }
    else
    {
        std::cout << " Matrix Broadcast Test Passed\n";
    }
    
    return computed == ref;
 
}

bool testZeros()
{
	Matrix ref(3,4);
	
	ref(0, 0) = 0; 
	ref(0, 1) = 0; 
	ref(0, 2) = 0; 
	ref(0, 3) = 0; 
	ref(1, 0) = 0; 
	ref(1, 1) = 0; 
	ref(1, 2) = 0; 
	ref(1, 3) = 0; 
	ref(2, 0) = 0; 
	ref(2, 1) = 0; 
	ref(2, 2) = 0; 
	ref(2, 3) = 0; 

	Matrix computed = zeros(ref.size(), ref.precision());
    
	if(computed != ref)
    {
        std::cout << " Matrix Zeros Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << ref.toString();
    }
    else
    {
        std::cout << " Matrix Zeros Test Passed\n";
    }
    
    return computed == ref;
}

/*
    A simple reshape test.
    
    [ 1 ] reshape = [ 1 5 4 ] 
    [ 3 ]           [ 3 2 6 ]
    [ 5 ]               
    [ 2 ]
    [ 4 ]
    [ 6 ]
    
*/
bool testReshape()
{
    Matrix a(6);
    
	a(0) = 1;
    a(1) = 3;
    a(2) = 5;
    a(3) = 2;
    a(4) = 4;
    a(5) = 6;
    
    Matrix c(2, 3);
    
    c(0, 0) = 1;
    c(0, 1) = 5;
    c(0, 2) = 4;
    c(1, 0) = 3;
    c(1, 1) = 2;
    c(1, 2) = 6;
    
    Matrix computed = reshape(a, c.size());
    
    if(computed != c)
    {
        std::cout << " Matrix Reshape Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix Reshape Test Passed\n";
    }
    
    return computed == c;
}

/*
    Test matrix copy
    
    [ 1 2 ]  = [ 1 2 ]
    [ 3 4 ]    [ 3 4 ]
    [ 5 6 ]    [ 5 6 ]
    
*/
bool testCopy()
{
    Matrix a(3, 2);
    
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;
    
    Matrix computed = copy(a);
    
    if(computed != a)
    {
        std::cout << " Matrix Copy Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << a.toString();
    }
    else
    {
        std::cout << " Matrix Copy Test Passed\n";
    }
    
    return computed == a;
}

/*
    Test matrix copy between precisions
    
    [ 1 2 ]  = [ 1 2 ]
    [ 3 4 ]    [ 3 4 ]
    [ 5 6 ]    [ 5 6 ]
    
*/
bool testCopyBetweenPrecisions()
{
    Matrix a({3, 2}, minerva::matrix::SinglePrecision());
    
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;
    
	Matrix c({3, 2}, minerva::matrix::DoublePrecision());
    
    c(0, 0) = 1;
    c(0, 1) = 2;
    c(1, 0) = 3;
    c(1, 1) = 4;
    c(2, 0) = 5;
    c(2, 1) = 6;
    
    Matrix computed = copy(Matrix(a), minerva::matrix::DoublePrecision());
    
    if(computed != c)
    {
        std::cout << " Matrix Copy Between Precisions Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix Copy Between Precisions Test Passed\n";
    }
    
    return computed == c;
}

/*
	Test uniform random matrix.
*/
bool testUniformRandom()
{
	minerva::matrix::srand(377);
	
	size_t size = 100;
	
	auto a = minerva::matrix::rand({size, size}, minerva::matrix::SinglePrecision());
	
	auto mean = reduce(apply(a, minerva::matrix::Divide(1.0 / (size * size))), {}, minerva::matrix::Add())[0];
	
	bool passed = (mean < 0.1 && mean > -0.1);
	
	if(!passed)
	{
        std::cout << " Matrix Uniform Random Test Failed:\n";
        std::cout << "  result mean " << mean << " is out of bounds.\n";
    }
    else
    {
        std::cout << " Matrix Uniform Random Test Passed\n";
	}
	
	return passed;
}


int main(int argc, char** argv)
{
    minerva::util::enableAllLogs();
    
    std::cout << "Running matrix test unit tests\n";
    
    bool passed = true;
    
    passed &= testMultiplySlice();
    passed &= testMultiply();
    passed &= testAddition();
    passed &= testScalarAddition();
    passed &= testReduce();
    passed &= test2dReduce();
    passed &= test2dReduce2();
	passed &= testBroadcast();
	passed &= testZeros();
	passed &= testReshape();
	passed &= testCopy();
	passed &= testCopyBetweenPrecisions();
	passed &= testUniformRandom();

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


