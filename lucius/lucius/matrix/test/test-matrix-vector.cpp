/*  \file   test-matrix-vector.cpp
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the matrix vector unit test class.
*/

// Lucius Includes
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixVectorOperations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <iostream>

// Global Typedefs
typedef lucius::matrix::Matrix Matrix;
typedef lucius::matrix::MatrixVector MatrixVector;

/*
    Test matrix addition

   [ [ 1 2 ] [ 1 3 ] ] + [ [ 5 6 ] [ 5 7 ] ] = [ [  6  8 ] [ 6 10 ] ]
   [ [ 3 4 ] [ 2 4 ] ]   [ [ 7 8 ] [ 6 8 ] ]   [ [ 10 12 ] [ 8 12 ] ]

*/
bool testAddition()
{
    MatrixVector a({Matrix(2, 2), Matrix(2, 2)});

    MatrixVector b({Matrix(2, 2), Matrix(2, 2)});

    MatrixVector c({Matrix(2, 2), Matrix(2, 2)});

    a[0](0, 0) = 1;
    a[0](0, 1) = 2;
    a[0](1, 0) = 3;
    a[0](1, 1) = 4;

    a[1](0, 0) = 1;
    a[1](0, 1) = 3;
    a[1](1, 0) = 2;
    a[1](1, 1) = 4;

    b[0](0, 0) = 5;
    b[0](0, 1) = 6;
    b[0](1, 0) = 7;
    b[0](1, 1) = 8;

    b[1](0, 0) = 5;
    b[1](0, 1) = 7;
    b[1](1, 0) = 6;
    b[1](1, 1) = 8;

    c[0](0, 0) = 6;
    c[0](0, 1) = 8;
    c[0](1, 0) = 10;
    c[0](1, 1) = 12;

    c[1](0, 0) = 6;
    c[1](0, 1) = 10;
    c[1](1, 0) = 8;
    c[1](1, 1) = 12;

    auto computed = apply(MatrixVector(a), b, lucius::matrix::Add());

    if(computed != c)
    {
        std::cout << " Matrix Vector Addition Test Failed:\n";
        std::cout << "  result matrix vector " << computed.toString();
        std::cout << "  does not match reference matrix vector " << c.toString();
    }
    else
    {
        std::cout << " Matrix Vector Addition Test Passed\n";
    }

    return computed == c;
}

/*
    Test matrix vector zeros

   zeros = [ [ 0 0 ] [ 0 0 ] ]
           [ [ 0 0 ] [ 0 0 ] ]

*/
bool testZeros()
{
    MatrixVector computed({Matrix(2, 2), Matrix(2, 2)});

    MatrixVector c({Matrix(2, 2), Matrix(2, 2)});

    zeros(computed);

    c[0](0, 0) = 0;
    c[0](0, 1) = 0;
    c[0](1, 0) = 0;
    c[0](1, 1) = 0;

    c[1](0, 0) = 0;
    c[1](0, 1) = 0;
    c[1](1, 0) = 0;
    c[1](1, 1) = 0;

    if(computed != c)
    {
        std::cout << " Matrix Vector Zeros Test Failed:\n";
        std::cout << "  result matrix vector " << computed.toString();
        std::cout << "  does not match reference matrix vector " << c.toString();
    }
    else
    {
        std::cout << " Matrix Vector Zeros Test Passed\n";
    }

    return computed == c;
}

/*
    Test matrix vector reduction

   [ [ 1 2 ] [ 1 3 ] ] reduce-sum  = [ 1 + 2 + 3 + 4 + 1 + 3 + 2 + 4 ] = 20
   [ [ 3 4 ] [ 2 4 ] ]

*/
bool testReduction()
{
    MatrixVector a({Matrix(2, 2), Matrix(2, 2)});

    a[0](0, 0) = 1;
    a[0](0, 1) = 2;
    a[0](1, 0) = 3;
    a[0](1, 1) = 4;

    a[1](0, 0) = 1;
    a[1](0, 1) = 3;
    a[1](1, 0) = 2;
    a[1](1, 1) = 4;

    double computed = reduce(a, {}, lucius::matrix::Add())[0][0];

    auto c = 20.0;

    if(computed != c)
    {
        std::cout << " Matrix Vector Reduction Test Failed:\n";
        std::cout << "  result value " << computed << "\n";
        std::cout << "  does not match reference value " << c << "\n";
    }
    else
    {
        std::cout << " Matrix Vector Reduction Test Passed\n";
    }

    return computed == c;
}

/*
    Test matrix vector dot product

   [ [ 1 2 ] [ 1 3 ] ] + [ [ 5 6 ] [ 5 7 ] ] = [ 5 + 12 + 21 + 32 + 5 + 21 + 12 + 32 ] = 140
   [ [ 3 4 ] [ 2 4 ] ]   [ [ 7 8 ] [ 6 8 ] ]

*/
bool testDotProduct()
{
    MatrixVector a({Matrix(2, 2), Matrix(2, 2)});

    MatrixVector b({Matrix(2, 2), Matrix(2, 2)});

    a[0](0, 0) = 1;
    a[0](0, 1) = 2;
    a[0](1, 0) = 3;
    a[0](1, 1) = 4;

    a[1](0, 0) = 1;
    a[1](0, 1) = 3;
    a[1](1, 0) = 2;
    a[1](1, 1) = 4;

    b[0](0, 0) = 5;
    b[0](0, 1) = 6;
    b[0](1, 0) = 7;
    b[0](1, 1) = 8;

    b[1](0, 0) = 5;
    b[1](0, 1) = 7;
    b[1](1, 0) = 6;
    b[1](1, 1) = 8;

    auto computed = dotProduct(a, b);

    auto c = 140.0;

    if(computed != c)
    {
        std::cout << " Matrix Vector Dot Product Test Failed:\n";
        std::cout << "  result value " << computed << "\n";
        std::cout << "  does not match reference value " << c << "\n";
    }
    else
    {
        std::cout << " Matrix Vector Dot Product Test Passed\n";
    }

    return computed == c;
}

int main(int argc, char** argv)
{
    lucius::util::enableAllLogs();

    std::cout << "Running matrix vector unit tests\n";

    bool passed = true;

    passed &= testAddition();
    passed &= testZeros();
    passed &= testReduction();
    passed &= testDotProduct();

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



