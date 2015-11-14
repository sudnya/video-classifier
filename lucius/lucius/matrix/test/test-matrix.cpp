/*  \file   test-matrix.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the matrix unit test class.
*/

// Lucius Includes
#include <lucius/matrix/interface/BlasOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/FileOperations.h>
#include <lucius/matrix/interface/ConvolutionalOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <iostream>

// Global Typedefs
typedef lucius::matrix::Matrix Matrix;

/* A simple test to save and load from a numpy array */
bool testSaveLoad()
{
    //create a matrix
    Matrix reference (3, 2);
    reference(0, 0) = 1;
    reference(0, 1) = 2;
    reference(1, 0) = 3;
    reference(1, 1) = 4;
    reference(2, 0) = 5;
    reference(2, 1) = 6;

    //read it into a stream and call save
    std::stringstream stream;

    //use load to load it from a stream
    lucius::matrix::save(stream, reference);

    //compare reference Matrix to loaded Matrix
    auto result = lucius::matrix::load(stream);

    if(reference != result)
    {
        std::cout << " Matrix Save Load Test Failed:\n";
        std::cout << "  result matrix " << result.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix Save Load Test Passed\n";
    }

    return reference == result;
}

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

    Matrix computed = apply(Matrix(a), b, lucius::matrix::Add());

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

    Matrix computed = apply(Matrix(a), lucius::matrix::Add(7));

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

    Matrix computed = reduce(a, {}, lucius::matrix::Add());

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

    Matrix computed = reduce(a, {1}, lucius::matrix::Add());

    if(computed != c)
    {
        std::cout << " Matrix 2D Reduction 2nd Dimension Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix 2D Reduction 2nd Dimension Test Passed\n";
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

    Matrix computed = reduce(a, {0}, lucius::matrix::Add());

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
    Broadcast vector over 1th matrix dimension
    [ 1 3 5 7]                    [7]    [  8 10 12 14]
    [ 7 1 3 5] broadcast(add)     [8]  = [ 15  9 11 13]
    [ 0 4 8 2]                    [9]    [  9 13 17 11]

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

    Matrix computed = broadcast(a, b, {}, lucius::matrix::Add());

    if(computed != ref)
    {
        std::cout << " Matrix Broadcast 1st Dimension Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << ref.toString();
    }
    else
    {
        std::cout << " Matrix Broadcast 1st Dimension Test Passed\n";
    }

    return computed == ref;

}

/*
    Test 4D reduce

  [ [ [ 1  2 ] ] ] = [ [  8 12 ] ]
  [ [ [ 3  4 ] ] ]   [ [ 11 15 ] ]
  [ [ [ 5  6 ] ] ]   [ [ 14 18 ] ]
  [              ]
  [ [ [ 7 10 ] ] ]
  [ [ [ 8 11 ] ] ]
  [ [ [ 9 12 ] ] ]



*/
bool test4dReduce()
{
    Matrix a(3, 2, 1, 2);

    Matrix c(3, 2, 1);

    a(0, 0, 0, 0) = 1;
    a(0, 1, 0, 0) = 2;
    a(1, 0, 0, 0) = 3;
    a(1, 1, 0, 0) = 4;
    a(2, 0, 0, 0) = 5;
    a(2, 1, 0, 0) = 6;

    a(0, 0, 0, 1) = 7;
    a(0, 1, 0, 1) = 10;
    a(1, 0, 0, 1) = 8;
    a(1, 1, 0, 1) = 11;
    a(2, 0, 0, 1) = 9;
    a(2, 1, 0, 1) = 12;

    c(0, 0, 0, 0) = 8;
    c(0, 1, 0, 0) = 12;
    c(1, 0, 0, 0) = 11;
    c(1, 1, 0, 0) = 15;
    c(2, 0, 0, 0) = 14;
    c(2, 1, 0, 0) = 18;

    Matrix computed = reduce(a, {3}, lucius::matrix::Add());

    if(computed != c)
    {
        std::cout << " Matrix 4D Reduction 1st 3 Dimension Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix 4D Reduction 1st 3 Dimension Test Passed\n";
    }

    return computed == c;
}

/*
    Broadcast vector over 0th matrix dimension

    [ 1 3 5 7]                                  [  8 5  9 10]
    [ 7 1 3 5] broadcast(add) [ 7, 2, 4, 3 ]  = [ 14 3  7  8]
    [ 0 4 8 2]                                  [  7 6 12  5]

*/
bool test2dBroadcast()
{
    Matrix a(3, 4);
    Matrix b(4);
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
    b(1) = 2;
    b(2) = 4;
    b(3) = 3;

    ref(0, 0) = 8;
    ref(0, 1) = 5;
    ref(0, 2) = 9;
    ref(0, 3) = 10;
    ref(1, 0) = 14;
    ref(1, 1) = 3;
    ref(1, 2) = 7;
    ref(1, 3) = 8;
    ref(2, 0) = 7;
    ref(2, 1) = 6;
    ref(2, 2) = 12;
    ref(2, 3) = 5;

    Matrix computed = broadcast(a, b, {0}, lucius::matrix::Add());

    if(computed != ref)
    {
        std::cout << " Matrix Broadcast 0th Dimension Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << ref.toString();
    }
    else
    {
        std::cout << " Matrix Broadcast 0th Dimension Test Passed\n";
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
    Test reshaping a slice.

    [ 1  7 ] slice = [  7 ] reshape = [ 7  9 11 ]
    [ 2  8 ]         [  8 ]           [ 8 10 12 ]
    [ 3  9 ]         [  9 ]
    [ 4 10 ]         [ 10 ]
    [ 5 11 ]         [ 11 ]
    [ 6 12 ]         [ 12 ]

*/
bool testReshapeSlice()
{
    Matrix a(6, 2);

    a(0, 0) = 1;
    a(1, 0) = 2;
    a(2, 0) = 3;
    a(3, 0) = 4;
    a(4, 0) = 5;
    a(5, 0) = 6;
    a(0, 1) = 7;
    a(1, 1) = 8;
    a(2, 1) = 9;
    a(3, 1) = 10;
    a(4, 1) = 11;
    a(5, 1) = 12;

    Matrix c(2, 3);

    c(0, 0) = 7;
    c(0, 1) = 9;
    c(0, 2) = 11;
    c(1, 0) = 8;
    c(1, 1) = 10;
    c(1, 2) = 12;

    Matrix computed = reshape(slice(a, {0, 1}, {6, 2}), c.size());

    if(computed != c)
    {
        std::cout << " Matrix Reshape Slice Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << c.toString();
    }
    else
    {
        std::cout << " Matrix Reshape Slice Test Passed\n";
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
    Matrix a({3, 2}, lucius::matrix::SinglePrecision());

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;

    Matrix c({3, 2}, lucius::matrix::DoublePrecision());

    c(0, 0) = 1;
    c(0, 1) = 2;
    c(1, 0) = 3;
    c(1, 1) = 4;
    c(2, 0) = 5;
    c(2, 1) = 6;

    Matrix computed = copy(Matrix(a), lucius::matrix::DoublePrecision());

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
    lucius::matrix::srand(377);

    size_t size = 100;

    auto a = lucius::matrix::rand({size, size}, lucius::matrix::SinglePrecision());

    double mean = reduce(apply(a, lucius::matrix::Divide((size * size))), {}, lucius::matrix::Add())[0];

    bool passed = (mean < 0.6 && mean > 0.4);

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

/*
    Test normal random matrix.
*/
bool testNormalRandom()
{
    lucius::matrix::srand(377);

    size_t size = 100;

    auto a = lucius::matrix::randn({size, size}, lucius::matrix::SinglePrecision());

    double mean = reduce(apply(a, lucius::matrix::Divide((size * size))), {}, lucius::matrix::Add())[0];

    bool passed = (mean < 0.1 && mean > -0.1);

    if(!passed)
    {
        std::cout << " Matrix Normal Random Test Failed:\n";
        std::cout << "  result mean " << mean << " is out of bounds.\n";
    }
    else
    {
        std::cout << " Matrix Normal Random Test Passed\n";
    }

    return passed;
}

/*
    Test 1D forward convolution.

    [ 0 ] conv [ 0 ] = [  1 ]
    [ 1 ]      [ 1 ]   [  4 ]
    [ 2 ]      [ 2 ]   [  7 ]
    [ 3 ]              [ 10 ]
    [ 4 ]
    [ 5 ]

*/
bool test1dForwardConvolution()
{
    int n = 1;
    int c = 1;
    int h = 1;
    int w = 6;

    Matrix input(w, h, c, n);

    input(0, 0, 0, 0) = 0;
    input(1, 0, 0, 0) = 1;
    input(2, 0, 0, 0) = 2;
    input(3, 0, 0, 0) = 3;
    input(4, 0, 0, 0) = 4;
    input(5, 0, 0, 0) = 5;

    int k = 1;
    int r = 1;
    int s = 3;

    Matrix filter(s, r, c, k);

    filter(0, 0, 0, 0) = 0;
    filter(1, 0, 0, 0) = 1;
    filter(2, 0, 0, 0) = 2;

    Matrix reference(4, 1, 1, 1);

    reference(0, 0, 0, 0) = 1;
    reference(1, 0, 0, 0) = 4;
    reference(2, 0, 0, 0) = 7;
    reference(3, 0, 0, 0) = 10;

    auto computed = forwardConvolution(input, filter, {1, 1}, {0, 0});

    if(reference != computed)
    {
        std::cout << " Matrix 1D Forward Convolution Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix 1D Forward Convolution Test Passed\n";
    }

    return reference == computed;
}

/*
    Test 2D forward convolution.

   [ [  0  4  8 12 ] ] conv [ [ 0  3 ] ] = [ 5*0 + 4*1 + 3*2 + 2*4 + 1*5 + 0*6 = 23 ] + [ 11*16 + 10*17 + 9*18 + 8*20 + 7*21 + 6*22 ] = [  970 ]
   [ [  1  5  9 13 ] ]      [ [ 1  4 ] ]   [ 5*1 + 4*2 + 3*3 + 2*5 + 1*6 + 0*7 = 38 ]   [ 11*17 + 10*18 + 9*19 + 8*21 + 7*22 + 6*23 ] = [ 1036 ]
   [ [  2  6 10 14 ] ]      [ [ 2  5 ] ]
   [ [  3  7 11 15 ] ]      [          ]
   [                 ]      [ [ 6  9 ] ]
   [ [ 16 20 24 28 ] ]      [ [ 7 10 ] ]
   [ [ 17 21 25 29 ] ]      [ [ 8 11 ] ]
   [ [ 18 22 26 30 ] ]
   [ [ 19 23 27 31 ] ]

*/
bool test2dForwardConvolution()
{
    int n = 1;
    int c = 2;
    int h = 4;
    int w = 4;

    Matrix input(w, h, c, n);

    for(size_t i = 0; i < input.elements(); ++i)
    {
        input(i) = i;
    }

    int k = 1;
    int r = 2;
    int s = 3;

    Matrix filter(s, r, c, k);

    for(size_t i = 0; i < filter.elements(); ++i)
    {
        filter(i) = i;
    }

    Matrix reference(2, 3, 1, 1);

    reference(0, 0, 0, 0) = 970;
    reference(1, 0, 0, 0) = 1036;
    reference(0, 1, 0, 0) = 1234;
    reference(1, 1, 0, 0) = 1300;
    reference(0, 2, 0, 0) = 1498;
    reference(1, 2, 0, 0) = 1564;

    auto computed = forwardConvolution(input, filter, {1, 1}, {0, 0});

    if(reference != computed)
    {
        std::cout << " Matrix 2D Forward Convolution Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix 2D Forward Convolution Test Passed\n";
    }

    return reference == computed;
}

/*
    Test 2D strided forward convolution.


   [ [  0  5 10 ] ]      [ [ 0 ] ] = [ [ [  0  10 ] ] ]
   [ [  1  6 11 ] ] conv [ [ 1 ] ]   [ [ [  2  12 ] ] ]
   [ [  2  7 12 ] ]      [       ]   [ [            ] ]
   [ [  3  8 13 ] ]      [ [ 2 ] ]   [ [ [  2  52 ] ] ]
   [ [  4  9 14 ] ]      [ [ 3 ] ]   [ [ [ 12  62 ] ] ]
   [              ]                  [                ]
   [ [ 15 20 25 ] ]                  [ [ [ 15  25 ] ] ]
   [ [ 16 21 26 ] ]                  [ [ [ 17  27 ] ] ]
   [ [ 17 22 27 ] ]                  [ [            ] ]
   [ [ 18 23 28 ] ]                  [ [ [ 77 127 ] ] ]
   [ [ 19 24 29 ] ]                  [ [ [ 87 137 ] ] ]

*/
bool test2dStridedForwardConvolution()
{
    int n = 2;
    int c = 1;
    int h = 3;
    int w = 5;

    Matrix input(w, h, c, n);

    for(size_t i = 0; i < input.elements(); ++i)
    {
        input(i) = i;
    }

    int k = 2;
    int r = 1;
    int s = 2;

    Matrix filter(s, r, c, k);

    for(size_t i = 0; i < filter.elements(); ++i)
    {
        filter(i) = i;
    }

    Matrix reference(2, 2, k, n);

    reference(0, 0, 0, 0) = 0;
    reference(1, 0, 0, 0) = 2;
    reference(0, 1, 0, 0) = 10;
    reference(1, 1, 0, 0) = 12;

    reference(0, 0, 1, 0) = 2;
    reference(1, 0, 1, 0) = 12;
    reference(0, 1, 1, 0) = 52;
    reference(1, 1, 1, 0) = 62;

    reference(0, 0, 0, 1) = 15;
    reference(1, 0, 0, 1) = 17;
    reference(0, 1, 0, 1) = 25;
    reference(1, 1, 0, 1) = 27;

    reference(0, 0, 1, 1) = 77;
    reference(1, 0, 1, 1) = 87;
    reference(0, 1, 1, 1) = 127;
    reference(1, 1, 1, 1) = 137;

    auto computed = forwardConvolution(input, filter, {2, 2}, {0, 0});

    if(reference != computed)
    {
        std::cout << " Matrix 2D Strided Forward Convolution Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix 2D Strided Forward Convolution Test Passed\n";
    }

    return reference == computed;
}

/*
    Test 1D backward convolution.

    [ 0 ] conv-back [ 0 ] = [ 0 ]
    [ 1 ]           [ 1 ]   [ 2 ]
    [ 2 ]           [ 2 ]   [ 5 ]
    [ 3 ]                   [ 8 ]
                            [ 3 ]
                            [ 0 ]
*/
bool test1dBackwardConvolution()
{
    int n = 1;
    int c = 1;
    int h = 1;
    int w = 4;

    Matrix deltas(w, h, c, n);

    deltas(0, 0, 0, 0) = 0;
    deltas(1, 0, 0, 0) = 1;
    deltas(2, 0, 0, 0) = 2;
    deltas(3, 0, 0, 0) = 3;

    int k = 1;
    int r = 1;
    int s = 3;

    Matrix filter(s, r, c, k);

    filter(0, 0, 0, 0) = 0;
    filter(1, 0, 0, 0) = 1;
    filter(2, 0, 0, 0) = 2;

    Matrix reference(6, 1, 1, 1);

    reference(0, 0, 0, 0) = 0;
    reference(1, 0, 0, 0) = 2;
    reference(2, 0, 0, 0) = 4;
    reference(3, 0, 0, 0) = 6;
    reference(4, 0, 0, 0) = 0;
    reference(5, 0, 0, 0) = 0;

    Matrix computed(reference.size(), reference.precision());

    reverseConvolutionDeltas(computed, filter, {1, 1}, deltas, {0, 0});

    if(reference != computed)
    {
        std::cout << " Matrix 1D Backward Convolution Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix 1D Backward Convolution Test Passed\n";
    }

    return reference == computed;
}

/*
    Test 2D backward convolution.

    [ 0 3 ] conv-back [ 0 2 4 ] = [  0 15  9 3 ]
    [ 1 4 ]           [ 1 3 5 ]   [  5 35 19 4 ]
    [ 2 5 ]                       [ 14 49 25 5 ]
                                  [  8 24 10 0 ]


*/
bool test2dBackwardConvolution()
{
    int n = 1;
    int c = 1;
    int h = 2;
    int w = 3;

    Matrix deltas(w, h, c, n);

    deltas(0, 0, 0, 0) = 0;
    deltas(1, 0, 0, 0) = 1;
    deltas(2, 0, 0, 0) = 2;
    deltas(0, 1, 0, 0) = 3;
    deltas(1, 1, 0, 0) = 4;
    deltas(2, 1, 0, 0) = 5;

    int k = 1;
    int r = 3;
    int s = 2;

    Matrix filter(s, r, c, k);

    filter(0, 0, 0, 0) = 0;
    filter(1, 0, 0, 0) = 1;
    filter(0, 1, 0, 0) = 2;
    filter(1, 1, 0, 0) = 3;
    filter(0, 2, 0, 0) = 4;
    filter(1, 2, 0, 0) = 5;

    Matrix reference(4, 4, 1, 1);

    reference(0, 0, 0, 0) = 0;
    reference(1, 0, 0, 0) = 5;
    reference(2, 0, 0, 0) = 14;
    reference(3, 0, 0, 0) = 11;

    reference(0, 1, 0, 0) = 15;
    reference(1, 1, 0, 0) = 34;
    reference(2, 1, 0, 0) = 55;
    reference(3, 1, 0, 0) = 34;

    reference(0, 2, 0, 0) = 6;
    reference(1, 2, 0, 0) = 11;
    reference(2, 2, 0, 0) = 14;
    reference(3, 2, 0, 0) = 5;

    reference(0, 3, 0, 0) = 0;
    reference(1, 3, 0, 0) = 0;
    reference(2, 3, 0, 0) = 0;
    reference(3, 3, 0, 0) = 0;

    Matrix computed(reference.size(), reference.precision());

    reverseConvolutionDeltas(computed, filter, {1, 1}, deltas, {0, 0});

    if(reference != computed)
    {
        std::cout << " Matrix 2D Backward Convolution Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix 2D Backward Convolution Test Passed\n";
    }

    return reference == computed;
}

/*
    Test 2D strided backward convolution.

    [ 0 3 ] conv-back-2-2 [ 0 2 4 ] = [  0  9 ]
    [ 1 4 ]               [ 1 3 5 ]   [ 14 25 ]
    [ 2 5 ]

*/
bool test2dStridedBackwardConvolution()
{
    int n = 1;
    int c = 1;
    int h = 2;
    int w = 3;

    Matrix deltas(w, h, c, n);

    deltas(0, 0, 0, 0) = 0;
    deltas(1, 0, 0, 0) = 1;
    deltas(2, 0, 0, 0) = 2;
    deltas(0, 1, 0, 0) = 3;
    deltas(1, 1, 0, 0) = 4;
    deltas(2, 1, 0, 0) = 5;

    int k = 1;
    int r = 3;
    int s = 2;

    Matrix filter(s, r, c, k);

    filter(0, 0, 0, 0) = 0;
    filter(1, 0, 0, 0) = 1;
    filter(0, 1, 0, 0) = 2;
    filter(1, 1, 0, 0) = 3;
    filter(0, 2, 0, 0) = 4;
    filter(1, 2, 0, 0) = 5;

    Matrix reference(2, 2, 1, 1);

    reference(0, 0, 0, 0) = 0;
    reference(1, 0, 0, 0) = 0;

    reference(0, 1, 0, 0) = 0;
    reference(1, 1, 0, 0) = 0;

    Matrix computed(reference.size(), reference.precision());

    reverseConvolutionDeltas(computed, filter, {2, 2}, deltas, {0, 0});

    if(reference != computed)
    {
        std::cout << " Matrix 2D Strided Backward Convolution Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix 2D Strided Backward Convolution Test Passed\n";
    }

    return reference == computed;
}

/*
    Test 1D convolution gradient.

    [ 0 ] conv-grad [ 0 ] = [ 8 ]
    [ 1 ]           [ 1 ]   [ 5 ]
    [ 2 ]           [ 2 ]
    [ 3 ]


*/
bool test1dConvolutionGradient()
{
    int n = 1;
    int c = 1;
    int h = 1;
    int w = 4;

    Matrix input(w, h, c, n);

    input(0, 0, 0, 0) = 0;
    input(1, 0, 0, 0) = 1;
    input(2, 0, 0, 0) = 2;
    input(3, 0, 0, 0) = 3;

    int k = 1;
    int r = 1;
    int s = 3;

    Matrix deltas(s, r, c, k);

    deltas(0, 0, 0, 0) = 0;
    deltas(1, 0, 0, 0) = 1;
    deltas(2, 0, 0, 0) = 2;

    Matrix reference(2, 1, 1, 1);

    reference(0, 0, 0, 0) = 8;
    reference(1, 0, 0, 0) = 5;

    Matrix computed(reference.size(), reference.precision());

    reverseConvolutionGradients(computed, input, deltas, {1, 1}, {0, 0});

    if(reference != computed)
    {
        std::cout << " Matrix 1D Convolution Gradients Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix 1D Convolution Gradients Test Passed\n";
    }

    return reference == computed;
}

/*
    Test 2D convolution gradient.

    [ 0 3 6 ] conv-grad [ 0 2 ] = [ 43 25 ]
    [ 1 4 7 ]           [ 1 3 ]   [ 37 19 ]
    [ 2 5 8 ]

*/
bool test2dConvolutionGradient()
{
    int n = 1;
    int c = 1;
    int h = 3;
    int w = 3;

    Matrix input(w, h, c, n);

    input(0, 0, 0, 0) = 0;
    input(1, 0, 0, 0) = 1;
    input(2, 0, 0, 0) = 2;
    input(0, 1, 0, 0) = 3;
    input(1, 1, 0, 0) = 4;
    input(2, 1, 0, 0) = 5;
    input(0, 2, 0, 0) = 6;
    input(1, 2, 0, 0) = 7;
    input(2, 2, 0, 0) = 8;

    int k = 1;
    int r = 2;
    int s = 2;

    Matrix deltas(s, r, c, k);

    deltas(0, 0, 0, 0) = 0;
    deltas(1, 0, 0, 0) = 1;
    deltas(0, 1, 0, 0) = 2;
    deltas(1, 1, 0, 0) = 3;

    Matrix reference(2, 2, 1, 1);

    reference(0, 0, 0, 0) = 43;
    reference(1, 0, 0, 0) = 37;
    reference(0, 1, 0, 0) = 25;
    reference(1, 1, 0, 0) = 19;

    Matrix computed(reference.size(), reference.precision());

    reverseConvolutionGradients(computed, input, deltas, {1, 1}, {0, 0});

    if(reference != computed)
    {
        std::cout << " Matrix 2D Convolution Gradients Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix 2D Convolution Gradients Test Passed\n";
    }

    return reference == computed;
}

/*
    Test 2D reduce get positions.

    [ 5 4 8  13 ] reduce-max(2, 2) = [ 1 0 0 1 ]
    [ 1 0 9  12 ]                    [ 0 0 0 0 ]
    [ 2 6 10 14 ]                    [ 0 0 0 0 ]
    [ 7 3 11 15 ]                    [ 1 0 0 1 ]

*/
bool test2dReduceGetPositions()
{
    int n = 1;
    int c = 1;
    int h = 4;
    int w = 4;

    Matrix input(w, h, c, n);

    input(0, 0, 0, 0) = 5;
    input(1, 0, 0, 0) = 1;
    input(2, 0, 0, 0) = 2;
    input(3, 0, 0, 0) = 7;
    input(0, 1, 0, 0) = 4;
    input(1, 1, 0, 0) = 0;
    input(2, 1, 0, 0) = 6;
    input(3, 1, 0, 0) = 3;
    input(0, 2, 0, 0) = 8;
    input(1, 2, 0, 0) = 9;
    input(2, 2, 0, 0) = 10;
    input(3, 2, 0, 0) = 11;
    input(0, 3, 0, 0) = 13;
    input(1, 3, 0, 0) = 12;
    input(2, 3, 0, 0) = 14;
    input(3, 3, 0, 0) = 15;

    Matrix reference(w, h, n, c);

    reference(0, 0, 0, 0) = 1;
    reference(1, 0, 0, 0) = 0;
    reference(2, 0, 0, 0) = 0;
    reference(3, 0, 0, 0) = 0;
    reference(0, 1, 0, 0) = 0;
    reference(1, 1, 0, 0) = 1;
    reference(2, 1, 0, 0) = 0;
    reference(3, 1, 0, 0) = 0;
    reference(0, 2, 0, 0) = 0;
    reference(1, 2, 0, 0) = 0;
    reference(2, 2, 0, 0) = 1;
    reference(3, 2, 0, 0) = 0;
    reference(0, 3, 0, 0) = 0;
    reference(1, 3, 0, 0) = 0;
    reference(2, 3, 0, 0) = 0;
    reference(3, 3, 0, 0) = 1;

    Matrix computed(reference.size(), reference.precision());

    reduceGetPositions(computed, input, {2, 2}, matrix::Max());

    if(reference != computed)
    {
        std::cout << " Matrix 2D Reduce Get Positions Test Failed:\n";
        std::cout << "  result matrix " << computed.toString();
        std::cout << "  does not match reference matrix " << reference.toString();
    }
    else
    {
        std::cout << " Matrix 2D Reduce Get Positions Test Passed\n";
    }

    return reference == computed;
}

int main(int argc, char** argv)
{
    //lucius::util::enableAllLogs();

    std::cout << "Running matrix unit tests\n";

    bool passed = true;

    passed &= testSaveLoad();

    passed &= testMultiplySlice();
    passed &= testMultiply();
    passed &= testAddition();
    passed &= testScalarAddition();
    passed &= testReduce();
    passed &= test2dReduce();
    passed &= test2dReduce2();
    passed &= test4dReduce();
    passed &= testBroadcast();
    passed &= test2dBroadcast();
    passed &= testZeros();
    passed &= testReshape();
    passed &= testReshapeSlice();
    passed &= testCopy();
    passed &= testCopyBetweenPrecisions();
    passed &= testUniformRandom();
    passed &= testNormalRandom();

    passed &= test1dForwardConvolution();
    passed &= test2dForwardConvolution();
    passed &= test2dStridedForwardConvolution();
    passed &= test1dBackwardConvolution();
    passed &= test2dBackwardConvolution();
    passed &= test2dStridedBackwardConvolution();
    passed &= test1dConvolutionGradient();
    passed &= test2dConvolutionGradient();

    passed &= test2dReduceGetPositions();

    passed &= testForwardMaxPooling();
    passed &= testBackwardMaxPooling();

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


