/*    \file   NesterovAcceleratedGradientSolver.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the NesterovAcceleratedGradientSolver class.
*/


#pragma once

// Lucius Includes
#include <lucius/optimizer/interface/GeneralDifferentiableSolver.h>

// Standard Library Includes
#include <memory>

namespace lucius
{

namespace optimizer
{

class NesterovAcceleratedGradientSolver : public GeneralDifferentiableSolver
{
public:
    NesterovAcceleratedGradientSolver();
    virtual ~NesterovAcceleratedGradientSolver();

public:
    virtual double solve(MatrixVector& inputs,
        const CostAndGradientFunction& callback);

public:
    static double getMemoryOverhead();

private:
    std::unique_ptr<MatrixVector> _velocity;

    double _runningExponentialCostSum;
    size_t _iterationsSoFar;

private:
    double  _learningRate;
    double  _momentum;
    double  _annealingRate;
    double  _maxGradNorm;
    size_t  _iterations;


};

typedef NesterovAcceleratedGradientSolver NAGSolver;

}

}


