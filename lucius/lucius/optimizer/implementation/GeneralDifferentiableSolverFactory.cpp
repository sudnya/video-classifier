/*    \file   GeneralDifferentiableSolverFactory.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the GeneralDifferentiableSolverFactory class.
*/

// Lucius Includes
#include <lucius/optimizer/interface/GeneralDifferentiableSolverFactory.h>
#include <lucius/optimizer/interface/GPULBFGSSolver.h>
#include <lucius/optimizer/interface/NesterovAcceleratedGradientSolver.h>

#include <lucius/util/interface/Knobs.h>

namespace lucius
{

namespace optimizer
{

GeneralDifferentiableSolver* GeneralDifferentiableSolverFactory::create(const std::string& name)
{
    GeneralDifferentiableSolver* solver = nullptr;

    if("LimitedMemoryBroydenFletcherGoldfarbShannoSolver" == name ||
        "LBFGSSolver" == name)
    {
        if(GPULBFGSSolver::isSupported())
        {
            solver = new GPULBFGSSolver;
        }
    }
    else if("NesterovAcceleratedGradientSolver" == name || "NAGSolver" == name)
    {
        solver = new NAGSolver;
    }

    return solver;
}

static std::string getSolverName()
{
    return util::KnobDatabase::getKnobValue("GeneralDifferentiableSolver::Type",
        "LBFGSSolver");
}

GeneralDifferentiableSolver* GeneralDifferentiableSolverFactory::create()
{
    auto solverName = getSolverName();

    return create(solverName);
}

double GeneralDifferentiableSolverFactory::getMemoryOverheadForSolver(const std::string& name)
{
    if("LimitedMemoryBroydenFletcherGoldfarbShannoSolver" == name ||
        "LBFGSSolver" == name)
    {
        if(GPULBFGSSolver::isSupported())
        {
            return GPULBFGSSolver::getMemoryOverhead();
        }
    }
    else if("NesterovAcceleratedGradientSolver" == name || "NAGSolver" == name)
    {
        return NAGSolver::getMemoryOverhead();
    }

    return 2.0;
}

double GeneralDifferentiableSolverFactory::getMemoryOverheadForSolver()
{
    auto solverName = getSolverName();

    return getMemoryOverheadForSolver(solverName);
}

GeneralDifferentiableSolverFactory::StringVector GeneralDifferentiableSolverFactory::enumerate()
{
    return
    {
        "NesterovAcceleratedGradientSolver",
        "LimitedMemoryBroydenFletcherGoldfarbShannoSolver"
    };
}

}

}

