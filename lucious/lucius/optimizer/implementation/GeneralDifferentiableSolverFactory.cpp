/*    \file   GeneralDifferentiableSolverFactory.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the GeneralDifferentiableSolverFactory class.
*/

// Lucious Includes
#include <lucious/optimizer/interface/GeneralDifferentiableSolverFactory.h>
#include <lucious/optimizer/interface/GPULBFGSSolver.h>
#include <lucious/optimizer/interface/NesterovAcceleratedGradientSolver.h>

#include <lucious/util/interface/Knobs.h>

namespace lucious
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

