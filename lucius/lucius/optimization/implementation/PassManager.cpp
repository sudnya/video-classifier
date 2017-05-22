/*  \file   PassManager.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the PassManager class.
*/

// Lucius Includes
#include <lucius/optimization/interface/PassManager.h>

#include <lucius/analysis/interface/AnalysisFactory.h>

// Standard Library Includes
#include <list>
#include <map>

namespace lucius
{
namespace optimization
{

typedef std::list<std::unique_ptr<Pass>> PassList;
typedef std::map<std::string, std::unique_ptr<Analysis>> AnalysisMap;

class PassManagerImplementation
{
public:
    PassManagerImplementation(PassManager* manager)
    : _manager(manager)
    {

    }

public:
    void runOnFunction(ir::Function& f)
    {
        AnalysisMap analyses;

        // schedule passes
        // TODO: group passes by shared analyses

        // Execute passes
        for(auto& pass : _passes)
        {
            auto requiredAnalyses = pass->getRequiredAnalyses();

            for(auto& analysisName : requiredAnalyses)
            {
                if(analyses.count(analysisName) == 0)
                {
                    auto newAnalysis = analyses.emplace(std::make_pair(analysisName,
                        AnalysisFactory::create(analysisName)));

                    newAnalysis->second->runOnFunction(f);
                }
            }

            // TODO: free no longer needed analyses

            pass->setManager(_manager);
            pass->runOnFunction(f);
        }
    }

private:
    PassManager* _manager;

private:
    PassList _passes;

};

PassManager::PassManager()
: _implementation(std::make_unique<PassManagerImplementation>())
{

}

PassManager::~PassManager()
{

}

void PassManager::runOnFunction(ir::Function& f)
{
    _implementation->runOnFunction(f);
}

void PassManager::runOnModule(ir::Module& m)
{
    for(auto& function : m)
    {
        runOnFunction(f);
    }
}

void PassManager::addPass(std::unique_ptr<Pass>&& pass)
{
    _implementation->addPass(std::move(pass));
}

} // namespace optimization
} // namespace lucius







