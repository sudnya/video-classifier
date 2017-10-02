/*  \file   PassManager.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The source file for the PassManager class.
*/

// Lucius Includes
#include <lucius/optimization/interface/PassManager.h>

#include <lucius/optimization/interface/Pass.h>

#include <lucius/analysis/interface/AnalysisFactory.h>
#include <lucius/analysis/interface/Analysis.h>

#include <lucius/ir/interface/Module.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Program.h>

// Standard Library Includes
#include <list>
#include <map>

namespace lucius
{
namespace optimization
{

using Analysis = analysis::Analysis;
using AnalysisFactory = analysis::AnalysisFactory;
using PassList = std::list<std::unique_ptr<Pass>>;
using AnalysisMap = std::map<std::string, std::unique_ptr<Analysis>>;

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
        // schedule passes
        // TODO: group passes by shared analyses

        // Execute passes
        for(auto& pass : _passes)
        {
            auto requiredAnalyses = pass->getRequiredAnalyses();

            for(auto& analysisName : requiredAnalyses)
            {
                if(_analyses.count(analysisName) == 0)
                {
                    auto newAnalysis = _analyses.emplace(std::make_pair(analysisName,
                        AnalysisFactory::create(analysisName))).first;

                    newAnalysis->second->runOnFunction(f);
                }
            }

            // TODO: free no longer needed analyses

            pass->setManager(_manager);
            pass->runOnFunction(f);
        }

        _analyses.clear();
    }

    void addPass(std::unique_ptr<Pass>&& pass)
    {
        _passes.emplace_back(std::move(pass));
    }

    Pass* getPass(const std::string& name)
    {
        for(auto& pass : _passes)
        {
            if(pass->name() == name)
            {
                return pass.get();
            }
        }

        return nullptr;
    }

    const Pass* getPass(const std::string& name) const
    {
        for(auto& pass : _passes)
        {
            if(pass->name() == name)
            {
                return pass.get();
            }
        }

        return nullptr;
    }

    Analysis* getAnalysis(const std::string& name)
    {
        auto newAnalysis = _analyses.find(name);

        if(newAnalysis == _analyses.end())
        {
            return nullptr;
        }

        return newAnalysis->second.get();
    }

    const Analysis* getAnalysis(const std::string& name) const
    {
        auto newAnalysis = _analyses.find(name);

        if(newAnalysis == _analyses.end())
        {
            return nullptr;
        }

        return newAnalysis->second.get();
    }

private:
    PassManager* _manager;

private:
    PassList _passes;
    AnalysisMap _analyses;

};

PassManager::PassManager()
: _implementation(std::make_unique<PassManagerImplementation>(this))
{

}

PassManager::~PassManager()
{

}

void PassManager::runOnFunction(ir::Function& function)
{
    _implementation->runOnFunction(function);
}

void PassManager::runOnModule(ir::Module& module)
{
    for(auto& function : module)
    {
        runOnFunction(function);
    }
}


void PassManager::runOnProgram(ir::Program& program)
{
    runOnModule(program.getModule());
}

void PassManager::addPass(std::unique_ptr<Pass>&& pass)
{
    _implementation->addPass(std::move(pass));
}

Pass* PassManager::getPass(const std::string& name)
{
    return _implementation->getPass(name);
}

const Pass* PassManager::getPass(const std::string& name) const
{
    return _implementation->getPass(name);
}

Analysis* PassManager::getAnalysis(const std::string& name)
{
    return _implementation->getAnalysis(name);
}

const Analysis* PassManager::getAnalysis(const std::string& name) const
{
    return _implementation->getAnalysis(name);
}

} // namespace optimization
} // namespace lucius







