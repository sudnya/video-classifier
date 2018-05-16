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

#include <lucius/util/interface/debug.h>

#include <lucius/util/interface/PointerCasts.h>

// Standard Library Includes
#include <list>
#include <map>
#include <cassert>

namespace lucius
{
namespace optimization
{

using Analysis = analysis::Analysis;
using AnalysisFactory = analysis::AnalysisFactory;
using FunctionPassList = std::list<std::unique_ptr<FunctionPass>>;
using ModulePassList = std::list<std::unique_ptr<ModulePass>>;
using ProgramPassList = std::list<std::unique_ptr<ProgramPass>>;

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
        for(auto& pass : _functionPasses)
        {
            _setupAnalysesForFunctionPass(f, *pass);

            pass->setManager(_manager);

            util::log("PassManager") << "Running pass " << pass->name()
                << " on function " << f.name() << "\n";
            pass->runOnFunction(f);
        }

        _analyses.clear();
    }

    void runOnModule(ir::Module& m)
    {
        // schedule passes
        // TODO: group passes by shared analyses

        // Execute passes
        for(auto& pass : _modulePasses)
        {
            _setupAnalysesForModulePass(m, *pass);

            pass->setManager(_manager);

            util::log("PassManager") << "Running pass " << pass->name()
                << " on module " << m.name() << "\n";
            pass->runOnModule(m);
        }

        _analyses.clear();

        for(auto& function : m)
        {
            runOnFunction(function);
        }
    }

    void runOnProgram(ir::Program& p)
    {
        // schedule passes
        // TODO: group passes by shared analyses

        // Execute passes
        for(auto& pass : _programPasses)
        {
            _setupAnalysesForProgramPass(p, *pass);

            pass->setManager(_manager);

            util::log("PassManager") << "Running pass " << pass->name()
                << " on program " << p.name() << "\n";
            pass->runOnProgram(p);
        }

        _analyses.clear();

        runOnModule(p.getModule());
    }

    void addPass(std::unique_ptr<Pass>&& pass)
    {
        assert(pass);

        if(pass->isProgramPass())
        {
            _programPasses.emplace_back(util::unique_pointer_cast<ProgramPass>(std::move(pass)));
        }
        else if(pass->isModulePass())
        {
            _modulePasses.emplace_back(util::unique_pointer_cast<ModulePass>(std::move(pass)));
        }
        else if(pass->isFunctionPass())
        {
            _functionPasses.emplace_back(util::unique_pointer_cast<FunctionPass>(std::move(pass)));
        }
    }

    Pass* getPass(const std::string& name)
    {
        for(auto& pass : _functionPasses)
        {
            if(pass->name() == name)
            {
                return pass.get();
            }
        }
        for(auto& pass : _modulePasses)
        {
            if(pass->name() == name)
            {
                return pass.get();
            }
        }
        for(auto& pass : _programPasses)
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
        for(auto& pass : _functionPasses)
        {
            if(pass->name() == name)
            {
                return pass.get();
            }
        }
        for(auto& pass : _modulePasses)
        {
            if(pass->name() == name)
            {
                return pass.get();
            }
        }
        for(auto& pass : _programPasses)
        {
            if(pass->name() == name)
            {
                return pass.get();
            }
        }

        return nullptr;
    }

    Analysis* getAnalysisForFunction(const ir::Function& function, const std::string& name)
    {
        auto analyses = _analyses.find(function);

        if(analyses == _analyses.end())
        {
            return nullptr;
        }

        auto newAnalysis = analyses->second.find(name);

        if(newAnalysis == analyses->second.end())
        {
            return nullptr;
        }

        return newAnalysis->second.get();
    }

    const Analysis* getAnalysisForFunction(const ir::Function& function,
        const std::string& name) const
    {
        auto analyses = _analyses.find(function);

        if(analyses == _analyses.end())
        {
            return nullptr;
        }

        auto newAnalysis = analyses->second.find(name);

        if(newAnalysis == analyses->second.end())
        {
            return nullptr;
        }

        return newAnalysis->second.get();
    }

private:
    void _setupAnalysesForFunctionPass(const ir::Function& function, const Pass& pass)
    {
        auto requiredAnalyses = pass.getRequiredAnalyses();

        for(auto& analysisName : requiredAnalyses)
        {
            auto& analyses = _analyses[function];

            if(analyses.count(analysisName) == 0)
            {
                auto createdAnalysis = AnalysisFactory::create(analysisName);
                assert(createdAnalysis);

                auto newAnalysis = analyses.emplace(std::make_pair(analysisName,
                    std::move(createdAnalysis))).first;

                newAnalysis->second->runOnFunction(function);
            }
        }

        // TODO: free no longer needed analyses
    }

    void _setupAnalysesForModulePass(const ir::Module& m, const Pass& pass)
    {
        for(auto& function : m)
        {
            _setupAnalysesForFunctionPass(function, pass);
        }
    }

    void _setupAnalysesForProgramPass(const ir::Program& p, const Pass& pass)
    {
        _setupAnalysesForModulePass(p.getModule(), pass);
    }


private:
    PassManager* _manager;

private:
    FunctionPassList _functionPasses;
    ModulePassList   _modulePasses;
    ProgramPassList  _programPasses;

private:
    using AnalysisMap = std::map<std::string, std::unique_ptr<Analysis>>;
    using FunctionAnalysisMap = std::map<ir::Function, AnalysisMap>;

private:
    FunctionAnalysisMap _analyses;

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
    _implementation->runOnModule(module);
}

void PassManager::runOnProgram(ir::Program& program)
{
    _implementation->runOnProgram(program);
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

const Analysis* PassManager::getAnalysisForFunction(const ir::Function& f,
    const std::string& name) const
{
    return _implementation->getAnalysisForFunction(f, name);
}

Analysis* PassManager::getAnalysisForFunction(const ir::Function& f, const std::string& name)
{
    return _implementation->getAnalysisForFunction(f, name);
}

} // namespace optimization
} // namespace lucius







