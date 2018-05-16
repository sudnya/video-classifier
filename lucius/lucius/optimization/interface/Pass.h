/*  \file   Pass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the Pass class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace analysis { class Analysis; } }

namespace lucius { namespace ir { class Function; } }
namespace lucius { namespace ir { class Program;  } }
namespace lucius { namespace ir { class Module;   } }

namespace lucius { namespace optimization { class PassManager; } }

// Standard Library Includes
#include <set>
#include <string>

namespace lucius
{
namespace optimization
{

using StringSet = std::set<std::string>;
using Function = ir::Function;
using Module = ir::Module;
using Program = ir::Program;
using Analysis = analysis::Analysis;

/*! \brief A class representing an optimization pass. */
class Pass
{
public:
    enum PassType
    {
        FunctionPassType,
        ModulePassType,
        ProgramPassType,
        InvalidPassType
    };

public:
    Pass();
    explicit Pass(const std::string& name, PassType type);
    virtual ~Pass();

public:
    void setManager(PassManager* manager);

public:
    PassManager* getManager();

public:
    const Analysis* getAnalysisForFunction(const ir::Function& function,
        const std::string& name) const;
    Analysis* getAnalysisForFunction(const ir::Function& function, const std::string& name);

public:
    virtual StringSet getRequiredAnalyses() const = 0;

public:
    const std::string& name() const;

public:
    bool isFunctionPass() const;
    bool isModulePass() const;
    bool isProgramPass() const;

private:
    PassManager* _manager;

private:
    std::string _name;
    PassType _type;
};

/*! \brief A pass that runs on one function at a time. */
class FunctionPass : public Pass
{
public:
    explicit FunctionPass(const std::string& name);
    virtual ~FunctionPass();

public:
    /*! \brief Apply the pass to the specified function. */
    virtual void runOnFunction(Function& ) = 0;

};

/*! \brief A pass that runs on one module at a time. */
class ModulePass : public Pass
{
public:
    explicit ModulePass(const std::string& name);
    virtual ~ModulePass();

public:
    /*! \brief Apply the pass to the specified module. */
    virtual void runOnModule(Module& ) = 0;

};

/*! \brief A pass that runs on one program at a time. */
class ProgramPass : public Pass
{
public:
    explicit ProgramPass(const std::string& name);
    virtual ~ProgramPass();

public:
    /*! \brief Apply the pass to the specified module. */
    virtual void runOnProgram(Program& ) = 0;

};

} // namespace optimization
} // namespace lucius





