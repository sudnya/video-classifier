/*  \file   PassManager.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the PassManager class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class Function; } }
namespace lucius { namespace ir { class Module;   } }
namespace lucius { namespace ir { class Program;  } }

namespace lucius { namespace optimization { class Pass; } }

namespace lucius
{
namespace optimization
{

class PassManagerImplementation;

/*! \brief A class that orchestrates optimization of modules. */
class PassManager
{
public:
    PassManager();
    ~PassManager();

public:
    PassManager(const PassManager& ) = delete;
    PassManager& operator=(const PassManager& ) = delete;

public:
    void runOnFunction(ir::Function& );
    void runOnModule(ir::Module& );
    void runOnProgram(ir::Program& );

public:
    /*! \brief Add a pass to the manager, manager takes ownership. */
    void addPass(std::unique_ptr<Pass>&&);

public:
    /*! \brief An interface to get a pass by name. Manager retains ownership. */
          Pass* getPass(const std::string& name);
    const Pass* getPass(const std::string& name) const;

private:
    std::unique_ptr<PassManagerImplementation> _implementation;
};

} // namespace optimization
} // namespace lucius






