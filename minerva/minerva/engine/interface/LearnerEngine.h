/*    \file   LearnerEngine.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the LearnerEngine class.
*/

#pragma once

// Lucious Includes
#include <lucious/engine/interface/Engine.h>

// Forward Declarations
namespace lucious { namespace engine { class Learner; } }

namespace lucious
{

namespace engine
{

class LearnerEngine : public Engine
{
public:
    LearnerEngine();
    virtual ~LearnerEngine();

public:
    LearnerEngine(const LearnerEngine&) = delete;
    LearnerEngine& operator=(const LearnerEngine&) = delete;

private:
    virtual void closeModel();

private:
    virtual ResultVector runOnBatch(Matrix&& input, Matrix&& reference);

    virtual bool requiresLabeledData() const;


};

}

}




