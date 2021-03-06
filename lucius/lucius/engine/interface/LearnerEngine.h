/*  \file   LearnerEngine.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the LearnerEngine class.
*/

#pragma once

// Lucius Includes
#include <lucius/engine/interface/Engine.h>

namespace lucius
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
    virtual ResultVector runOnBatch(const Bundle& bundle);

    virtual bool requiresLabeledData() const;


};

}

}




