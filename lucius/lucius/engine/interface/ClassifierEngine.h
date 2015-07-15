/*    \file   ClassifierEngine.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ClassifierEngine class.
*/

#pragma once

// Lucius Includes
#include <lucius/engine/interface/Engine.h>

// Standard Library Includes
#include <map>

namespace lucius
{

namespace engine
{

class ClassifierEngine : public Engine
{
public:
    ClassifierEngine();
    virtual ~ClassifierEngine();

public:
    void setUseLabeledData(bool useIt);

protected:
    virtual ResultVector runOnBatch(Matrix&& inputs, Matrix&& reference);
    virtual bool requiresLabeledData() const;

private:
    bool _shouldUseLabeledData;

};

}

}




