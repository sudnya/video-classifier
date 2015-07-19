/*  \file   ModelCheckpointer.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ModelCheckpointer class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Lucius Includes
#include <lucius/engine/interface/EngineObserver.h>

namespace lucius
{

namespace engine
{

/*! \brief An interface to an engine observer. */
class ModelCheckpointer : public EngineObserver
{
public:
    ModelCheckpointer(const std::string& path);
    virtual ~ModelCheckpointer();

public:
    /*! \brief Called after each iteration completed by the engine. */
    virtual void epochCompleted(Engine& engine);

private:
    std::string _path;
};

}

}




