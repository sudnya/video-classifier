/*  \file   EngineObserver.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the EngineObserver class.
*/

#pragma once

// Forward Declarations
namespace lucious { namespace engine { class Engine; } }

namespace lucious
{

namespace engine
{

/*! \brief An interface to an engine observer. */
class EngineObserver
{
public:
    EngineObserver();
    virtual ~EngineObserver();

public:
    /*! \brief Called after each iteration completed by the engine. */
    virtual void epochCompleted(const Engine& engine) = 0;

};

}

}



