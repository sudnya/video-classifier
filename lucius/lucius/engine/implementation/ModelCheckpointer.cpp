/*  \file   ModelCheckpointer.cpp
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ModelCheckpointer class.
*/

// Lucious Includes
#include <lucious/engine/interface/ModelCheckpointer.h>

#include <lucious/engine/interface/Engine.h>

#include <lucious/model/interface/Model.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>

namespace lucious
{

namespace engine
{

ModelCheckpointer::ModelCheckpointer(const std::string& path) : _path(path)
{

}

ModelCheckpointer::~ModelCheckpointer()
{

}

void ModelCheckpointer::epochCompleted(const Engine& engine)
{
    std::ofstream file(_path);

    if(!file.is_open())
    {
        throw std::runtime_error("Failed to open '" + _path + "' to save model.");
    }

    engine.getModel()->save(file);
}

}

}





