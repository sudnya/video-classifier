/*  \file   ModelCheckpointer.cpp
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ModelCheckpointer class.
*/

// Lucius Includes
#include <lucius/engine/interface/ModelCheckpointer.h>

#include <lucius/engine/interface/Engine.h>

#include <lucius/model/interface/Model.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>

namespace lucius
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





