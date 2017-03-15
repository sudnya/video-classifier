/*  \file   ControllerLayer.cpp
    \author Gregory Diamos
    \date   May 15, 2017
    \brief  The source for the ControllerLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/ControllerLayer.h>

#include <lucius/util/interface/PropertyTree.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace network
{

ControllerLayer::ControllerLayer()
{
    // intentionally blank
}

ControllerLayer::~ControllerLayer()
{
    // intentionally blank
}

ControllerLayer::ControllerLayer(const ControllerLayer& l)
: Layer(l), _controller(l._controller->clone())
{

}

ControllerLayer& ControllerLayer::operator=(const ControllerLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    _controller = l._controller->clone();

    return *this;
}

void ControllerLayer::setController(std::unique_ptr<Layer>&& l)
{
    _controller = std::move(l);
}

const Layer& ControllerLayer::getController() const
{
    assert(_controller);

    return *_controller;
}

Layer& ControllerLayer::getController()
{
    assert(_controller);

    return *_controller;
}

void ControllerLayer::saveLayer(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    auto& controllerProperties = properties["controller"];

    _controller->save(archive, controllerProperties);

    Layer::saveLayer(archive, properties);
}

void ControllerLayer::loadLayer(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    auto& controllerProperties = properties["controller"];

    _controller->load(archive, controllerProperties);

    Layer::loadLayer(archive, properties);
}

}

}

