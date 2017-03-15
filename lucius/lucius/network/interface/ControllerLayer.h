/*  \file   ControllerLayer.h
    \author Gregory Diamos
    \date   May 15, 2017
    \brief  The interface for the ControllerLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

namespace lucius
{

namespace network
{

/*! \brief A layer abstraction that contains a single sublayer. */
class ControllerLayer : public Layer
{
public:
    ControllerLayer();
    virtual ~ControllerLayer();

public:
    ControllerLayer(const ControllerLayer& );
    ControllerLayer& operator=(const ControllerLayer&);

public:
    void setController(std::unique_ptr<Layer>&& l);
    const Layer& getController() const;
    Layer& getController();

public:
    /*! \brief Save the layer to the tar file and header. */
    virtual void saveLayer(util::OutputTarArchive& archive, util::PropertyTree& properties) const;
    /*! \brief Intialize the layer from the tar file and header. */
    virtual void loadLayer(util::InputTarArchive& archive, const util::PropertyTree& properties);

protected:
    std::unique_ptr<Layer> _controller;

};

}

}

