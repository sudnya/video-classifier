/*	\file   VideoDisplayResultProcessor.cpp
	\date   Saturday August 10, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the VideoDisplayResultProcessor class.
*/

#include <lucius/results/interface/VideoDisplayResultProcessor.h>
#include <lucius/results/interface/LabelResult.h>
#include <lucius/results/interface/ResultVector.h>

#include <lucius/video/interface/ImageLibraryInterface.h>

namespace lucius
{

namespace results
{

VideoDisplayResultProcessor::~VideoDisplayResultProcessor()
{
    // intentionally blank
}

void VideoDisplayResultProcessor::process(const ResultVector& results)
{
    auto& mostRecentResult = static_cast<LabelResult&>(*results.back());

    video::ImageLibraryInterface::addTextToStatusBar(mostRecentResult.label);
    video::ImageLibraryInterface::waitForKey();
    video::ImageLibraryInterface::deleteWindow();
}

std::string VideoDisplayResultProcessor::toString() const
{
    return "";
}

}

}

