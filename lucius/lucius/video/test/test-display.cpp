/*! \file   test-display.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Tuesday November 19, 2013
	\brief  A unit test for image and video display.
*/

// Lucius Includes
#include <lucius/video/interface/Image.h>
#include <lucius/video/interface/Video.h>
#include <lucius/video/interface/ImageLibraryInterface.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/ArgumentParser.h>

// Standard Library Includes
#include <stdexcept>

namespace lucius
{

namespace video
{

static void displayImage(const std::string& inputPath, size_t xPixels,
	size_t yPixels, size_t colors, const std::string& text)
{
	Image image(inputPath);

	image.load();
	
	auto downsampledImage = image.downsample(xPixels, yPixels, colors);
	
	downsampledImage.displayOnScreen();
	downsampledImage.addTextToDisplay(text);	

	//ImageLibraryInterface::waitForKey();

}

static void displayVideo(const std::string& inputPath, size_t xPixels,
	size_t yPixels, size_t colors, const std::string& text)
{
	assertM(false, "Not implemented.");
}

static void runTest(const std::string& inputPath, size_t xPixels,
	size_t yPixels, size_t colors, const std::string& displayText)
{
	if(Image::isPathAnImage(inputPath))
	{
		displayImage(inputPath, xPixels, yPixels, colors, displayText);
	}
	else if(Video::isPathAVideo(inputPath))
	{
		displayVideo(inputPath, xPixels, yPixels, colors, displayText);
	}
	else
	{
		throw std::runtime_error("Input path '" + inputPath +
			"' is neither an image or a video.");
	}
	
	std::cout << "Test Passed\n";
}

}

}


int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);
    
    bool verbose = false;
    std::string loggingEnabledModules;

	std::string imagePath;
	std::string displayText;	

	size_t xPixels = 0;
	size_t yPixels = 0;
	size_t colors  = 0;

    parser.description("The lucius video and image display test.");

    parser.parse("-i", "--input-path", imagePath, "images/cat.jpg",
        "The input video or image path to display..");
    parser.parse("-x", "--x-pixels", xPixels, 64,
        "The number of X pixels to consider from the input image.");
	parser.parse("-y", "--y-pixels", yPixels, 64,
		"The number of Y pixels to consider from the input image");
	parser.parse("-c", "--colors", colors, 3,
		"The number of color components (e.g. RGB) to consider from the input image");
	parser.parse("-t", "--text", displayText, "Some Example Text",
		"The text to display on the generated window.");
    parser.parse("-L", "--log-module", loggingEnabledModules, "",
		"Print out log messages during execution for specified modules "
		"(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-v", "--verbose", verbose, false,
        "Print out all log messages during execution");

	parser.parse();

    if(verbose)
	{
		lucius::util::enableAllLogs();
	}
	else
	{
		lucius::util::enableSpecificLogs(loggingEnabledModules);
	}
    
    lucius::util::log("TestDisplay") << "Test begins\n";
    
    try
    {
        lucius::video::runTest(imagePath, xPixels, yPixels, colors, displayText);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Display Test Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}


