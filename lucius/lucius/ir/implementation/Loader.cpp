/*  \file   Loader.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Loader class.
*/

// Lucius Includes
#include <lucius/ir/interface/Loader.h>

namespace lucius
{

namespace ir
{

Loader::Loader(Context& context)
: _context(context)
{

}

Program Loader::load(std::istream& stream)
{
    util::InputTarArchive tar(stream);

    std::stringstream metadataText;

    tar.extractFile("metadata.txt", metadataText);

    auto metadataObject = util::PropertyTree::loadJson(metadataText);

    auto& metadata = metadataObject["metadata"];

    auto& programDescription = metadata["program"];

    return loadProgram(tar, programDescription, metadata);
}

} // namespace ir
} // namespace lucius

