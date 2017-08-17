/*  \file   Loader.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Loader class.
*/

// Lucius Includes
#include <lucius/ir/interface/Loader.h>
#include <lucius/ir/interface/Program.h>

#include <lucius/util/interface/TarArchive.h>
#include <lucius/util/interface/PropertyTree.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{

namespace ir
{

Loader::Loader(Context& context)
: _context(&context)
{

}

static Program loadProgram(util::InputTarArchive& tar, const util::PropertyTree& tree,
    const util::PropertyTree& metadata, Context* context)
{
    Program program(*context);

    assertM(false, "Not implemented.");

    return program;
}

Program Loader::load(std::istream& stream)
{
    util::InputTarArchive tar(stream);

    std::stringstream metadataText;

    tar.extractFile("metadata.txt", metadataText);

    auto metadataObject = util::PropertyTree::loadJson(metadataText);

    auto& metadata = metadataObject["metadata"];

    auto& programDescription = metadata["program"];

    return loadProgram(tar, programDescription, metadata, _context);
}

} // namespace ir
} // namespace lucius

