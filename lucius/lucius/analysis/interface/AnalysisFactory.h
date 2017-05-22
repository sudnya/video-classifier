/*  \file   AnalysisFactory.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the AnalysisFactory class.
*/

#pragma once

// forward declarations
namespace lucius { namespace analysis { class Analysis; } }

namespace lucius
{
namespace analysis
{

/*! \brief A class used to construct analyses. */
class AnalysisFactory
{
public:
    static std::unique_ptr<Analysis> create(const std::string& );
};

} // namespace analysis
} // namespace lucius







