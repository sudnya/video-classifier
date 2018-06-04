/*  \file   OperationFactory.h
    \author Gregory Diamos
    \date   August 4, 2017
    \brief  The header file for the OperationFactory class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for creating operations. */
class OperationFactory
{
public:
    class BackPropagationOperandDescriptor;

public:
    /*! \brief Create the back propagation version of another operation. */
    static Operation createBackPropagationOperation(const Operation& op);

    /*! \brief Get the set of saved operands needed for back propagation for this operation. */
    static BackPropagationOperandDescriptor getBackPropagationOperandDescriptor(
        const Operation& op);

public:
    class BackPropagationOperandDescriptor
    {
    public:
        using IndexVector = std::vector<size_t>;

    public:
        BackPropagationOperandDescriptor(bool needsSavedOutput, const IndexVector&);

    public:
        bool needsSavedOutput() const;

    public:
        using iterator = IndexVector::iterator;
        using const_iterator = IndexVector::const_iterator;

    public:
        iterator begin();
        const_iterator begin() const;

        iterator end();
        const_iterator end() const;

    private:
        bool _needsSavedOutput;

    private:
        IndexVector _savedIndices;

    };
};

} // namespace ir
} // namespace lucius



