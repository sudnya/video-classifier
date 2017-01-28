

#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/assert.h>

namespace lucius
{
namespace parallel
{

/*! \brief A CUDA compatible list class with C++ standard library syntax and semantics. */
template <typename T>
class list
{
public:
    class Node
    {
    public:
        Node(const T& value, Node* previous, Node* next)
        : previous(previous), next(next), value(value)
        {

        }

    public:
        CUDA_DECORATOR bool isEnd() const
        {
            return next == nullptr;
        }

    public:
        Node* previous;
        Node* next;

        T value;
    };

    class const_iterator;

    class iterator
    {
    public:
        CUDA_DECORATOR iterator()
        : iterator(nullptr)
        {

        }

        CUDA_DECORATOR iterator(Node* n)
        : _node(n)
        {

        }

    public:
        CUDA_DECORATOR bool operator==(const const_iterator& i) const
        {
            return i._node == _node;
        }

        CUDA_DECORATOR bool operator==(const iterator& i) const
        {
            return i._node == _node;
        }

        CUDA_DECORATOR bool operator!=(const const_iterator& i) const
        {
            return i._node != _node;
        }

        CUDA_DECORATOR bool operator!=(const iterator& i) const
        {
            return i._node != _node;
        }

    public:
        CUDA_DECORATOR iterator& operator++()
        {
            if(!_node->isEnd())
            {
                _node = _node->next;
            }

            return *this;
        }

        CUDA_DECORATOR iterator& operator++(int)
        {
            iterator previous = *this;

            ++(*this);

            return previous;
        }

    public:
        CUDA_DECORATOR T& operator*() const
        {
            return _node->value;
        }

        CUDA_DECORATOR T* operator->() const
        {
            return &_node->value;
        }

    public:
        CUDA_DECORATOR Node* getNode() const
        {
            return _node;
        }

    private:
        Node* _node;
    };

    class const_iterator
    {
    public:
        CUDA_DECORATOR const_iterator()
        : const_iterator(nullptr)
        {

        }

        CUDA_DECORATOR const_iterator(const Node* n)
        : _node(n)
        {

        }

    public:
        CUDA_DECORATOR bool operator==(const const_iterator& i) const
        {
            return i._node == _node;
        }

        CUDA_DECORATOR bool operator==(const iterator& i) const
        {
            return i._node == _node;
        }

        CUDA_DECORATOR bool operator!=(const const_iterator& i) const
        {
            return i._node != _node;
        }

        CUDA_DECORATOR bool operator!=(const iterator& i) const
        {
            return i._node != _node;
        }

    public:
        CUDA_DECORATOR const_iterator& operator++()
        {
            if(!_node->isEnd())
            {
                _node = _node->next;
            }

            return *this;
        }

        CUDA_DECORATOR const_iterator& operator++(int)
        {
            const_iterator previous = *this;

            ++(*this);

            return previous;
        }

    public:
        CUDA_DECORATOR const T& operator*() const
        {
            return _node->value;
        }

        CUDA_DECORATOR const T* operator->() const
        {
            return &_node->value;
        }

    public:
        CUDA_DECORATOR const Node* getNode() const
        {
            return _node;
        }

    private:
        const Node* _node;
    };
public:
    CUDA_DECORATOR list()
    : _begin(nullptr), _end(nullptr), _size(0)
    {
        _addEndNode();
    }

    CUDA_DECORATOR list(const list<T>& l)
    {

    }

    CUDA_DECORATOR ~list()
    {
        clear();

        delete _begin;
    }

public:
    CUDA_DECORATOR size_t size() const
    {
        return _size;
    }

    CUDA_DECORATOR bool empty() const
    {
        return size() == 0;
    }

public:
    CUDA_DECORATOR void push_back(const T& value)
    {
        insert(end(), value);
    }

    CUDA_DECORATOR iterator insert(iterator position, const T& value)
    {
        auto* next = position.getNode();

        auto* previous = next->previous;

        auto* newNode = new Node(value, previous, next);

        ++_size;

        next->previous = newNode;

        if(previous != nullptr)
        {
            previous->next = newNode;
        }
        else
        {
            _begin = newNode;
        }

        return iterator(newNode);
    }

    CUDA_DECORATOR iterator erase(iterator position)
    {
        assert(size() > 0);

        auto* node = position.getNode();

        auto* previous = node->previous;
        auto* next     = node->next;

        assert(next != nullptr);

        delete node;

        next->previous = previous;

        if(previous != nullptr)
        {
            previous->next = next;
        }

        --_size;

        return iterator(next);
    }

    CUDA_DECORATOR void clear()
    {
        while(!empty())
        {
            erase(begin());
        }
    }

public:
    CUDA_DECORATOR iterator begin()
    {
        return iterator(_begin);
    }

    CUDA_DECORATOR const_iterator begin() const
    {
        return const_iterator(_begin);
    }

    CUDA_DECORATOR iterator end()
    {
        return iterator(_end);
    }

    CUDA_DECORATOR const_iterator end() const
    {
        return const_iterator(_end);
    }

private:
    CUDA_DECORATOR void _addEndNode()
    {
        auto* newNode = new Node(T(), nullptr, nullptr);

        _begin = newNode;
        _end   = newNode;
    }

private:
    Node*  _begin;
    Node*  _end;
    size_t _size;

};

} // parallel
} // lucius

