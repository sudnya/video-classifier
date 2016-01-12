/*  \file   PropertyTree.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the PropertyTree class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

namespace lucius
{

namespace util
{

class PropertyTreeImplementation;
class IteratorImplementation;
class ConstIteratorImplementation;

/*! \brief A simple tree of dynamically typed properties, similar to JSON. */
class PropertyTree
{
public:
    explicit PropertyTree(const std::string& name);
    PropertyTree();
    ~PropertyTree();

public:
    PropertyTree(const PropertyTree& tree);
    PropertyTree(PropertyTree&& tree);

public:
    template<typename T>
    PropertyTree& operator=(const T& t);

    PropertyTree& operator=(const std::string& value);
    PropertyTree& operator=(const PropertyTree& tree);
    PropertyTree& operator=(PropertyTree&& tree);

public:
    PropertyTree& get(const std::string& field);
    const PropertyTree& get(const std::string& field) const;

    PropertyTree& get();
    const PropertyTree& get() const;

    template<typename T>
    T get(const std::string& field) const;

    template<typename T>
    T get(const std::string& field, const T& defaultValue) const;

public:
    template<typename T>
    T key() const;

public:
    bool exists(const std::string& ) const;

public:
    std::string& key();
    const std::string& key() const;

public:
    const std::string& value() const;

public:
    std::string& path() const;

public:
    operator std::string() const;

public:
    template<typename T>
    PropertyTree& operator[](const T& key);
    template<typename T>
    const PropertyTree& operator[](const T& key) const;

    PropertyTree& operator[](const std::string& key);
    const PropertyTree& operator[](const std::string& key) const;

public:
    void add(const PropertyTree& child);
    void addListElement(const PropertyTree& child);

public:
    bool empty() const;
    size_t size() const;

public:
    bool isList() const;

public:
    bool operator<(const PropertyTree& ) const;

public:
    void saveJson(std::ostream& json) const;

public:
    std::string jsonString() const;

public:
    static PropertyTree loadJson(std::istream& json);

public:
    class Begin {};
    class End {};

public:
    class const_iterator;

    class iterator
    {
    public:
        iterator();
        iterator(PropertyTree& tree, Begin);
        iterator(PropertyTree& tree, End);
        iterator(const iterator& );
        ~iterator();

    public:
        iterator& operator=(const iterator& );

    public:
        iterator& operator++();

    public:
        const PropertyTree& operator*() const;
        const PropertyTree* operator->() const;

    public:
        bool operator==(const iterator&);
        bool operator==(const const_iterator&);

        bool operator!=(const iterator&);
        bool operator!=(const const_iterator&);

    private:
        bool _isList() const;

    private:
        std::unique_ptr<IteratorImplementation> _implementation;

    private:
        friend class const_iterator;
    };

public:
    class const_iterator
    {
    public:
        const_iterator(const PropertyTree& tree, Begin);
        const_iterator(const PropertyTree& tree, End);
        const_iterator();
        const_iterator(const iterator& );
        const_iterator(const const_iterator& );
        ~const_iterator();

    public:
        const_iterator& operator=(const const_iterator& );
        const_iterator& operator=(const iterator& );

    public:
        const_iterator& operator++();

    public:
        const PropertyTree& operator*() const;
        const PropertyTree* operator->() const;

    public:
        bool operator==(const iterator&);
        bool operator==(const const_iterator&);

        bool operator!=(const iterator&);
        bool operator!=(const const_iterator&);

    private:
        bool _isList() const;

    private:
        std::unique_ptr<ConstIteratorImplementation> _implementation;

    private:
        friend class iterator;
    };

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

private:
    std::unique_ptr<PropertyTreeImplementation> _implementation;

private:
    friend class IteratorImplementation;
    friend class ConstIteratorImplementation;
    friend class PropertyTreeImplementation;

};

}

}

#include <lucius/util/implementation/PropertyTree-inl.h>

