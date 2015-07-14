/*	\file   PropertyTree.h
	\date   Saturday August 10, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the PropertyTree class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <set>

namespace lucious
{

namespace util
{

class PropertyTreeImplementation;

/*! \brief A simple tree of dynamically typed properties, similar to JSON. */
class PropertyTree
{
public:
    typedef std::set<PropertyTree> TreeSet;
    typedef TreeSet::iterator iterator;
    typedef TreeSet::const_iterator const_iterator;

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

public:
    template<typename T>
    T get(const std::string& field) const;

public:
    template<typename T>
    T key() const;

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

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
    bool empty() const;
    size_t size() const;

public:
    bool operator<(const PropertyTree& ) const;

public:
    void saveJson(std::ostream& json) const;

public:
    std::string jsonString() const;

public:
    static PropertyTree loadJson(std::istream& json);

private:
    std::unique_ptr<PropertyTreeImplementation> _implementation;

private:
    friend class PropertyTreeImplementation;

};

}

}

#include <lucious/util/implementation/PropertyTree-inl.h>

