
#pragma once

// Standard Library Includes
#include <string>

namespace lucious
{

namespace util
{

class PropertyTreeImplementation;

/*! \brief A simple tree of dynamically typed properties, similar to JSON. */
class PropertyTree
{
public:
    typedef std::vector<PropertyTree> TreeVector;
    typedef TreeVector::iterator iterator;
    typedef TreeVector::const_iterator const_iterator;

public:
    explicit PropertyTree(const std::string& name);
    PropertyTree();
    ~PropertyTree();

public:
    PropertyTree(const PropertyTree& tree);

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
    std::string& key() const;

public:
    std::string& value() const;

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
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
    bool empty();
    size_t size() const;

public:
    void saveJson(std::ostream& json) const;

public:
    static PropertyTree loadJson(std::istream& json);

private:
    std::unique_ptr<PropertyTreeImplementation> _implementation;

};

}

}




