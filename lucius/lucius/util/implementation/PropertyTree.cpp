/*  \file   PropertyTree.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the PropertyTree class.
*/

// Lucius Includes
#include <lucius/util/interface/PropertyTree.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/string.h>

// Standard Library Includes
#include <cassert>
#include <list>
#include <set>

#include <iostream>

namespace lucius
{

namespace util
{

class PropertyTreeImplementation
{
public:
    PropertyTreeImplementation(const std::string& name)
    : PropertyTreeImplementation("", name)
    {

    }

    PropertyTreeImplementation(const std::string& p, const std::string& k)
    : path(p), key(k)
    {

    }

public:
    void createValue()
    {
        if(!children.empty())
        {
            return;
        }

        PropertyTree newTree;

        newTree._implementation.reset(new PropertyTreeImplementation(_formPath(), ""));

        children.emplace(newTree);
    }

    void setValue(const PropertyTree& tree)
    {
        assert(children.size() == 1);

        *children.begin()->_implementation = *tree._implementation;

        _fixPaths(*children.begin()->_implementation);
    }

public:
    const PropertyTree& getValue(const std::string& key) const
    {
        auto existingChild = children.find(PropertyTree(key));

        assert(existingChild != children.end());

        return *existingChild;
    }

    PropertyTree& createAndGetValue(const std::string& value)
    {
        auto existingChild = children.find(PropertyTree(value));

        if(existingChild == children.end())
        {
            existingChild = children.emplace(PropertyTree(value)).first;

            _fixPaths(*existingChild->_implementation);
        }

        return const_cast<PropertyTree&>(*existingChild);
    }

    bool exists(const std::string& value)
    {
        auto existingChild = children.find(PropertyTree(value));

        return existingChild != children.end();
    }

private:
    void _fixPaths(PropertyTreeImplementation& node)
    {
        node.path = _formPath();

        for(auto& child : node.children)
        {
            node._fixPaths(*child._implementation);
        }
    }

    std::string _formPath() const
    {
        auto newPath = path;

        if(!newPath.empty())
        {
            newPath += ".";
        }

        newPath += key;

        return newPath;
    }

public:
    typedef std::set<PropertyTree>  TreeSet;
    typedef std::list<PropertyTree> TreeList;

    TreeSet  children;
    TreeList childrenList;

public:
    std::string path;
    std::string key;

};

PropertyTree::PropertyTree(const std::string& name)
: _implementation(new PropertyTreeImplementation(name))
{

}

PropertyTree::PropertyTree()
: PropertyTree("")
{

}

PropertyTree::~PropertyTree()
{

}

PropertyTree::PropertyTree(const PropertyTree& tree)
: _implementation(new PropertyTreeImplementation(*tree._implementation))
{

}

PropertyTree::PropertyTree(PropertyTree&& tree)
: _implementation(std::move(tree._implementation))
{

}

PropertyTree& PropertyTree::operator=(const std::string& v)
{
    _implementation->createValue();
    _implementation->setValue(PropertyTree(v));

    return *this;
}

PropertyTree& PropertyTree::operator=(const PropertyTree& tree)
{
    _implementation->createValue();
    _implementation->setValue(tree);

    return *this;
}

PropertyTree& PropertyTree::operator=(PropertyTree&& tree)
{
    _implementation = std::move(tree._implementation);

    return *this;
}

PropertyTree& PropertyTree::get(const std::string& field)
{
    return (*this)[field];
}

const PropertyTree& PropertyTree::get(const std::string& field) const
{
    return (*this)[field];
}

PropertyTree& PropertyTree::get()
{
    return get(key());
}

const PropertyTree& PropertyTree::get() const
{
    assert(size() == 1);

    return *begin();
}

void PropertyTree::setKey(const std::string& value)
{
    _implementation->key = value;
}

bool PropertyTree::exists(const std::string& key) const
{
    auto* base = this;
    auto components = split(key, ".");

    if(components.size() > 1)
    {
        auto end = --components.end();

        for(auto component = components.begin(); component != end; ++component)
        {
            if(!base->exists(*component))
            {
                return false;
            }

            base = &(*base)[*component];
        }
    }

    return base->_implementation->exists(components.back());
}

std::string& PropertyTree::key()
{
    return _implementation->key;
}

const std::string& PropertyTree::key() const
{
    return _implementation->key;
}

const std::string& PropertyTree::value() const
{
    assert(size() == 1);

    return begin()->key();
}

std::string& PropertyTree::path() const
{
    return _implementation->path;
}

PropertyTree::operator std::string() const
{
    return value();
}

PropertyTree& PropertyTree::operator[](const std::string& key)
{
    auto* base = this;
    auto components = split(key, ".");

    if(components.size() > 1)
    {
        auto end = --components.end();

        for(auto component = components.begin(); component != end; ++component)
        {
            base = &(*base)[*component];
        }
    }

    return base->_implementation->createAndGetValue(components.back());
}

const PropertyTree& PropertyTree::operator[](const std::string& key) const
{
    auto* base = this;
    auto components = split(key, ".");

    if(components.size() > 1)
    {
        auto end = --components.end();

        for(auto component = components.begin(); component != end; ++component)
        {
            base = &(*base)[*component];
        }
    }

    return base->_implementation->getValue(components.back());
}

void PropertyTree::add(const PropertyTree& tree)
{
    _implementation->children.emplace(tree);
}

void PropertyTree::addListElement(const PropertyTree& child)
{
    assert(_implementation->children.empty());

    _implementation->childrenList.push_back(child);
}

PropertyTree::iterator PropertyTree::begin()
{
    return iterator(*this, Begin());
}

PropertyTree::const_iterator PropertyTree::begin() const
{
    return const_iterator(*this, Begin());
}

PropertyTree::iterator PropertyTree::end()
{
    return iterator(*this, End());
}

PropertyTree::const_iterator PropertyTree::end() const
{
    return const_iterator(*this, End());
}

bool PropertyTree::empty() const
{
    return _implementation->children.empty() && _implementation->childrenList.empty();
}

size_t PropertyTree::size() const
{
    if(isList())
    {
        return _implementation->childrenList.size();
    }
    else
    {
        return _implementation->children.size();
    }
}

bool PropertyTree::isList() const
{
    return !_implementation->childrenList.empty();
}

bool PropertyTree::operator<(const PropertyTree& right) const
{
    return key() < right.key();
}

static void saveJson(const PropertyTree& tree, std::ostream& json)
{
    if(tree.empty())
    {
        json << "\"" << tree.key() << "\"";
        return;
    }

    if(tree.isList())
    {
        json << "\"" << tree.key() << "\" : [ ";

        bool first = true;

        for(auto& child : tree)
        {
            if(!first) json << ", ";

            first = false;
            saveJson(child, json);
        }

        json << " ]";
    }
    else if(tree.size() == 1)
    {
        if(!tree.key().empty())
        {
            json << "\"" << tree.key() << "\" : ";

            if(!tree.begin()->empty())
            {
                json << "{ ";
            }
        }

        saveJson(*tree.begin(), json);

        if(!tree.key().empty())
        {
            if(!tree.begin()->empty())
            {
                json << " }";
            }
        }
    }
    else
    {
        if(!tree.key().empty())
        {
            json << "\"" << tree.key() << "\" : ";

            json << "{ ";
        }

        bool first = true;

        for(auto& child : tree)
        {
            if(first)
            {
                first = false;
            }
            else
            {
                json << ", ";
            }

            saveJson(child, json);
        }

        if(!tree.key().empty())
        {
            json << " }";
        }
    }

}

void PropertyTree::saveJson(std::ostream& json) const
{
    json << "{ ";
    util::saveJson(*this, json);
    json << " }";
}

std::string PropertyTree::jsonString() const
{
    std::stringstream stream;

    saveJson(stream);

    return stream.str();
}

static bool isWhitespace(char c)
{
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static void eatWhitespace(std::istream& json)
{
    while(isWhitespace(json.peek()))
    {
        json.get();
    }
}

static bool isToken(char c)
{
    return !isWhitespace(c);
}

static bool isFixedSizeToken(char c)
{
    return c == '\"' || c == '{' || c == '}' || c == ',' || c == ':' || c == '[' || c == ']';
}

static std::string getNextToken(std::istream& json)
{
    std::string token;

    eatWhitespace(json);

    while(json.good() && isToken(json.peek()))
    {
        auto nextCharacter = json.peek();

        if(isFixedSizeToken(nextCharacter))
        {
            if(token.empty())
            {
                token += json.get();
            }

            break;
        }

        token += json.get();
    }

    return token;
}

static std::string getNextString(std::istream& json)
{
    std::string token;

    while(json.good() && json.peek() != '\"')
    {
        token += json.get();
    }

    return token;
}

static std::string peekToken(std::istream& json)
{
    size_t position = json.tellg();

    auto token = getNextToken(json);

    json.seekg(position, std::ios::beg);

    return token;
}

static void parseOpenBrace(std::istream& json)
{
    auto token = getNextToken(json);

    if(token == "{")
    {
        return;
    }

    throw std::runtime_error("Expecting a '{'., but got a '" + token + "'");
}

static void parseOpenBracket(std::istream& json)
{
    auto token = getNextToken(json);

    if(token == "[")
    {
        return;
    }

    throw std::runtime_error("Expecting a '['., but got a '" + token + "'");
}

static void parseQuote(std::istream& json)
{
    auto token = getNextToken(json);

    if(token == "\"")
    {
        return;
    }

    throw std::runtime_error("Expecting a '\"'., but got a '" + token + "'");
}

static void parseKey(PropertyTree& result, std::istream& json)
{
    parseQuote(json);

    auto token = getNextString(json);

    parseQuote(json);

    result.key() = token;
}

static void parseSingleValue(PropertyTree& result, std::istream& json)
{
    parseQuote(json);

    auto token = getNextString(json);

    parseQuote(json);

    result.add(PropertyTree(token));
}

static void parseValue(PropertyTree& result, std::istream& json)
{
    auto token = peekToken(json);

    if(token == "\"")
    {
        parseSingleValue(result, json);
    }
    else
    {
        auto value = PropertyTree::loadJson(json);

        if(value.isList())
        {
            for(auto& child : value)
            {
                result.addListElement(child);
            }
        }
        else
        {
            for(auto& child : value)
            {
                result.add(child);
            }
        }
    }
}

static void parseColon(std::istream& json)
{
    auto token = getNextToken(json);

    if(token == ":")
    {
        return;
    }

    throw std::runtime_error("Expecting a ':'., but got a '" + token + "'");
}

static void parseJsonObjectBody(PropertyTree& result, std::istream& json)
{
    while(true)
    {
        PropertyTree child;

        parseKey(child, json);

        parseColon(json);

        parseValue(child, json);

        result.add(child);

        auto comma = peekToken(json);

        if(comma == ",")
        {
            getNextToken(json);
            continue;
        }

        break;
    }
}

static void parseJsonArrayBody(PropertyTree& result, std::istream& json)
{
    while(true)
    {
        PropertyTree child;

        parseKey(child, json);

        result.addListElement(child);

        auto comma = peekToken(json);

        if(comma == ",")
        {
            getNextToken(json);
            continue;
        }

        break;
    }
}

static void parseCloseBrace(std::istream& json)
{
    auto token = getNextToken(json);

    if(token == "}")
    {
        return;
    }

    throw std::runtime_error("Expecting a '}', but got a '" + token + "'.");
}

static void parseCloseBracket(std::istream& json)
{
    auto token = getNextToken(json);

    if(token == "]")
    {
        return;
    }

    throw std::runtime_error("Expecting a ']', but got a '" + token + "'.");
}

static bool nextTokenIsOpenBracket(std::istream& json)
{
    return peekToken(json) == "[";
}

PropertyTree PropertyTree::loadJson(std::istream& json)
{
    PropertyTree result;

    if(nextTokenIsOpenBracket(json))
    {
        parseOpenBracket(json);

        parseJsonArrayBody(result, json);

        parseCloseBracket(json);
    }
    else
    {
        parseOpenBrace(json);

        parseJsonObjectBody(result, json);

        parseCloseBrace(json);
    }

    return result;
}

class IteratorImplementation
{
public:
    IteratorImplementation()
    : isList(false)
    {

    }

    IteratorImplementation(PropertyTree& tree, PropertyTree::Begin)
    : isList(tree.isList()),
      list(tree._implementation->childrenList.begin()),
      set(tree._implementation->children.begin())
    {

    }

    IteratorImplementation(PropertyTree& tree, PropertyTree::End)
    : isList(!tree._implementation->childrenList.empty()),
      list(tree._implementation->childrenList.end()),
      set(tree._implementation->children.end())
    {

    }

public:
    bool isList;
    PropertyTreeImplementation::TreeList::iterator list;
    PropertyTreeImplementation::TreeSet::iterator set;
};

class ConstIteratorImplementation
{
public:
    ConstIteratorImplementation()
    : isList(false)
    {

    }

    ConstIteratorImplementation(const IteratorImplementation& it)
    : isList(it.isList), list(it.list), set(it.set)
    {

    }

    ConstIteratorImplementation(const PropertyTree& tree, PropertyTree::Begin)
    : isList(tree.isList()),
      list(tree._implementation->childrenList.begin()),
      set(tree._implementation->children.begin())
    {

    }

    ConstIteratorImplementation(const PropertyTree& tree, PropertyTree::End)
    : isList(!tree._implementation->childrenList.empty()),
      list(tree._implementation->childrenList.end()),
      set(tree._implementation->children.end())
    {

    }

public:
    bool isList;
    PropertyTreeImplementation::TreeList::const_iterator list;
    PropertyTreeImplementation::TreeSet::const_iterator set;
};

PropertyTree::iterator::iterator()
: _implementation(std::make_unique<IteratorImplementation>())
{

}

PropertyTree::iterator::iterator(PropertyTree& tree, PropertyTree::Begin begin)
: _implementation(std::make_unique<IteratorImplementation>(tree, begin))
{

}

PropertyTree::iterator::iterator(PropertyTree& tree, PropertyTree::End end)
: _implementation(std::make_unique<IteratorImplementation>(tree, end))
{

}

PropertyTree::iterator::iterator(const iterator& it)
: _implementation(std::make_unique<IteratorImplementation>(*it._implementation))
{

}

PropertyTree::iterator::~iterator() = default;

PropertyTree::iterator& PropertyTree::iterator::operator=(const iterator& it)
{
    *_implementation = *it._implementation;

    return *this;
}

PropertyTree::iterator& PropertyTree::iterator::operator++()
{
    if(_isList())
    {
        ++_implementation->list;
    }
    else
    {
        ++_implementation->set;
    }

    return *this;
}

const PropertyTree& PropertyTree::iterator::operator*() const
{
    if(_isList())
    {
        return *_implementation->list;
    }

    return *_implementation->set;
}

const PropertyTree* PropertyTree::iterator::operator->() const
{
    if(_isList())
    {
        return &*_implementation->list;
    }

    return &*_implementation->set;
}

bool PropertyTree::iterator::operator==(const iterator& it)
{
    if(it._isList() != _isList())
    {
        return false;
    }

    if(_isList())
    {
        return _implementation->list == it._implementation->list;
    }

    return _implementation->set == it._implementation->set;
}

bool PropertyTree::iterator::operator==(const const_iterator& it)
{
    if(it._isList() != _isList())
    {
        return false;
    }

    if(_isList())
    {
        return _implementation->list == it._implementation->list;
    }

    return _implementation->set == it._implementation->set;
}

bool PropertyTree::iterator::operator!=(const iterator& it)
{
    return !(*this == it);
}

bool PropertyTree::iterator::operator!=(const const_iterator& it)
{
    return !(*this == it);
}

bool PropertyTree::iterator::_isList() const
{
    return _implementation->isList;
}

PropertyTree::const_iterator::const_iterator()
: _implementation(std::make_unique<ConstIteratorImplementation>())
{

}

PropertyTree::const_iterator::const_iterator(const PropertyTree& tree, PropertyTree::Begin begin)
: _implementation(std::make_unique<ConstIteratorImplementation>(tree, begin))
{

}

PropertyTree::const_iterator::const_iterator(const PropertyTree& tree, PropertyTree::End end)
: _implementation(std::make_unique<ConstIteratorImplementation>(tree, end))
{

}

PropertyTree::const_iterator::const_iterator(const iterator& it)
: _implementation(std::make_unique<ConstIteratorImplementation>(*it._implementation))
{

}

PropertyTree::const_iterator::const_iterator(const const_iterator& it)
: _implementation(std::make_unique<ConstIteratorImplementation>(*it._implementation))
{

}

PropertyTree::const_iterator::~const_iterator() = default;

PropertyTree::const_iterator& PropertyTree::const_iterator::operator=(const const_iterator& it)
{
    if(this == &it)
    {
        return *this;
    }

    *_implementation = *it._implementation;

    return *this;
}

PropertyTree::const_iterator& PropertyTree::const_iterator::operator=(const iterator& it)
{
    _implementation->isList = it._implementation->isList;
    _implementation->list   = it._implementation->list;
    _implementation->set    = it._implementation->set;

    return *this;
}

PropertyTree::const_iterator& PropertyTree::const_iterator::operator++()
{
    if(_isList())
    {
        ++_implementation->list;
    }
    else
    {
        ++_implementation->set;
    }

    return *this;
}

const PropertyTree& PropertyTree::const_iterator::operator*() const
{
    if(_isList())
    {
        return *_implementation->list;
    }

    return *_implementation->set;
}

const PropertyTree* PropertyTree::const_iterator::operator->() const
{
    if(_isList())
    {
        return &*_implementation->list;
    }

    return &*_implementation->set;
}

bool PropertyTree::const_iterator::operator==(const iterator& it)
{
    if(it._isList() != _isList())
    {
        return false;
    }

    if(_isList())
    {
        return _implementation->list == it._implementation->list;
    }

    return _implementation->set == it._implementation->set;
}

bool PropertyTree::const_iterator::operator==(const const_iterator& it)
{
    if(it._isList() != _isList())
    {
        return false;
    }

    if(_isList())
    {
        return _implementation->list == it._implementation->list;
    }

    return _implementation->set == it._implementation->set;
}

bool PropertyTree::const_iterator::operator!=(const iterator& it)
{
    return !(*this == it);
}

bool PropertyTree::const_iterator::operator!=(const const_iterator& it)
{
    return !(*this == it);
}

bool PropertyTree::const_iterator::_isList() const
{
    return _implementation->isList;
}

}

}



