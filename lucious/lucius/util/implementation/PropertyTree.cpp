/*  \file   PropertyTree.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the PropertyTree class.
*/

// Lucious Includes
#include <lucious/util/interface/PropertyTree.h>

// Standard Library Includes
#include <cassert>

namespace lucious
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
    PropertyTree::TreeSet children;

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
    return _implementation->createAndGetValue(key);
}

const PropertyTree& PropertyTree::operator[](const std::string& key) const
{
    return _implementation->getValue(key);
}

void PropertyTree::add(const PropertyTree& tree)
{
    _implementation->children.emplace(tree);
}

PropertyTree::iterator PropertyTree::begin()
{
    return _implementation->children.begin();
}

PropertyTree::const_iterator PropertyTree::begin() const
{
    return _implementation->children.begin();
}

PropertyTree::iterator PropertyTree::end()
{
    return _implementation->children.end();
}

PropertyTree::const_iterator PropertyTree::end() const
{
    return _implementation->children.end();
}

bool PropertyTree::empty() const
{
    return _implementation->children.empty();
}

size_t PropertyTree::size() const
{
    return _implementation->children.size();
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

    if(tree.size() == 1)
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
    return c == '\"' || c == '{' || c == '}' || c == ',' || c == ':';
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

    throw std::runtime_error("Expecting a '{'.");
}

static void parseQuote(std::istream& json)
{
    auto token = getNextToken(json);

    if(token == "\"")
    {
        return;
    }

    throw std::runtime_error("Expecting a '\"'.");
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

        for(auto& child : value)
        {
            result.add(child);
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

    throw std::runtime_error("Expecting a ':'.");
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

static void parseCloseBrace(std::istream& json)
{
    auto token = getNextToken(json);

    if(token == "}")
    {
        return;
    }

    throw std::runtime_error("Expecting a '}'.");
}

PropertyTree PropertyTree::loadJson(std::istream& json)
{
    PropertyTree result;

    parseOpenBrace(json);

    parseJsonObjectBody(result, json);

    parseCloseBrace(json);

    return result;
}

}

}



