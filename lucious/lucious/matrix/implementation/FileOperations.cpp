// Lucious Includes
#include <lucious/matrix/interface/FileOperations.h>

#include <lucious/matrix/interface/TransposeOperations.h>
#include <lucious/matrix/interface/CopyOperations.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/Dimension.h>
#include <lucious/matrix/interface/Precision.h>

#include <lucious/util/interface/debug.h>
#include <lucious/util/interface/string.h>

#include <fstream>
#include <map>

namespace lucious
{
namespace matrix
{

static void addMagic(std::ostream& file)
{
    std::string magicString("\x93NUMPY");
    file.write(const_cast<const char*>(magicString.data()), 6);
}


static void addVersion(std::ostream& file)
{
    uint8_t majorVersion = 0x01;
    uint8_t minorVersion = 0x00;

    file.write(reinterpret_cast<const char*>(&majorVersion), 1);
    file.write(reinterpret_cast<const char*>(&minorVersion), 1);
}

static size_t getAlignment(size_t currentSize, size_t alignment)
{
    size_t remainder = currentSize % alignment;
    
    return remainder == 0 ? 0 : alignment - remainder;
}


static std::string getEncodedPrecision(const Precision& precision)
{
    if(precision == SinglePrecision())
    {
        return "<f4";
    }
    else if(precision == DoublePrecision())
    {
        return "<f8";
    }
    else if(precision == HalfPrecision())
    {
        return "<f2";
    }
    
    return "<f4";
}


static void addHeader(std::ostream& file, const std::string& matrixShapeString, const Precision& matrixPrecision)
{
    std::string headerPrefix("{'descr': '" + getEncodedPrecision(matrixPrecision) + "', 'fortran_order': True, 'shape': ");
    size_t boundaryAlignment = 16;
    size_t magicAndVersionLength = 10;
    std::string shapeDim      = "(" + matrixShapeString + "), }";
    uint8_t remainingBytes    = getAlignment(headerPrefix.size() + shapeDim.size() + 1 + magicAndVersionLength, boundaryAlignment);
    std::string headerPostfix = std::string(remainingBytes, ' ') + "\n";

    std::string headerString  = headerPrefix + shapeDim + headerPostfix;     
    uint16_t headerLength     = headerString.size();
    file.write(reinterpret_cast<const char*>(&headerLength), 2);
    file.write(reinterpret_cast<const char*>(headerString.data()), headerString.size());
}


static void checkMagic(std::istream& file)
{
    std::string magicString("......");
    file.read(const_cast<char*>(magicString.data()), 6);
    if(magicString != "\x93NUMPY")
    {
        throw std::runtime_error("Magic string: " + magicString + " does not match expected x93NUMPY");
    }
}

static void checkVersion(std::istream& file)
{
    uint8_t majorVersion = 0;
    uint8_t minorVersion = 0;

    file.read(reinterpret_cast<char*>(&majorVersion), 1);
    file.read(reinterpret_cast<char*>(&minorVersion), 1);

    if(majorVersion != 0x01)
    {
        throw std::runtime_error("Major version: " + std::to_string(majorVersion) + " does not match expected 0x01");
    }
    if(minorVersion != 0x00)
    {
        throw std::runtime_error("Minor version: " + std::to_string(minorVersion) + " does not match expected 0x00");
    }
}

namespace
{

class Header
{
public:
    Header(std::istream& file)
    {
        uint16_t headerLength = 0;
        file.read(reinterpret_cast<char*>(&headerLength), 2);

        util::log("FileOperations") << " header length: " << headerLength << std::endl;
    
        std::string header(headerLength, ' ');

        file.read(const_cast<char*>(header.data()), headerLength);
        
        auto dictionary = _parseDictionary(header);

        if(dictionary.count("'descr'") == 0)
        {
            throw std::runtime_error("Header missing descr field.");
        }

        if(dictionary.count("'fortran_order'") == 0)
        {
            throw std::runtime_error("Header missing fortran_order field.");
        }

        if(dictionary.count("'shape'") == 0)
        {
            throw std::runtime_error("Header missing shape field.");
        }

        _precision    = _parsePrecision(dictionary["'descr'"]);
        _dimension    = _parseDimension(dictionary["'shape'"]);
        _fortranOrder = _parseFortranOrder(dictionary["'fortran_order'"]);
    }

    Dimension size() const
    {
        return _dimension;
    }

    Precision precision() const
    {
        return _precision;
    }

    bool isFortranOrder() const
    {
        return _fortranOrder;
    }

private:
    typedef std::map<std::string, std::string> Dictionary;

private:
    Dictionary _parseDictionary(const std::string& dictionary) const
    {
        Dictionary result;

        auto separate_entries = util::split(dictionary, ",");
        
        util::StringVector entries;

        bool searching = false;
        std::string compound = "";

        for(auto entry : separate_entries)
        {
            if(entry.find("(") != std::string::npos)
            {
                compound += entry;
                searching = true;
                continue;
            }
            
            if(searching)
            {
                compound += "," + entry;

                if(entry.find(")") != std::string::npos)
                {
                    util::log("FileOperations") << "compound entry: '" << compound << "'\n";
                    entries.push_back(compound);
                    compound.clear();
                    searching = false;
                }
                
                continue;
            }
            
            util::log("FileOperations") << "entry: '" << entry << "'\n";
            if(entry.find(":") != std::string::npos)
            {
                entries.push_back(entry);
            }
        }

        for(auto entry : entries)
        {
            auto keyAndValue = util::split(entry, ":");

            if(keyAndValue.size() != 2)
            {
                throw std::runtime_error("Invalid dictionary format, missing key/value pair in " + entry);
            }

            auto key   = util::removeWhitespace(util::strip(util::strip(keyAndValue[0], "}"), "{"));
            auto value = util::removeWhitespace(util::strip(util::strip(keyAndValue[1], "}"), "{"));

            util::log("FileOperations") << "key: " << key << ", value: " << value << "\n";

            result[key] = value;
        }

        return result;
    }

    Precision _parsePrecision(const std::string& precision) const
    {
        if(precision == "'<f8'")
        {
            return DoublePrecision();
        }
        else if(precision == "'<f4'")
        {
            return SinglePrecision();
        }

        throw std::runtime_error("Unknown/Unsupported precision type '" + precision + "'");
    }

    Dimension _parseDimension(const std::string& dimension) const
    {
        auto dimensions = util::split(dimension, ",");

        Dimension result;

        for(auto dimension : dimensions)
        {
            result.push_back(std::stoi(util::strip(util::strip(dimension, "("), ")")));
        }

        return result;
    }

    bool _parseFortranOrder(const std::string& order) const
    {
        return order == "True";
    }


private:
    Precision _precision;
    Dimension _dimension;
    bool      _fortranOrder;
};

}

Matrix load(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    
    if(!file.is_open()) 
    {
        throw std::runtime_error("Could not open file containing numpy matrix from " + path);
    }

    return load(file);
}

Matrix load(std::istream& file)
{
    checkMagic(file);
    checkVersion(file);

    Header header(file);

    Matrix m(header.size(), header.precision());

    file.read(reinterpret_cast<char*>(m.data()), m.size().product() * m.precision().size());

    if(!header.isFortranOrder())
    {
        transpose(m, copy(m));
    }

    return m;
    
}

void save(const std::string& path, const Matrix& input)
{
    std::ofstream file(path, std::ios::binary);

    if(!file.is_open())
    {
        throw std::runtime_error("Could not create numpy file to write matrix to, at " + path);
    }
    
    save(file, input);
}

void save(std::ostream& file, const Matrix& input)
{
    addMagic(file);
    addVersion(file);
    addHeader(file, input.shapeString(), input.precision());

    file.write(reinterpret_cast<const char*>(input.data()), input.elements() * input.precision().size());
}

}
}




