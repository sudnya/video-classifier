/*	\file   TarArchive.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the tar archive class.

*/

// Lucious Includes
#include <lucious/util/interface/TarArchive.h>

#include <lucious/util/interface/debug.h>

// Standard Library Includes
#include <stdexcept>
#include <iomanip>
#include <map>

namespace lucious
{

namespace util
{

class TarFileHeader
{
public:
    TarFileHeader(const std::string& name, size_t size)
    : _name(name), _size(size)
    {

    }

    TarFileHeader(std::istream& file)
    : TarFileHeader("", 0)
    {
        _loadFromFile(file);
    }

public:
    void write(std::ostream& file)
    {
        std::stringstream stream;

        // 100-byte file name
        _writeAndPad(stream, _name.data(), _name.size(), 100);

        // 8-byte mode
        _writeAndPad(stream, "000777 ", 7, 8);

        // 8-byte user id
        _writeAndPad(stream, "001000 ", 7, 8);

        // 8-byte group id
        _writeAndPad(stream, "001000 ", 7, 8);

        // 12-byte file size
        _writeAndPad(stream, _toUstarNumber(_size).data(), _toUstarNumber(_size).size(), 12);

        // 12-byte timestamp
        _writeAndPad(stream, "00000000000 ", 12, 12);

        // 8-byte checksum
        _writeAndPad(stream, "        ", 8, 8);

        // 1-byte type (0 == normal file)
        _writeAndPad(stream, "0", 1, 1);

        // 100-byte link name
        _writeAndPad(stream, "", 0, 100);

        // ustar
        _writeAndPad(stream, "ustar", 5, 6);

        // ustar version
        _writeAndPad(stream, "00", 2, 2);

        // 32-byte user name
        _writeAndPad(stream, "normal", 6, 32);

        // 32-byte group name
        _writeAndPad(stream, "normal", 6, 32);

        // 8-byte device major number
        _writeAndPad(stream, "000000 ", 7, 8);

        // 8-byte device minor number
        _writeAndPad(stream, "000000 ", 7, 8);

        // 155-byte filename prefix
        _writeAndPad(stream, "\0", 1, 155);

        // zero pad
        _writeAndPad(stream, "", 0, 11);

        _writeChecksum(stream);

        _writeToFile(file, stream);
    }

public:
    const std::string& name() const
    {
        return _name;
    }

    size_t size() const
    {
        return _size;
    }

private:
    void _writeAndPad(std::ostream& stream, const void* data, size_t bytes, size_t totalBytes)
    {
        stream.write(reinterpret_cast<const char*>(data), bytes);

        for(size_t i = bytes; i < totalBytes; ++i)
        {
            stream.put(0);
        }
    }

    void _writeChecksum(std::stringstream& stream)
    {
        uint32_t checksum = 0;

        stream.seekg(0, std::ios::beg);

        while(stream.good())
        {
            checksum += stream.get();
        }

        stream.clear();

        std::stringstream number;

        number.width(6);
        number.fill('0');
        number << std::setbase(8);

        number << checksum;

        number.put('\0');

        stream.seekp(148, std::ios::beg);

        stream.write(number.str().c_str(), number.str().size());

        stream.put(' ');

        stream.seekg(0, std::ios::beg);
    }

    void _writeToFile(std::ostream& output, std::istream& input)
    {
        while(input.good())
        {
            output.put(input.get());
        }
    }

    std::vector<uint8_t> _toUstarNumber(size_t number)
    {
        std::vector<uint8_t> data(12);

        std::stringstream stream;

        stream.width(11);
        stream.fill('0');
        stream << std::setbase(8);

        stream << number;

        std::memcpy(data.data(), stream.str().data(), 11);
        data[11] = ' ';

/*
        std::memcpy(data.data(), &number, sizeof(size_t));

        data[11] |= 0x80;
*/
        return data;
    }

private:
    void _loadFromFile(std::istream& file)
    {
        size_t start = file.tellg();

        _name = _loadString(file, 100);

        file.seekg(start, std::ios::beg);
        file.seekg(124,   std::ios::cur);

        _size = _loadNumber(file, 12);

        file.seekg(start, std::ios::beg);
        file.seekg(512,   std::ios::cur);
    }

    std::string _loadString(std::istream& file, size_t maxSize)
    {
        std::string result;

        for(size_t i = 0; i < maxSize; ++i)
        {
            char character = file.get();

            if(character == '\0')
            {
                break;
            }

            result += character;
        }

        return result;
    }

    size_t _loadNumber(std::istream& file, size_t bytes)
    {
        std::string number(' ', bytes);

        file.read(const_cast<char*>(number.c_str()), bytes);

        std::stringstream stream;

        stream << std::setbase(8);
        stream << number;

        size_t result = 0;

        stream >> result;

        return result;
    }

private:
    std::string _name;
    size_t      _size;

};

static size_t getFileSize(std::istream& file)
{
    auto start = file.tellg();

    file.seekg(0, std::ios::end);

    auto size = file.tellg() - start;

    file.seekg(start, std::ios::beg);

    return size;
}

static size_t getPadding(size_t size, size_t alignment)
{
    size_t remainder = size % alignment;

    return remainder == 0 ? 0 : alignment - remainder;
}

class OutputTarArchiveImplementation
{
public:
    OutputTarArchiveImplementation(std::ostream& s)
    : _stream(s)
    {

    }

public:
    void addFile(const std::string& name, std::istream& file)
    {
        _writeHeader(name, file);
        _writeData(file);
    }

private:
    void _writeHeader(const std::string& name, std::istream& file)
    {
        size_t size = getFileSize(file);

        TarFileHeader header(name, size + getPadding(size, 512));

        header.write(_stream);
    }

    void _writeData(std::istream& file)
    {
        size_t remainingSize = getFileSize(file);
        size_t padding = getPadding(remainingSize, 512);

        for(size_t i = 0; i < remainingSize; ++i)
        {
            _stream.put(file.get());
        }

        for(size_t i = 0; i < padding; ++i)
        {
            _stream.put(0);
        }
    }

private:
    std::ostream& _stream;

};

OutputTarArchive::OutputTarArchive(std::ostream& s)
: _implementation(new OutputTarArchiveImplementation(s))
{

}

OutputTarArchive::~OutputTarArchive()
{

}

void OutputTarArchive::addFile(const std::string& name, std::istream& file)
{
    _implementation->addFile(name, file);
}

class InputTarArchiveImplementation
{
public:
    InputTarArchiveImplementation(std::istream& s)
    : _stream(s)
    {
        _findFiles();
    }

public:
    void extractFile(const std::string& filename, std::ostream& file)
    {
        auto position = files.find(filename);

        assert(position != files.end());

        _stream.clear();

        _stream.seekg(std::get<0>(position->second), std::ios::beg);

        for(size_t i = 0; i < std::get<1>(position->second); ++i)
        {
            file.put(_stream.get());
        }
    }

private:
    void _findFiles()
    {
        while(_stream.good())
        {
            TarFileHeader header(_stream);

            if(!_stream.good())
            {
                break;
            }

            util::log("TarFileHeader") << "Loaded TAR header for file " << header.name()
                << " with " << header.size() << " bytes\n";

            files.emplace(header.name(),
                std::make_tuple(_stream.tellg(), header.size()));

            _stream.seekg(header.size(), std::ios::cur);
        }
    }

public:
    typedef std::map<std::string, std::tuple<size_t, size_t>> NameToPositionMap;

public:
    NameToPositionMap files;

private:
    std::istream& _stream;

};

InputTarArchive::InputTarArchive(std::istream& s)
: _implementation(new InputTarArchiveImplementation(s))
{

}

InputTarArchive::~InputTarArchive()
{

}

InputTarArchive::StringVector InputTarArchive::list() const
{
    StringVector files;

    for(auto& file : _implementation->files)
    {
        files.push_back(file.first);
    }

    return files;
}

void InputTarArchive::extractFile(const std::string& name, std::ostream& file)
{
    _implementation->extractFile(name, file);
}

}

}



