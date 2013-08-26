/*	\file   TarArchive.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the tar archive class.
	
*/

// Minerva Includes
#include <minerva/util/interface/TarArchive.h>
#include <minerva/util/interface/TarLibrary.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

namespace minerva
{

namespace util
{

static size_t getSize(std::istream& stream)
{
	size_t position = stream.tellg();

	stream.seekg(0, std::ios::end);
	
	size_t size = stream.tellg();
	
	stream.seekg(position, std::ios::beg);

	return size;
}

class TarArchiveImplementation
{
public:
	TarArchiveImplementation(const std::string& p, const std::string& m)
	: _path(p), _mode(m), _archive(nullptr), _file(nullptr)
	{
		util::log("TarArchive") << "Creating tar archive '" + p +
			"' with mode '" + m + "'\n";
		
	}
	
	~TarArchiveImplementation()
	{
		close();
	}

public:
	void initialize()
	{
		TarLibrary::load();
		
		if(!TarLibrary::loaded())
		{
			throw std::runtime_error("Failed to load tar library.");
		}
	
		if(isReadMode())
		{
			_file = std::fopen(_path.c_str(), "r");
			
			if(_file == nullptr)
			{
				throw std::runtime_error("Failed to open archive file '"
					+ _path + "' for reading.");
			}

			_archive = TarLibrary::archive_read_new();
			
			if(_archive == nullptr)
			{
				throw std::runtime_error("Failed to create new archive.");
			}
			
			if(TarLibrary::archive_read_support_filter_all(_archive) !=
				TarLibrary::OK)
			{
				throw std::runtime_error("Failed to setup "
					"archive compression.");
			}
			
			if(TarLibrary::archive_read_support_format_all(_archive) !=
				TarLibrary::OK)
			{
				throw std::runtime_error("Failed to setup "
					"archive read formats.");
			}
			
			if(TarLibrary::archive_read_open_FILE(_archive, _file) !=
				TarLibrary::OK)
			{
				throw std::runtime_error("Failed to open archive.");
			}
			
			util::log("TarArchive") << " Opened archive in read mode...\n";
		
			
		}
		else if(isWriteMode())
		{
			_file = std::fopen(_path.c_str(), "w");

			if(_file == nullptr)
			{
				throw std::runtime_error("Failed to open archive file '"
					+ _path + "' for writing.");
			}
			
			_archive = TarLibrary::archive_write_new();
			
			if(_archive == nullptr)
			{
				throw std::runtime_error("Failed to create new archive.");
			}
			
			if(TarLibrary::archive_write_set_compression_gzip(_archive) !=
				TarLibrary::OK)
			{
				throw std::runtime_error("Failed to setup "
					"archive compression.");
			}
			
			if(TarLibrary::archive_write_set_format_pax_restricted(_archive) !=
				TarLibrary::OK)
			{
				throw std::runtime_error("Failed to setup "
					"archive write format to TAR.");
			}
			
			if(TarLibrary::archive_write_open_FILE(_archive, _file) !=
				TarLibrary::OK)
			{
				throw std::runtime_error("Failed to open archive.");
			}

			util::log("TarArchive") << " Opened archive in write mode...\n";
		}
		else
		{
			throw std::runtime_error("mode " + _mode +
				" is not supported by TarArchive.");
		}
	}
	
	void close()
	{
		if(_archive != nullptr)
		{
			if(isReadMode())
			{
				TarLibrary::archive_read_free(_archive);
			}
			else
			{
				TarLibrary::archive_write_close(_archive);
				TarLibrary::archive_write_free(_archive);
			}
		}
		
		if(_file != nullptr)
		{
			std::fclose(_file);
		}
	}
	
	void reset()
	{
		util::log("TarArchive") << " Resetting archive...\n";
		
		close();
		
		initialize();
	}

	void addFile(const std::string& name, std::istream& file)
	{
		util::log("TarArchive") << " Adding file '" + name +
			"' to archive '" + _path + "'\n";
		
		auto entry = TarLibrary::archive_entry_new();
		 
		if(entry == nullptr)
		{
			throw std::runtime_error("Failed to create archive entry.");
		}

		util::log("TarArchive") << "  Creating entry in archive...\n";
		
		size_t size = getSize(file);
		
		TarLibrary::archive_entry_set_uid(entry, 0);
		TarLibrary::archive_entry_set_uname(entry, "root");
		TarLibrary::archive_entry_set_gid(entry, 0);
		TarLibrary::archive_entry_set_gname(entry, "wheel");
		TarLibrary::archive_entry_set_filetype(entry, TarLibrary::RegularFile);
		TarLibrary::archive_entry_set_perm(entry, 0644);
		TarLibrary::archive_entry_set_size(entry, size);
		TarLibrary::archive_entry_set_pathname(entry, name.c_str());
		
		if(TarLibrary::archive_write_header(_archive, entry) !=
			TarLibrary::OK)
		{
			TarLibrary::archive_entry_free(entry);
			
			throw std::runtime_error("Failed to write entry to archive. "
				"Message: " + TarLibrary::archive_error_string(_archive));
		}

		util::log("TarArchive") << "  Writing data (" << size
			<< " bytes) to archive...\n";
		
		char buffer[1024];
		
		while(size > 0)
		{
			int count = file.readsome(buffer, 1024);
		
			size -= count;
		
			if(TarLibrary::archive_write_data(_archive, buffer, count) != count)
			{
				throw std::runtime_error("Failed to write data to archive."
					"Message: " + TarLibrary::archive_error_string(_archive));
			}
		}

		if(TarLibrary::archive_write_finish_entry(_archive) != TarLibrary::OK)
		{
			throw std::runtime_error("Failed to finish "
				"writing data to archive.");
		}

		TarLibrary::archive_entry_free(entry);

		util::log("TarArchive") << "  File added successfully...\n";
	}

	void extractFile(const std::string& name, std::ostream& file)
	{
		reset();
		
		TarLibrary::archive_entry* entry = nullptr;
		
		int status = TarLibrary::archive_read_next_header(_archive, &entry);
		
		while(status == TarLibrary::OK)
		{
			if(name == TarLibrary::archive_entry_pathname(entry))
			{
				size_t size = TarLibrary::archive_entry_size(entry);

				char buffer[1024];
				
				while(size > 0)
				{
					size_t transferSize = std::min((size_t)1024, size);
				
					if(TarLibrary::archive_read_data(_archive, buffer,
						transferSize) != transferSize)
					{
						TarLibrary::archive_entry_free(entry);
						
						throw std::runtime_error("Failed to read data "
							"from archive.");
					}
					
					file.write(buffer, transferSize);
					
					size -= transferSize;
				}
			
				return;
			}
			
			status = TarLibrary::archive_read_next_header(_archive, &entry);
		}
		
		throw std::runtime_error("Could not find filename '" + name +
			"' in archive '" + _path + "'");
	}

private:
	bool isReadMode()
	{
		return _mode == "r:gz";
	}

	bool isWriteMode()
	{
		return _mode == "w:gz";
	}

private:
	std::string _path;
	std::string _mode;
	
private:
	TarLibrary::archive* _archive;
	FILE*                _file;
	
};

TarArchive::TarArchive(const std::string& path, const std::string& mode)
: _archive(new TarArchiveImplementation(path, mode))
{
	try
	{
		_archive->initialize();
	}
	catch(...)
	{
		delete _archive;
	
		throw;
	}
}

TarArchive::~TarArchive()
{
	delete _archive;
}

TarArchive::StringVector TarArchive::list() const
{
	StringVector files;
	
	assertM(false, "Get list of files not implemented.");
	
	return files;
}

void TarArchive::addFile(const std::string& name, std::istream& file)
{
	_archive->addFile(name, file);
}

void TarArchive::extractFile(const std::string& name, std::ostream& file)
{
	_archive->extractFile(name, file);
}

}

}



