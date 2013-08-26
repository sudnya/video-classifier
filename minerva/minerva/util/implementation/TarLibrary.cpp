/*	\file   TarLibrary.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  An source to the libarchive interface.
	
*/

// Minerva Includes
#include <minerva/util/interface/TarLibrary.h>
#include <minerva/util/interface/Casts.h>

// Standard Library Includes
#include <stdexcept>

// System-Specific Includes
#include <dlfcn.h>

namespace minerva
{

namespace util
{

void TarLibrary::load()
{
	_interface.load();
}

bool TarLibrary::loaded()
{
	return _interface.loaded();
}

TarLibrary::archive* TarLibrary::archive_read_new()
{
	_check();
	
	return (*_interface.archive_read_new)();
}

int TarLibrary::archive_read_free(archive* a)
{
	_check();

    if(_interface.archive_write_free == nullptr)
    {
        return OK;
    }
	
	return (*_interface.archive_read_free)(a);
}

int TarLibrary::archive_read_support_filter_all(archive* a)
{
	_check();
	
	return (*_interface.archive_read_support_filter_all)(a);
}

int TarLibrary::archive_read_support_format_all(archive* a)
{
	_check();
	
	return (*_interface.archive_read_support_format_all)(a);
}

int TarLibrary::archive_read_open_FILE(archive* a, FILE* file)
{
	_check();
	
	return (*_interface.archive_read_open_FILE)(a, file);
}

int TarLibrary::archive_read_next_header(archive* a, archive_entry** e)
{
	_check();
	
	return (*_interface.archive_read_next_header)(a, e);
}

size_t TarLibrary::archive_read_data(archive* a, void* buffer, size_t bytes)
{
	_check();
	
	return (*_interface.archive_read_data)(a, buffer, bytes);
}

TarLibrary::archive* TarLibrary::archive_write_new()
{
	_check();
	
	return (*_interface.archive_write_new)();
}

int TarLibrary::archive_write_free(archive* a)
{
	_check();

    if(_interface.archive_write_free == nullptr)
    {
        return OK;
    }

	return (*_interface.archive_write_free)(a);
}

int TarLibrary::archive_write_set_compression_gzip(archive* a)
{
	_check();
	
	return (*_interface.archive_write_set_compression_gzip)(a);
}

int TarLibrary::archive_write_set_format_pax_restricted(archive* a)
{
	_check();
	
	return (*_interface.archive_write_set_format_pax_restricted)(a);
}

int TarLibrary::archive_write_open_FILE(archive* a, FILE* file)
{
	_check();
	
	return (*_interface.archive_write_open_FILE)(a, file);
}

int TarLibrary::archive_write_header(archive* a, archive_entry* e)
{
	_check();
	
	return (*_interface.archive_write_header)(a, e);
}

int TarLibrary::archive_write_data(archive* a, const void* buffer, size_t bytes)
{
	_check();
	
	return (*_interface.archive_write_data)(a, buffer, bytes);
}

int TarLibrary::archive_write_finish_entry(archive* a)
{
	_check();
	
	return (*_interface.archive_write_finish_entry)(a);
}

int TarLibrary::archive_write_close(archive* a)
{
	_check();
	
	return (*_interface.archive_write_close)(a);
}

TarLibrary::archive_entry* TarLibrary::archive_entry_new()
{
	_check();
	
	return (*_interface.archive_entry_new)();
}

int TarLibrary::archive_entry_free(archive_entry* a)
{
	_check();
	
	return (*_interface.archive_entry_free)(a);
}
	
void TarLibrary::archive_entry_set_uid(archive_entry* entry, uint64_t id)
{
	_check();
	
	return (*_interface.archive_entry_set_uid)(entry, id);
}

void TarLibrary::archive_entry_set_uname(archive_entry* entry, const char* name)
{
	_check();
	
	return (*_interface.archive_entry_set_uname)(entry, name);
}

void TarLibrary::archive_entry_set_gid(archive_entry* entry, uint64_t id)
{
	_check();
	
	return (*_interface.archive_entry_set_gid)(entry, id);
}

void TarLibrary::archive_entry_set_gname(archive_entry* entry, const char* name)
{
	_check();
	
	return (*_interface.archive_entry_set_gname)(entry, name);
}

void TarLibrary::archive_entry_set_filetype(
	archive_entry* entry, unsigned int type)
{
	_check();
	
	return (*_interface.archive_entry_set_filetype)(entry, type);
}

void TarLibrary::archive_entry_set_perm(archive_entry* entry, int permission)
{
	_check();
	
	return (*_interface.archive_entry_set_perm)(entry, permission);
}

void TarLibrary::archive_entry_set_size(archive_entry* entry, size_t size)
{
	_check();
	
	return (*_interface.archive_entry_set_size)(entry, size);
}

void TarLibrary::archive_entry_set_pathname(archive_entry* entry,
	const char* name)
{
	_check();
	
	return (*_interface.archive_entry_set_pathname)(entry, name);
}

size_t TarLibrary::archive_entry_size(archive_entry* entry)
{
	_check();
	
	return (*_interface.archive_entry_size)(entry);
}

const char* TarLibrary::archive_entry_pathname(archive_entry* entry)
{
	_check();
	
	return (*_interface.archive_entry_pathname)(entry);
}


std::string TarLibrary::archive_error_string(archive* archive)
{
	_check();
	
	return (*_interface.archive_error_string)(archive);
}

void TarLibrary::_check()
{
	if(!loaded())
	{
		throw std::runtime_error("Tried to call libarchive function when "
			"the library is not loaded.");
	}
}

TarLibrary::Interface TarLibrary::_interface;

TarLibrary::Interface::Interface()
: _library(nullptr)
{

}

TarLibrary::Interface::~Interface()
{
	unload();
}


void TarLibrary::Interface::load()
{
	if(loaded()) return;
	
    #if __APPLE__
    _library = dlopen("libarchive.dylib", RTLD_LAZY);
    #else
	_library = dlopen("libarchive.so", RTLD_LAZY);
    #endif

	if(!loaded())
	{
		return;
	}
	
	#define DynLink( function ) bit_cast( function, dlsym(_library, #function))
	
	DynLink(archive_read_new);
	DynLink(archive_read_free);
	DynLink(archive_read_support_filter_all);
	DynLink(archive_read_support_format_all);
	DynLink(archive_read_open_FILE);
	DynLink(archive_read_next_header);
	DynLink(archive_read_data);
	
	DynLink(archive_write_new);
	DynLink(archive_write_free);
	DynLink(archive_write_set_compression_gzip);
	DynLink(archive_write_set_format_pax_restricted);
	DynLink(archive_write_open_FILE);
	DynLink(archive_write_header);
	DynLink(archive_write_data);
	
	DynLink(archive_write_finish_entry);
	DynLink(archive_write_close);
	
	DynLink(archive_entry_new);
	DynLink(archive_entry_free);
	DynLink(archive_entry_set_uid);
	DynLink(archive_entry_set_uname);
	DynLink(archive_entry_set_gid);
	DynLink(archive_entry_set_gname);
	DynLink(archive_entry_set_filetype);
	DynLink(archive_entry_set_perm);
	DynLink(archive_entry_set_size);
	DynLink(archive_entry_set_pathname);
	DynLink(archive_entry_size);
	DynLink(archive_entry_pathname);
	
	DynLink(archive_error_string);
	
	#undef DynLink

    // update deprecated APIs
    if(archive_read_support_filter_all == nullptr)
    {
        bit_cast(archive_read_support_filter_all, dlsym(_library, "archive_read_support_compression_all"));
    }


}

bool TarLibrary::Interface::loaded() const
{
	return _library != nullptr;
}

void TarLibrary::Interface::unload()
{
	if(!loaded()) return;

	dlclose(_library);
	_library = nullptr;
}

}

}



