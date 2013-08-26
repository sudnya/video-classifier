/*	\file   TarLibrary.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  An interface to lib tar.
	
 */

#pragma once

// Standard Library Includes
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <string>

namespace minerva
{

namespace util
{

class TarLibrary
{
public:
	struct archive;
	struct archive_entry;
	
public:
	static const int OK = 0;
	static const unsigned int RegularFile = 0100000;
	
public:
	static void load();
	static bool loaded();

public:
	static archive* archive_read_new();
	static int archive_read_free(archive*);
	
	static int archive_read_support_filter_all(archive* );
	static int archive_read_support_format_all(archive* );
	static int archive_read_open_FILE(archive* , FILE* file);

	static int archive_read_next_header(archive*, archive_entry**);
	
	static size_t archive_read_data(archive*, void* buffer, size_t bytes);

public:
	static archive* archive_write_new();
	static int archive_write_free(archive*);
	
	static int archive_write_set_compression_gzip(archive* );
	static int archive_write_set_format_pax_restricted(archive* );
	static int archive_write_open_FILE(archive* , FILE* file);
	
	static int archive_write_header(archive*, archive_entry*);
	static int archive_write_data(archive*, const void*, size_t);
	static int archive_write_finish_entry(archive*);
	static int archive_write_close(archive*);

public:
	static archive_entry* archive_entry_new();
	static int archive_entry_free(archive_entry*);

	static void archive_entry_set_uid(archive_entry* entry, uint64_t);
	static void archive_entry_set_uname(archive_entry* entry, const char*);
	static void archive_entry_set_gid(archive_entry* entry, uint64_t);
	static void archive_entry_set_gname(archive_entry* entry, const char*);
	static void archive_entry_set_filetype(archive_entry* entry, unsigned int);

	static void archive_entry_set_perm(archive_entry* entry, int permission);
	static void archive_entry_set_size(archive_entry* entry, size_t );
	static void archive_entry_set_pathname(archive_entry* entry, const char*);
	
	static size_t archive_entry_size(archive_entry* entry);
	static const char* archive_entry_pathname(archive_entry* entry);

public:
	static std::string archive_error_string(archive* archive);


private:
	static void _check();

private:
	class Interface
	{
	public:
		archive* (*archive_read_new)();
		int (*archive_read_free)(archive*);
		
		int (*archive_read_support_filter_all)(archive* );
		int (*archive_read_support_format_all)(archive* );
		int (*archive_read_open_FILE)(archive* , FILE* file);

		int (*archive_read_next_header)(archive*, archive_entry**);
	
		size_t (*archive_read_data)(archive*, void* buffer, size_t bytes);
		
	public:
		archive* (*archive_write_new)();
		int (*archive_write_free)(archive*);
	
		int (*archive_write_set_compression_gzip)(archive* );
		int (*archive_write_set_format_pax_restricted)(archive* );
		int (*archive_write_open_FILE)(archive* , FILE* file);
	
		int (*archive_write_header)(archive*, archive_entry*);
		int (*archive_write_data)(archive*, const void*, size_t);
		int (*archive_write_finish_entry)(archive*);
		int (*archive_write_close)(archive*);
	
	public:
		archive_entry* (*archive_entry_new)();
		int (*archive_entry_free)(archive_entry*);
	
		void (*archive_entry_set_uid)(archive_entry* entry, uint64_t);
		void (*archive_entry_set_uname)(archive_entry* entry, const char*);
		void (*archive_entry_set_gid)(archive_entry* entry, uint64_t);
		void (*archive_entry_set_gname)(archive_entry* entry, const char*);
		void (*archive_entry_set_filetype)(archive_entry* entry, unsigned int);

		void (*archive_entry_set_perm)(archive_entry* entry, int permission);
		void (*archive_entry_set_size)(archive_entry* entry, size_t );
		void (*archive_entry_set_pathname)(archive_entry* entry, const char*);
		
		size_t (*archive_entry_size)(archive_entry* entry);
		const char* (*archive_entry_pathname)(archive_entry* entry);
	
	public:
		const char* (*archive_error_string)(archive* archive);
	
	public:
		/*! \brief The constructor zeros out all of the pointers */
		Interface();
		
		/*! \brief The destructor closes dlls */
		~Interface();
		/*! \brief Load the library */
		void load();
		/*! \brief Has the library been loaded? */
		bool loaded() const;
		/*! \brief unloads the library */
		void unload();
		
	private:
		void* _library;
	};
	
private:
	static Interface _interface;
	
};

}

}



