
#pragma once

typedef void* Model;
typedef void* DataItem;
typedef const char* CString;

enum LuciusStatus
{
    LUCIUS_SUCCESS = 0,
    LUCIUS_ERROR   = 1
};

/*! \brief Get the error message associated with the last call to a lucius API function. */
extern "C" const char* luciusGetLastError();

/*! \brief Create a new data item using the specified type.

    \param item A pointer to a DataItem handle.
                It will be initialized to a new data item on success or nullptr on failure.

    \param type The type of data item to create.
                Currrently supported types (string, mp3, wav).

    \returns Status code indicating success or failure.
*/
extern "C" int luciusCreateDataItem(DataItem* item, CString type);

/*! \brief Set the contents of a DataItem.

    \param item A handle to the DataItem being set.

    \param type The type of data item to create.
                Currrently supported types (string, mp3, wav).

    \returns Status code indicating success or failure.
*/
extern "C" int luciusSetDataItemContents(DataItem item, const void* data, size_t size);

/*! \brief Get the type of a DataItem.

    \param type A pointer to a type string that will be returned on success.  Nullptr will be
               return on failure.  Note that the lifetime of this string is until the
               corresponding DataItem is mutated.

    \param item The DataItem to read the type from.

    \returns Status code indicating success or failure.
*/
extern "C" int luciusGetDataItemType(CString* type, DataItem item);

/*! \brief Get the size of the contents of a DataItem.

    \param size A pointer to a variable to store the size (in bytes) into.

    \param item The DataItem to get the contents size from.

    \returns Status code indicating success or failure.
*/
extern "C" int luciusGetDataItemContentsSize(size_t* size, DataItem item);

/*! \brief Get the data contents of a DataItem.

    \param type A pointer to a data string that will be returned on success.  Nullptr will be
               return on failure.  Note that the lifetime of this string is until the
               corresponding DataItem is mutated.  The size of the string is exactly
               the value returned by luciusGetDataItemContentsSize.

    \param item The DataItem to read the contents from.

    \returns Status code indicating success or failure.
*/
extern "C" int luciusGetDataItemContentsAsString(CString* data, DataItem item);

/*! \brief Destroy a DataItem.

    \param item The DataItem to destroy.

    \returns Status code indicating success or failure.
*/
extern "C" int luciusDestroyDataItem(DataItem item);

/*! \brief Load an existing model from the specified file.

    \param model A pointer to a Model handle.  It will be initialized to a newly created model on
                 success, or nullptr on failure.
    \param filename A string representing a path to an existing model.

    \returns Status indicating success or failure.
 */
extern "C" int luciusLoadModel(Model* model, CString filename);

/*! \brief Run inference using the specified model on the specified data item.  A new DataItem
           will be created.

    \param output The newly created DataItem containing the result of model inference.
                  Note that this item should eventually be destroyed using luciusDestroyDataItem.

    \param model An existing model that will be run on the input DataItem.

    \param input Input DataItem that will be run through the model.

    \returns Status indicating success or failure.
*/
extern "C" int luciusInfer(DataItem* output, Model model, DataItem input);

/*! \brief Destroy a previously created model.

    \returns Status indicating success or failure.
 */
extern "C" int luciusDestroyModel(Model model);



