
// Lucius Includes
#include <lucius/api/interface/lucius.h>

extern "C" const char* luciusGetLastError()
{
    return "C Interface is not implemented.";
}

extern "C" int luciusCreateDataItem(DataItem* item, CString type)
{
    item = nullptr;

    return LUCIUS_ERROR;
}

extern "C" int luciusSetDataItemContents(DataItem item, const void* data, size_t size)
{
    return LUCIUS_ERROR;
}

extern "C" int luciusGetDataItemType(CString* type, DataItem item)
{
    type = nullptr;

    return LUCIUS_ERROR;
}

extern "C" int luciusGetDataItemContentsSize(size_t* size, DataItem item)
{
    return LUCIUS_ERROR;
}

extern "C" int luciusGetDataItemContentsAsString(CString* data, DataItem item)
{
    data = nullptr;

    return LUCIUS_ERROR;
}

extern "C" int luciusDestroyDataItem(DataItem item)
{
    return LUCIUS_ERROR;
}

extern "C" int luciusLoadModel(Model* model, CString filename)
{
    model = nullptr;

    return LUCIUS_ERROR;
}

extern "C" int luciusInfer(DataItem* output, Model model, DataItem input)
{
    output = nullptr;

    return LUCIUS_ERROR;
}

extern "C" int luciusDestroyModel(Model model)
{
    return LUCIUS_ERROR;
}


