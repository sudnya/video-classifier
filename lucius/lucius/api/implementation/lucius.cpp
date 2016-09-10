
// Lucius Includes
#include <lucius/api/interface/lucius.h>

static thread_local std::string lastErrorMessage;

extern "C" CString luciusGetLastError()
{
    return lastErrorMessage.c_str();
}

extern "C" int luciusCreateDataItem(DataItem* item, CString type)
{
    *item = new LuciusDataItem(type);

    return LUCIUS_SUCCESS;
}

extern "C" int luciusSetDataItemContents(DataItem item, const void* data, size_t size)
{
    reinterpret_cast<LuciusDataItem*>(item)->setContents(data, size);

    return LUCIUS_SUCCESS;
}

extern "C" int luciusGetDataItemType(CString* type, DataItem item)
{
    *type = reinterpret_cast<LuciusDataItem*>(item)->getType().c_str();

    return LUCIUS_SUCCESS;
}

extern "C" int luciusGetDataItemContentsSize(size_t* size, DataItem item)
{
    *size = reinterpret_cast<LuciusDataItem*>(item)->getSize();

    return LUCIUS_SUCCESS;
}

extern "C" int luciusGetDataItemContentsAsString(CString* data, DataItem item)
{
    *data = reinterpret_cast<char*>(reinterpret_cast<LuciusDataItem*>(item)->getData());

    return LUCIUS_SUCCESS;
}

extern "C" int luciusDestroyDataItem(DataItem item)
{
    delete item;

    return LUCIUS_SUCCESS;
}

extern "C" int luciusLoadModel(Model* m, CString filename)
{
    try
    {
        *m = new model::Model(filename);
    }
    catch(std::exception& e)
    {
        *m = nullptr;
        lastErrorMessage = e.what();
        return LUCIUS_ERROR;
    }

    return LUCIUS_SUCCESS;
}

static std::string getInputType(DataItem item)
{
    return reinterpret_cast<LuciusDataItem*>(item)->getType();
}

static std::unique_ptr<ResultProcessor> getResultProcessorForInputType(
    const std::string& inputType)
{
    if(isText(inputType))
    {
        return ResultProcessorFactory::create("LabelResultProcessor");
    }

    throw std::invalid_argument("Invalid input type '" + inputType + "'");
}

static unique_ptr<InputDataProducer> getProducerForInputType(const std::string& inputType)
{
    if(isText(inputType))
    {
        return InputDataProducerFactory::create("InputTextDataProducer");
    }
    else if(isAudio(inputType))
    {
        return InputAudioDataProducer::create("InputAudioDataProducer");
    }
    else if(isImage(inputType))
    {
        return InputVisualDataProducer::create("InputVisualDataProducer");
    }

    throw std::invalid_argument("Invalid input type '" + inputType + "'");
}

extern "C" int luciusInfer(DataItem* output, Model m, DataItem input)
{
    if(m == nullptr)
    {
        lastErrorMessage = "Invalid model.";

        return LUCIUS_ERROR;
    }

    if(input == nullptr)
    {
        lastErrorMessage = "Invalid input data item.";

        return LUCIUS_ERROR;
    }

    try
    {
        auto engine = EngineFactory::create("ClassifierEngine");
        auto type = getInputType(input);

        auto resultProcessor = getResultProcessorForInputType(type);
        auto producer        = getProducerForInputType(type);

        engine->setBatchSize(1);
        engine->setModel(reinterpret_cast<model::Model*>(m));
        engine->setResultProcessor(resultProcessor.get());

        engine->runOnDataProducer(*producer);

        setResult(output, resultProcessor, type);
    }
    catch(std::exception& e)
    {
        *output = nullptr;
        lastErrorMessage = e.what();
        return LUCIUS_ERROR;
    }

    return LUCIUS_SUCCESS;
}

extern "C" int luciusDestroyModel(Model model)
{
    delete model;

    return LUCIUS_SUCCESS;
}


