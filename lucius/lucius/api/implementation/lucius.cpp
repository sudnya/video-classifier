
// Lucius Includes
#include <lucius/api/interface/lucius.h>

#include <lucius/engine/interface/EngineFactory.h>
#include <lucius/engine/interface/Engine.h>

#include <lucius/model/interface/Model.h>

#include <lucius/results/interface/ResultProcessorFactory.h>
#include <lucius/results/interface/ResultProcessor.h>

#include <lucius/input/interface/InputDataProducer.h>
#include <lucius/input/interface/InputDataProducerFactory.h>

#include <lucius/network/interface/NeuralNetwork.h>

#include <lucius/audio/interface/Audio.h>

#include <lucius/text/interface/Text.h>

#include <lucius/video/interface/Video.h>
#include <lucius/video/interface/Image.h>

#include <lucius/util/interface/debug.h>


// Standard Library Includes
#include <string>
#include <thread>
#include <mutex>
#include <map>
#include <vector>

static std::mutex messageMutex;
static std::map<std::thread::id, std::string> lastErrorMessage;

class LuciusDataItem
{
public:
    LuciusDataItem(const std::string& type)
    : _type(type)
    {

    }

public:
    void setContents(const void* data, size_t size)
    {
        _contents.resize(size);
        std::memcpy(_contents.data(), data, size);
    }

    const std::string& getType() const
    {
        return _type;
    }

    size_t getSize() const
    {
        return _contents.size();
    }

    void* getData()
    {
        return _contents.data();
    }

private:
    typedef std::vector<int8_t> ByteVector;

private:
    std::string _type;
    ByteVector  _contents;

};

static std::string& getLastErrorMessage()
{
    std::unique_lock<std::mutex> lock(messageMutex);

    return lastErrorMessage[std::this_thread::get_id()];
}

extern "C" CString luciusGetLastError()
{
    return getLastErrorMessage().c_str();
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
    delete reinterpret_cast<LuciusDataItem*>(item);

    return LUCIUS_SUCCESS;
}

extern "C" int luciusLoadModel(Model* m, CString filename)
{
    try
    {
        auto* newModel = new lucius::model::Model(filename);
        newModel->load();
        *m = newModel;
    }
    catch(std::exception& e)
    {
        *m = nullptr;
        getLastErrorMessage() = e.what();
        return LUCIUS_ERROR;
    }

    return LUCIUS_SUCCESS;
}

static std::string getInputType(DataItem item)
{
    return reinterpret_cast<LuciusDataItem*>(item)->getType();
}

static std::unique_ptr<lucius::results::ResultProcessor> getResultProcessorForInputType(
    const std::string& inputType)
{
    return lucius::results::ResultProcessorFactory::create("LabelResultProcessor");
}

static std::unique_ptr<lucius::input::InputDataProducer>
    getProducerForInputType(const std::string& inputType)
{
    if(lucius::text::Text::isPathATextFile(inputType))
    {
        return lucius::input::InputDataProducerFactory::create("InputTextDataProducer");
    }
    else if(lucius::audio::Audio::isPathAnAudioFile(inputType))
    {
        return lucius::input::InputDataProducerFactory::create("InputAudioDataProducer");
    }
    else if(lucius::video::Image::isPathAnImageFile(inputType))
    {
        return lucius::input::InputDataProducerFactory::create("InputVisualDataProducer");
    }

    throw std::invalid_argument("Invalid input type for data producer '" + inputType + "'");
}

static std::string sanitizeLabel(const std::string& label)
{
    std::string result;

    for(auto& c : label)
    {
        if(c != '\0')
        {
            result.push_back(c);
        }
    }

    return result;
}

static void setResult(DataItem* output, lucius::results::ResultProcessor* processor,
    const std::string& inputType)
{
    auto label = sanitizeLabel(processor->toString());

    auto dataItem = new LuciusDataItem("string");

    dataItem->setContents(label.data(), label.size() + 1);

    *output = dataItem;
}

extern "C" int luciusInfer(DataItem* output, Model m, DataItem input)
{
    if(m == nullptr)
    {
        getLastErrorMessage() = "Invalid model.";

        return LUCIUS_ERROR;
    }

    if(input == nullptr)
    {
        getLastErrorMessage() = "Invalid input data item.";

        return LUCIUS_ERROR;
    }

    try
    {
        lucius::util::log("Lucius") << "Creating classifier engine.\n";
        auto engine = lucius::engine::EngineFactory::create("ClassifierEngine");
        auto type = getInputType(input);

        lucius::util::log("Lucius") << " for input type '" << type << "'\n";

        lucius::util::log("Lucius") << "Creating input producer.\n";
        auto producer = getProducerForInputType(type);
        producer->setModel(reinterpret_cast<lucius::model::Model*>(m));

        lucius::util::log("Lucius") << "Adding raw sample to the producer.\n";
        producer->addRawSample(reinterpret_cast<LuciusDataItem*>(input)->getData(),
            reinterpret_cast<LuciusDataItem*>(input)->getSize(), type, "");

        lucius::util::log("Lucius") << "Setting up the classifier engine.\n";
        engine->setBatchSize(1);
        engine->setModel(reinterpret_cast<lucius::model::Model*>(m));
        engine->setResultProcessor(getResultProcessorForInputType(type));

        lucius::util::log("Lucius") << "Running the engine.\n";
        engine->runOnDataProducer(*producer);

        lucius::util::log("Lucius") << "Extracting the result from the result processor.\n";
        setResult(output, engine->getResultProcessor(), type);
    }
    catch(std::exception& e)
    {
        *output = nullptr;
        getLastErrorMessage() = e.what();
        return LUCIUS_ERROR;
    }

    return LUCIUS_SUCCESS;
}

extern "C" int luciusDestroyModel(Model model)
{
    delete reinterpret_cast<lucius::model::Model*>(model);

    return LUCIUS_SUCCESS;
}

extern "C" int luciusEnableAllLogs()
{
    lucius::util::enableAllLogs();

    return LUCIUS_SUCCESS;
}


