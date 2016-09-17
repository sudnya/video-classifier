
import ctypes
import logging
import platform

def checkStatus(status, liblucius):
    if status != 0:
        luciusGetLastError = liblucius.luciusGetLastError
        luciusGetLastError.restype = ctypes.c_char_p

        errorMessage = luciusGetLastError()

        raise RuntimeError("Lucius C function call failed with error: '" + str(errorMessage) + "'")

def getSharedLibraryExtension():
    if platform.system() == 'Darwin':
        return '.dylib'
    else:
        return '.so'

def getLibrary():
    ctypes.cdll.LoadLibrary("liblucius" + getSharedLibraryExtension())
    return ctypes.CDLL("liblucius" + getSharedLibraryExtension())

def loadModelHandle(path, liblucius):
    cPath = ctypes.c_char_p(path)
    cHandle = ctypes.c_void_p(None)

    status = liblucius.luciusLoadModel(ctypes.pointer(cHandle), cPath)

    checkStatus(status, liblucius)

    return cHandle

def checkOutputType(typename):
    if typename != "string":
        raise RuntimeError("Expecting string data type as output of inference, but got '" +
            typename + "'")

def checkOutputLength(length, contents):
    if (len(contents) + 1) != length:
        raise RuntimeError("Output data item from inference length  '" + str(length) +
            "' does not match expected length '" + str(len(contents)) + "'")

def inferWithHandle(model, data, typeName, liblucius, logger):
    luciusCreateDataItem = liblucius.luciusCreateDataItem
    luciusSetDataItemContents = liblucius.luciusSetDataItemContents
    luciusDestroyDataItem = liblucius.luciusDestroyDataItem
    luciusInfer = liblucius.luciusInfer
    luciusGetDataItemType = liblucius.luciusGetDataItemType
    luciusGetDataItemContentsSize = liblucius.luciusGetDataItemContentsSize
    luciusGetDataItemContentsAsString = liblucius.luciusGetDataItemContentsAsString

    cInputDataItem = ctypes.c_void_p(None)
    cOutputDataItem = ctypes.c_void_p(None)

    try:
        logger.info("Creating input data item")
        checkStatus(luciusCreateDataItem(ctypes.pointer(cInputDataItem), typeName),
            liblucius)

        logger.info(" setting contents")
        checkStatus(luciusSetDataItemContents(cInputDataItem, data, ctypes.c_ulonglong(len(data))),
            liblucius)

        logger.info("Running inference")
        checkStatus(luciusInfer(ctypes.pointer(cOutputDataItem), model, cInputDataItem),
            liblucius)

        logger.info("Getting output data item type")
        cOutputDataItemType = ctypes.c_char_p(None)
        checkStatus(luciusGetDataItemType(ctypes.pointer(cOutputDataItemType), cOutputDataItem),
            liblucius)
        outputDataItemType = str(cOutputDataItemType.value)
        logger.info(" type is " + outputDataItemType)

        logger.info("Getting output data item size")
        cOutputDataItemSize = ctypes.c_ulonglong(0)
        checkStatus(luciusGetDataItemContentsSize(
            ctypes.pointer(cOutputDataItemSize), cOutputDataItem), liblucius)
        outputDataItemSize = cOutputDataItemSize.value
        logger.info(" size is " + str(outputDataItemSize))

        logger.info("Getting output data item contents")
        cOutputDataItemContents = ctypes.c_char_p(None)
        checkStatus(luciusGetDataItemContentsAsString(ctypes.pointer(cOutputDataItemContents),
            cOutputDataItem), liblucius)
        outputDataItemContents = str(cOutputDataItemContents.value)
        logger.info(" result is '" + outputDataItemContents + "'")

    except Exception as e:
        raise e
    finally:
        checkStatus(luciusDestroyDataItem(cInputDataItem), liblucius)
        checkStatus(luciusDestroyDataItem(cOutputDataItem), liblucius)

    checkOutputType(outputDataItemType)
    checkOutputLength(outputDataItemSize, outputDataItemContents)

    return outputDataItemContents

def cleanupHandle(model, liblucius):
    status = liblucius.luciusDestroyModel(model)

    checkStatus(status, liblucius)

def enableAllLogs(liblucius, logger):
    logger.info("Enabling lucius logs")
    status = liblucius.luciusEnableAllLogs()

    checkStatus(status, liblucius)

class Model:
    def __init__(self, modelPath, isVerbose=False):
        self.path = modelPath
        self.isVerbose = isVerbose

        self.createLogger()

    def __enter__(self):
        self.logger.info("Loading lucius C library")
        self.liblucius = getLibrary()
        self.logger.info("Successfully loaded lucius C library")

        self.logger.info("Loading model handle")
        self.handle = loadModelHandle(self.path, self.liblucius)
        self.logger.info("Successfully loaded model handle " + str(self.handle))

        if self.isVerbose:
            enableAllLogs(self.liblucius, self.logger)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cleanupHandle(self.handle, self.liblucius)

    def infer(self, data, typeName):
        self.logger.info("Running inference on data with type '" + str(typeName) + "'")
        return inferWithHandle(self.handle, data, typeName, self.liblucius, self.logger)

    def createLogger(self):
        self.logger = logging.getLogger("Downloader")

        # create console handler and set level to debug
        ch = logging.StreamHandler()

        if self.isVerbose:
            ch.setLevel(logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.WARNING)
            self.logger.setLevel(logging.WARNING)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)


