
import Model
import ctypes

def checkStatus(status, liblucius):
    if value != 0:
        errorMessage = liblucius.luciusGetLastError()

        raise RuntimeError(errorMessage)

def getLibrary():
    ctypes.LoadLibrary("liblucius.so")
    return ctypes.CDLL("liblucius.so")

def loadModelHandle(path, liblucius):
    cPath = ctypes.c_char_p(path)
    cHandle = ctypes.c_void_p(None)

    status = ctypes.c_int(liblucius.luciusLoadModel(ctypes.pointer(cHandle), cPath))

    checkErrors(status, liblucius)

    return cHandle

def inferWithHandle(model, data, typeName, liblucius):

def cleanupHandle(model, liblucius):
    status = ctypes.c_int(liblucius.luciusDestroyModel(model))

    checkErrors(status, liblucius)


class Model:
    def __init__(self, modelPath):
        self.path = modelPath

    def __enter__(self)
        self.liblucius = getLibrary()
        self.handle = loadModelHandle(self.path, self.liblucius)

    def __exit__(self, exc_type, exc_value, traceback):
        cleanupHandle(self.handle, self.liblucius)

    def infer(self, data, typeName):
        inferWithHandle(self.handle, data, typeName, self.liblucius)


