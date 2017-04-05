import os
import configparser

def CheckFileExist(filePath,
                   throwError = True):
    if not os.path.exists(filePath):
        print ('%s does not exist ! !' % (filePath))
        if throwError is True:
            assert ('%s does not exist ! !' % (filePath))
        return False
    return True

def CheckPathExist(path,
                   throwError = True):
    if not os.path.isdir(path):
        print ('%s does not exist ! Make sure you choose right location !' % (path))
        if throwError is True:
            assert ('%s does not exist ! Make sure you choose right location !' % (path))
        return False
    return True

def CheckNotNone(something,
                 name,
                 throwError = True):
    if something is None:
        print ('%s can not be none !' % (name))
        if throwError is True:
            assert ('%s can not be none !' % (name))
        return False
    return True

def GetAllSubfoldersPath(path):
    # Check parameter
    CheckNotNone(path, 'path'); CheckPathExist(path)

    allSubFoldersPath = sorted([os.path.join(os.path.abspath(path), name + '/') for name in os.listdir(path)
                                if CheckPathExist(os.path.join(path, name))])
    return allSubFoldersPath

def GetAllFiles(path):
    # Check parameters
    CheckNotNone(path); CheckPathExist(path)

    allFilesPath = sorted([os.path.join(os.path.abspath(path), filename) for filename in os.listdir(path)
                           if CheckFileExist(os.path.join(path, filename))])
    return allFilesPath

def ReadFile(filePath):
    # Check file exist
    CheckFileExist(filePath)

    file = open(filePath)
    allData = tuple(file)
    file.close()

    return allData

def ReadFileIni(filePath):
    # Check file exist
    CheckFileExist(filePath)

    config = configparser.ConfigParser()
    config.sections()
    config.read(filePath)
    return config