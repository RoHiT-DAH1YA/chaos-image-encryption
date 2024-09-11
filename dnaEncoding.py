import numpy as np
from PIL import Image

allRules = [['A', 'G', 'C', 'T'],
            ['A', 'C', 'G', 'T'],
            ['G', 'A', 'T', 'C'],
            ['G', 'T', 'A', 'C'],
            ['C', 'A', 'T', 'G'],
            ['C', 'T', 'A', 'G'],
            ['T', 'G', 'C', 'A'],
            ['T', 'C', 'G', 'A']]


def getDnaString(imageObject: Image, ruleNumber=0):
    pixelArray = np.asarray(imageObject)
    return getDnaStringFromNumpyArray(pixelArray,ruleNumber)


def getDnaStringFromNumpyArray(imageArray:np.ndarray, ruleNumber=0):
    pixelArray = imageArray.flatten()
    ruleUsed = allRules[ruleNumber]
    binlist = [[int('11000000', 2), int('00110000', 2), int('00001100', 2), int('00000011', 2)], [6, 4, 2, 0]]

    dnastring = ''.join(
        [ruleUsed[(num & binlist[0][index]) >> binlist[1][index]] for num in pixelArray for index in range(0, 4)])
    return dnastring


def getImageFromDnaString(dnaString, width, height, ruleNumber=0):
    if width * height * 3 * 4 != len(dnaString):
        print("Size doesn't match string length.", len(dnaString))
        exit(1)

    newImageArray = getArrayFromDnaString(dnaString, width * height * 3, ruleNumber)
    newImageArray = newImageArray.reshape(height, width, 3)
    newImage = Image.fromarray(newImageArray.astype(np.uint8))
    return newImage


def getArrayFromDnaString(dnaString, size, ruleNumber=0):
    newImageArray = np.zeros(size)
    ruleUsed = allRules[ruleNumber]
    dict = {string: index for index, string in enumerate(ruleUsed)}
    i = 0
    while i < len(dnaString):
        pixel = (dict[dnaString[i]] << 6)
        i += 1
        pixel += (dict[dnaString[i]] << 4)
        i += 1
        pixel += (dict[dnaString[i]] << 2)
        i += 1
        pixel += dict[dnaString[i]]
        newImageArray[i // 4] = pixel
        i += 1
    return newImageArray


def dnaXor(dnaString1, dnaString2):
    xortable = {'A': {'A': 'A',
                      'G': 'G',
                      'C': 'C',
                      'T': 'T'},
                'G': {'A': 'G',
                      'G': 'A',
                      'C': 'T',
                      'T': 'C'},
                'C': {'A': 'C',
                      'G': 'T',
                      'C': 'A',
                      'T': 'G'},
                'T': {'A': 'T',
                      'G': 'C',
                      'C': 'G',
                      'T': 'A'}
                }
    xorString = ''.join([xortable[dnaString1[i]][dnaString2[i]] for i in range(len(dnaString1))])
    return xorString


def dnaAddition(dnaString1, dnaString2):
    addTable = {'A': {'A': 'A',
                      'G': 'G',
                      'C': 'C',
                      'T': 'T'},
                'G': {'A': 'G',
                      'G': 'C',
                      'C': 'T',
                      'T': 'A'},
                'C': {'A': 'C',
                      'G': 'T',
                      'C': 'A',
                      'T': 'G'},
                'T': {'A': 'T',
                      'G': 'A',
                      'C': 'G',
                      'T': 'C'}
                }
    addString = ''.join([addTable[dnaString1[i]][dnaString2[i]] for i in range(len(dnaString1))])
    return addString


def dnaSubtraction(dnaString1, dnaString2):
    subTable = {'A': {'A': 'A',
                      'G': 'T',
                      'C': 'C',
                      'T': 'G'},
                'G': {'A': 'G',
                      'G': 'A',
                      'C': 'T',
                      'T': 'C'},
                'C': {'A': 'C',
                      'G': 'G',
                      'C': 'A',
                      'T': 'T'},
                'T': {'A': 'T',
                      'G': 'C',
                      'C': 'G',
                      'T': 'A'}
                }
    subString = ''.join([subTable[dnaString1[i]][dnaString2[i]] for i in range(len(dnaString1))])
    return subString


def dnaComplement(danString):
    complementTable = {'A': 'T',
                      'G': 'C',
                      'C': 'G',
                      'T': 'A'}
    complementString = ''.join([complementTable[ch] for ch in danString])
    return complementString


def isValidDnaString(dnaString):
    for i in range(len(dnaString)):
        if dnaString[i] not in ['A', 'G', 'C', 'T']:
            print(i)
            return
