import random
import numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def histogramRGB(image:Image, imageName='', saveFolder='./'):
    histogramRGBfromArray(np.asarray((image), dtype=np.uint8), imageName, saveFolder)


def histogramRGBfromArray(imageArray:np.ndarray, imageName='', saveFolder='./'):
    plt.close()
    plt.figure()
    plt.hist(imageArray[:, :, 0].ravel(), 256, facecolor='red')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(saveFolder + imageName + "_redHistogram.png", orientation='portrait', dpi=300)
    plt.close()

    plt.figure()
    plt.hist(imageArray[:, :, 1].ravel(), 256, facecolor='green')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(saveFolder + imageName + "_greenHistogram.png", orientation='portrait', dpi=300)
    plt.close()

    plt.figure()
    plt.hist(imageArray[:, :, 2].ravel(), 256, facecolor='blue')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(saveFolder + imageName + "_blueHistogram.png", orientation='portrait', dpi=300)
    plt.close()

def npcrImage(image1: Image, image2: Image):
    if image1.size != image2.size:
        print("Error: Image dimensions do not match.")
        return None
    imageArray1 = np.asarray(image1)
    imageArray2 = np.asarray(image2)
    return npcrArray(imageArray1, imageArray2)


def npcrArray(imageArray1: np.ndarray, imageArray2: np.ndarray):
    # Check dimensions of input arrays
    if imageArray1.shape != imageArray2.shape:
        print("Error: imageArray dimensions do not match.")
        return None

    # Calculate NPCR for each color channel
    diffR = np.sum(imageArray1[:, :, 0] != imageArray2[:, :, 0])
    diffG = np.sum(imageArray1[:, :, 1] != imageArray2[:, :, 1])
    diffB = np.sum(imageArray1[:, :, 2] != imageArray2[:, :, 2])

    # Calculate NPCR percentages
    noOfPixels = imageArray1.shape[0] * imageArray1.shape[1]
    resultR = diffR * 100 / noOfPixels
    resultG = diffG * 100 / noOfPixels
    resultB = diffB * 100 / noOfPixels

    # Calculate average NPCR and overall NPCR (resultRGB)
    resultRGB = (resultR + resultG + resultB) / 3.0

    return resultRGB, resultR, resultG, resultB

def ucaiImage(image1: Image, image2: Image):
    return ucaiArray(np.asarray(image1), np.asarray(image2))


def ucaiArray(imageArray1: np.ndarray, imageArray2: np.ndarray):
    # Check dimensions of input arrays
    if imageArray1.shape != imageArray2.shape:
        print("Error: imageArray dimensions do not match.")
        return None

    # Calculate absolute differences for each color channel
    absDiffR = np.abs(imageArray1[:, :, 0] - imageArray2[:, :, 0])
    absDiffG = np.abs(imageArray1[:, :, 1] - imageArray2[:, :, 1])
    absDiffB = np.abs(imageArray1[:, :, 2] - imageArray2[:, :, 2])

    # Calculate mean absolute differences normalized by maximum possible value (255)
    noOfPixels = imageArray1.shape[0] * imageArray1.shape[1]
    resultR = np.sum(absDiffR) * 100 / (noOfPixels * 255.0)
    resultG = np.sum(absDiffG) * 100 / (noOfPixels * 255.0)
    resultB = np.sum(absDiffB) * 100 / (noOfPixels * 255.0)

    # Calculate average UCAI and overall UCAI (resultRGB)
    resultRGB = (resultR + resultG + resultB) / 3.0

    return resultRGB, resultR, resultG, resultB

def correlationImage(image: Image):
    def findDeviation(imageArray, expectation):
        return np.mean(np.square(imageArray - expectation))

    def calculateCovariance(imageArray1, mean1, imageArray2, mean2):
        return np.mean((imageArray1 - mean1) * (imageArray2 - mean2))

    def correlation(array1, array2):
        exp1 = np.mean(array1)
        exp2 = np.mean(array2)

        d1 = findDeviation(array1, exp1)
        d2 = findDeviation(array2, exp2)

        cov = calculateCovariance(array1, exp1, array2, exp2)
        cor = cov / (np.sqrt(d1) * np.sqrt(d2))
        return cor

    def combineRGB(redArray, greenArray, blueArray):
        return np.stack((redArray, greenArray, blueArray), axis=-1).reshape(-1, 3)

    imageArray = np.asarray(image)
    height, width = imageArray.shape[:2]
    redArray = imageArray[:, :, 0]
    greenArray = imageArray[:, :, 1]
    blueArray = imageArray[:, :, 2]

    # Horizontal
    redHorArray = redArray.flatten()
    greenHorArray = greenArray.flatten()
    blueHorArray = blueArray.flatten()
    rgbHorArray = combineRGB(redHorArray, greenHorArray, blueHorArray)

    print("Horizontal : ")
    print("Red:", correlation(redHorArray, np.roll(redHorArray, -1)))
    print("Green:", correlation(greenHorArray, np.roll(greenHorArray, -1)))
    print("Blue:", correlation(blueHorArray, np.roll(blueHorArray, -1)))
    print("Collective RGB:", correlation(rgbHorArray, np.roll(rgbHorArray, -1, axis=0)))

    # Vertical
    redVerArray = np.array([redArray[h][w] for w in range(redArray.shape[1]) for h in range(redArray.shape[0])])
    greenVerArray = np.array([greenArray[h][w] for w in range(greenArray.shape[1]) for h in range(greenArray.shape[0])])
    blueVerArray = np.array([blueArray[h][w] for w in range(blueArray.shape[1]) for h in range(blueArray.shape[0])])
    rgbVerArray = combineRGB(redVerArray, greenVerArray, blueVerArray)

    print("Vertical : ")
    print("Red:", correlation(redVerArray, np.roll(redVerArray, -1)))
    print("Green:", correlation(greenVerArray, np.roll(greenVerArray, -1)))
    print("Blue:", correlation(blueVerArray, np.roll(blueVerArray, -1)))
    print("Collective RGB:", correlation(rgbVerArray, np.roll(rgbVerArray, -1, axis=0)))

    # Diagonal
    redDiagArray = np.concatenate((
        np.array([redArray[h + t][t] for h in range(height - 1, -1, -1) for t in range(0, min(min(height, width), (height - h)))]),
        np.array([redArray[t][w + t] for w in range(1, width) for t in range(0, min(min(height, width), (width - w)))])))
    greenDiagArray = np.concatenate((
        np.array([greenArray[h + t][t] for h in range(height - 1, -1, -1) for t in range(0, min(min(height, width), (height - h)))]),
        np.array([greenArray[t][w + t] for w in range(1, width) for t in range(0, min(min(height, width), (width - w)))])))
    blueDiagArray = np.concatenate((
        np.array([blueArray[h + t][t] for h in range(height - 1, -1, -1) for t in range(0, min(min(height, width), (height - h)))]),
        np.array([blueArray[t][w + t] for w in range(1, width) for t in range(0, min(min(height, width), (width - w)))])))
    rgbDiagArray = combineRGB(redDiagArray, greenDiagArray, blueDiagArray)

    print("Diagonal : ")
    print("Red:", correlation(redDiagArray, np.roll(redDiagArray, -1)))
    print("Green:", correlation(greenDiagArray, np.roll(greenDiagArray, -1)))
    print("Blue:", correlation(blueDiagArray, np.roll(blueDiagArray, -1)))
    print("Collective RGB:", correlation(rgbDiagArray, np.roll(rgbDiagArray, -1, axis=0)))

    return correlation

# Example usage:
# from PIL import Image
# image = Image.open('path_to_image.png')
# correlationImage(image)



def correlationImage1(image: Image):
    def findDeviation(imageArray, expectation):
        return np.mean(np.square(imageArray - expectation))

    def calculateCovariance(imageArray1, mean1, imageArray2, mean2):
        return np.mean((imageArray1 - mean1) * (imageArray2 - mean2))

    def correlation(array1, array2):
        exp1 = np.mean(array1)
        exp2 = np.mean(array2)

        d1 = findDeviation(array1, exp1)
        d2 = findDeviation(array2, exp2)

        cov = calculateCovariance(array1, exp1, array2, exp2)
        cor = cov / (np.sqrt(d1) * np.sqrt(d2))
        return cor

    imageArray = np.asarray(image)
    height, width = len(imageArray), len(imageArray[0])
    redArray = imageArray[:,:,0]
    greenArray = imageArray[:,:,1]
    blueArray = imageArray[:,:,2]


    # horizontal
    redHorArray = redArray.flatten()
    greenHorArray = greenArray.flatten()
    blueHorArray = blueArray.flatten()

    print("Horizontal : ")
    print(correlation(redHorArray, np.roll(redHorArray, -1)))
    print(correlation(greenHorArray, np.roll(greenHorArray, -1)))
    print(correlation(blueHorArray, np.roll(blueHorArray, -1)))

    #vertival
    redVerArray = np.array([redArray[h][w]for w in range(redArray.shape[1]) for h in range(redArray.shape[0])])
    greenVerArray = np.array([greenArray[h][w]for w in range(greenArray.shape[1]) for h in range(greenArray.shape[0])])
    blueVerArray = np.array([blueArray[h][w]for w in range(blueArray.shape[1]) for h in range(blueArray.shape[0])])

    print("Vertical : ")
    print(correlation(redVerArray, np.roll(redVerArray, -1)))
    print(correlation(greenVerArray, np.roll(greenVerArray, -1)))
    print(correlation(blueVerArray, np.roll(blueVerArray, -1)))
    #diagonal
    redDiagArray = np.concatenate((np.array([redArray[h+t][t] for h in range(height-1, -1, -1) for t in range(0, min(min(height,width), (height-h)))]),
        np.array([redArray[t][w+t] for w in range(1, width) for t in range(0, min(min(height,width), (width-w)))])))
    greenDiagArray = np.concatenate((np.array([greenArray[h+t][t] for h in range(height-1, -1, -1) for t in range(0, min(min(height,width), (height-h)))]),
        np.array([greenArray[t][w+t] for w in range(1, width) for t in range(0, min(min(height,width), (width-w)))])))
    redDiagArray = np.concatenate((np.array([blueArray[h+t][t] for h in range(height-1, -1, -1) for t in range(0, min(min(height,width), (height-h)))]),
        np.array([blueArray[t][w+t] for w in range(1, width) for t in range(0, min(min(height,width), (width-w)))])))

    print("Diagonal : ")
    print(correlation(redDiagArray, np.roll(redDiagArray, -1)))
    print(correlation(greenDiagArray, np.roll(greenDiagArray, -1)))
    print(correlation(redDiagArray, np.roll(redDiagArray, -1)))

    return correlation


def correlationGraphs(imageArray:np.ndarray, cipherArray:numpy.ndarray, imageName='', saveFolder='./'):
    height = len(imageArray)
    width = len(imageArray[0])
    origional_image_Red_2D = imageArray[:, :, 0]
    origional_image_green_2D = imageArray[:, :, 1]
    origional_image_blue_2D = imageArray[:, :, 2]
    length = height * width
    origional_image_Red_1D = origional_image_Red_2D.ravel()
    origional_image_green_1D = origional_image_green_2D.ravel()
    origional_image_blue_1D = origional_image_blue_2D.ravel()

    cipher_image_Red_2D = cipherArray[:, :, 0]
    cipher_image_green_2D = cipherArray[:, :, 1]
    cipher_image_blue_2D = cipherArray[:, :, 2]
    length = height * width
    cipher_image_Red_1D = cipher_image_Red_2D.ravel()
    cipher_image_green_1D = cipher_image_green_2D.ravel()
    cipher_image_blue_1D = cipher_image_blue_2D.ravel()

    filename = imageName + '_origional_red.png'
    Location = saveFolder + filename
    plt.scatter(origional_image_Red_1D[0:(height * width) - 1], origional_image_Red_1D[1:(height * width)], s=0.2, color="red")
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(Location, orientation='portrait', dpi=300)
    plt.close()

    filename = imageName + '_origional_green.png'
    Location = saveFolder + filename
    plt.scatter(origional_image_green_1D[0:(height * width) - 1], origional_image_green_1D[1:(height * width)], s=0.2, color='green')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(Location, orientation='portrait', dpi=300)
    plt.close()

    filename = imageName + '_origional_blue.png'
    Location = saveFolder + filename
    plt.scatter(origional_image_blue_1D[0:(height * width) - 1], origional_image_blue_1D[1:(height * width)], s=0.2, color='blue')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(Location, orientation='portrait', dpi=300)
    plt.close()

    filename = imageName + '_cipher_red.png'
    Location = saveFolder + filename
    plt.scatter(cipher_image_Red_1D[0:(height * width) - 1], cipher_image_Red_1D[1:(height * width)], s=0.2, color='red')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(Location, orientation='portrait', dpi=300)
    plt.close()

    filename = imageName + '_cipher_green.png'
    Location = saveFolder + filename
    plt.scatter(cipher_image_green_1D[0:(height * width) - 1], cipher_image_green_1D[1:(height * width)], s=0.2, color='green')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(Location, orientation='portrait', dpi=300)
    plt.close()

    filename = imageName + '_cipher_blue.png'
    Location = saveFolder + filename
    plt.scatter(cipher_image_blue_1D[0:(height * width) - 1], cipher_image_blue_1D[1:(height * width)], s=0.2, color='blue')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(Location, orientation='portrait', dpi=300)
    plt.close()


def psnr(image, noiseDecryptedImage):
    array = np.asarray(image)
    noiseDecryptedArray = np.asarray(noiseDecryptedImage)
    MSE = np.square(array - noiseDecryptedArray).mean()
    tmp = (255.0 * 255.0) / (np.sqrt(MSE))
    print(MSE)
    return 10.0 * np.log10(tmp)


def entropy(image):
    imageArray = np.asarray(image)
    count = np.zeros(256)
    img1 = imageArray.ravel()
    size = len(img1)
    for x in img1:
        count[int(x)] += 1
    h = 0.0
    for m in count:
        if m > 0:
            p = m / size
            h += p * np.log2(1 / p)
    print("entropy = ", h)
    return h


if __name__ == '__main__':
    folderName = './images/'
    histogramFolder = './images/histograms/'
    imageList = ['baboon.png', 'lena.png', 'beach.png']
    HISTOGRAMS = False
    NPCR = False
    UACI = False
    CORRELATION = True
    CORRELATION_GRAPHS = False
    PSNR = False
    ENTROPY = False

    for imageName in imageList:
        image = Image.open(folderName + imageName)
        imageName = imageName.split('.p')[0]
        cipherImage = Image.open(folderName + imageName + '_cipher.png')
        decryptedImage = Image.open(folderName + imageName + '_decrypt.png')

        print(imageName + ": \n")
        if HISTOGRAMS:
            histogramRGB(image, imageName, histogramFolder)
            histogramRGB(cipherImage, imageName + '_cipher', histogramFolder)

        if NPCR:
            cipherImage2 = Image.open(folderName + imageName + '_one_bit_change_cipher.png')
            npcr = npcrImage(image, cipherImage2)
            print("  npcr (rgb, r, g, b) : ")
            print(npcr)

        if UACI:
            cipherImage2 = Image.open(folderName + imageName + '_one_bit_change_cipher.png')
            uaci = ucaiImage(image, cipherImage2)
            print("   uaci (rgb, r, g, b) : ")
            print(uaci)

        if CORRELATION:
            print("original: ", imageName)
            correlationImage(image)
            print("cipher: ", imageName)
            correlationImage(cipherImage)

        if CORRELATION_GRAPHS:
            correlationGraphs(np.asarray(image), np.asarray(cipherImage), imageName, "./images/")

        if PSNR:
            noiseDecryptedImage = Image.open(folderName + imageName + '_noise_decrypt.png')
            psnrValue = psnr(image, noiseDecryptedImage)
            print("  psnr:   ")
            print(psnrValue)

        if ENTROPY:
            print("image: ", end='')
            entropy(image)
            print("\ncipher: ", end='')
            entropy(cipherImage)
            print("")
        print("-----------------------------")