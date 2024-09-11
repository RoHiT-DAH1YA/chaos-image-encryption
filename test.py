from PIL import Image
import chaotic_map as cm
import dnaEncoding as de
import numpy as np
import permutation as per
import time
import diffusion
import random


def encrypt(orgImage:Image, key1, key2, controlParameter1, controlParameter2, diffusionKey=None):
    start = time.time()
    width, height = orgImage.size
    if diffusionKey is None:
        diffusionKey = key1

    newImage = Image.fromarray(diffusion.diffusionXorImage(orgImage))
    newImage, diffusionKey = per.permutation(newImage,key=diffusionKey,controlParameter=controlParameter1)
    newImageDnaString = de.getDnaString(newImage)
    print("new dna string len ", len(newImageDnaString))
    print("hello")
    dnakey1 = cm.generateDnaString(len(newImageDnaString), key=key1, controlParameter=controlParameter1)
    newImageDnaString = de.dnaXor(newImageDnaString, dnakey1)
    print("hello")
    dnakey2 = cm.generateDnaString(len(newImageDnaString),key=key2, controlParameter=controlParameter2)
    print("hello")
    newImageDnaString = de.dnaAddition(newImageDnaString, dnakey2)
    print("hello")
    newImage = de.getImageFromDnaString(newImageDnaString, width, height)
    print("hello")
    print("encryption Time: ", time.time() - start)
    return newImage, diffusionKey

def decrypt(encryptedImage:Image, key1, key2, controlParameter1, controlParameter2, permutationKey=None):
    start = time.time()
    if permutationKey is None:
        permutationKey = key1
    width, height = encryptedImage.size
    encryptedImageDnaString = de.getDnaString(encryptedImage)

    dnakey2 = cm.generateDnaString(len(encryptedImageDnaString), key=key2, controlParameter=controlParameter2)
    encryptedImageDnaString = de.dnaSubtraction(encryptedImageDnaString, dnakey2)

    dnakey1 = cm.generateDnaString(len(encryptedImageDnaString), key=key1, controlParameter=controlParameter1)
    encryptedImageDnaString = de.dnaXor(encryptedImageDnaString, dnakey1)

    decryptedImage = de.getImageFromDnaString(encryptedImageDnaString, width=width, height=height)
    decryptedImage = per.reversePermutation(decryptedImage, permutationKey=permutationKey,  controlParameter=controlParameter1)
    decryptedImage = Image.fromarray(diffusion.reverseDiffusionXorArray(np.asarray(decryptedImage)))
    print("decryption Time: ", time.time() - start)
    return decryptedImage


def noise(imageArray:np.ndarray, probability):
    for h in range(imageArray.shape[0]):
        for w in range(imageArray.shape[1]):
            for i in range(imageArray.shape[2]):
                ran = random.random()
                threshold = 1 - probability
                if ran < probability:
                    imageArray[h][w][i] = 0
                elif ran > threshold:
                    imageArray[h][w][i] = 255
    return imageArray

def getRGBcomponents(image: Image):
    imageArray = np.asarray(image)

    # Create separate RGB component images
    redImageArray = np.zeros(imageArray.shape)
    greenImageArray = np.zeros(imageArray.shape)
    blueImageArray = np.zeros(imageArray.shape)

    # Set respective color channel for each component image
    redImageArray[:, :, 0] = imageArray[:, :, 0]  # Red channel
    greenImageArray[:, :, 1] = imageArray[:, :, 1]  # Green channel
    blueImageArray[:, :, 2] = imageArray[:, :, 2]  # Blue channel

    # Convert back to PIL Image format
    redImage = Image.fromarray(redImageArray.astype('uint8'))
    greenImage = Image.fromarray(greenImageArray.astype('uint8'))
    blueImage = Image.fromarray(blueImageArray.astype('uint8'))

    return redImage, greenImage, blueImage


if __name__ == '__main__':
    controlParameter1 = 2814.2345
    key1 = 0.2345
    controlParameter2 = 2168.5439
    key2 = 0.9834
    diffusionKey = 0.5293
    folderPath = "./images/"
    imageList = ["lena.png", "baboon.png", "beach.png"]
    NORMAL_ENCRYPTION = True
    ONE_BIT_CHANGE_ENCRYPTION = True
    RGB_COMPONENTS = True
    NOISE_DECRYPT = True

    for imageName in imageList:
        imageName = folderPath + imageName
        originalImage = Image.open(imageName)
        print(originalImage.size)
        if NORMAL_ENCRYPTION:
            encryptedImage, diffusionKey = encrypt(originalImage, key1, key2, controlParameter1, controlParameter2, diffusionKey)
            decryptedImage = decrypt(encryptedImage, key1, key2, controlParameter1, controlParameter2, diffusionKey)
            encryptedImage.save(imageName.split('.p')[0] + '_cipher.png')
            decryptedImage.save(imageName.split('.p')[0] + '_decrypt.png')
            if NOISE_DECRYPT:
                encryptedImageArray = np.asarray(encryptedImage)
                noiseEncryptedImage = Image.fromarray(noise(encryptedImageArray.copy(), 0.01))
                noiseDecryptedImage = decrypt(noiseEncryptedImage, key1, key2, controlParameter1, controlParameter2, diffusionKey)
                # noiseDecryptedImage = Image.fromarray(array)
                noiseDecryptedImage.save(imageName.split('.p')[0] + '_noise_decrypt.png')

        if ONE_BIT_CHANGE_ENCRYPTION:
            oneBitChangeImage = originalImage.copy()
            pixel = originalImage.getpixel((0, 0))
            print(pixel)
            pixel = tuple([i ^ 1 for i in pixel])
            print(pixel)
            oneBitChangeImage.putpixel((0, 0), pixel)

            encryptedImage, diffusionKey = encrypt(oneBitChangeImage, key1, key2, controlParameter1, controlParameter2,
                                                   diffusionKey)
            encryptedImage.save(imageName.split('.p')[0] + '_one_bit_change_cipher.png')

        if RGB_COMPONENTS:
            redImage, greenImage, blueImage = getRGBcomponents(originalImage)
            redImage.save(imageName.split('.p')[0] + '_red_component.png')
            greenImage.save(imageName.split('.p')[0] + '_green_component.png')
            blueImage.save(imageName.split('.p')[0] + '_blue_component.png')

        if not NORMAL_ENCRYPTION and NOISE_DECRYPT:
            encryptedImageArray = np.asarray(Image.open(imageName.split('.p')[0] + '_cipher.png'))
            noiseEncryptedImage = Image.fromarray(noise(encryptedImageArray.copy(), 0.01))
            noiseDecryptedImage = decrypt(noiseEncryptedImage, key1, key2, controlParameter1, controlParameter2, diffusionKey)
            noiseDecryptedImage.save(imageName.split('.p')[0] + '_noise_decrypt.png')
