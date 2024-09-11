from PIL import Image
import numpy as np
import time


def diffusionXorImage(image:Image):
    imageArray = np.asarray(image)
    print(image.size)
    return  diffusionXorArray(imageArray, image.size[0], image.size[1])

def diffusionXorArray(imageArray:np.ndarray, width, height):
    newImageArray = np.copy(imageArray)

    for w in range(width - 2):
        for h in range(height - 2):
            newImageArray[h + 2][w + 1] = newImageArray[h + 2][w + 1] ^ newImageArray[h][w]
            newImageArray[h + 1][w + 2] = newImageArray[h + 1][w + 2] ^ newImageArray[h][w]

    for a in range(width - 2):
        for b in range(height - 2):
            w = width - a - 1
            h = height - b - 1
            newImageArray[h - 2][w - 1] = newImageArray[h - 2][w - 1] ^ newImageArray[h][w]
            newImageArray[h - 1][w - 2] = newImageArray[h - 1][w - 2] ^ newImageArray[h][w]


    return newImageArray

def reverseDiffusionXorArray(imageArray:np.ndarray):
    width = len(imageArray[0])
    height = len(imageArray)
    newImageArray = np.copy(imageArray)

    for w in range(2, width):
        for h in range(2, height):
            newImageArray[h - 1][w - 2] = newImageArray[h - 1][w - 2] ^ newImageArray[h][w]
            newImageArray[h - 2][w - 1] = newImageArray[h - 2][w - 1] ^ newImageArray[h][w]

    for w in range(width-2-1,-1,-1):
        for h in range(height-2-1, -1, -1):
            newImageArray[h + 2][w + 1] = newImageArray[h + 2][w + 1] ^ newImageArray[h][w]
            newImageArray[h + 1][w + 2] = newImageArray[h + 1][w + 2] ^ newImageArray[h][w]

    return newImageArray


if __name__ == '__main__':
    start = time.time()
    orgImage = Image.open("wind.png")
    newImageArray = diffusionXorImage(orgImage)
    print("xor completed: ", time.time() - start)
    print(newImageArray.dtype)
    newImage = Image.fromarray(newImageArray)
    newImage.save("./diffusionImage.png")
    print("enc image save completed: ", time.time() - start)

    reverseArray = reverseDiffusionXorArray(newImageArray)
    reverseImage = Image.fromarray(reverseArray)
    reverseImage.save("./diffusionImageReverse.png")

    print("reverse image completed: ", time.time() - start)
