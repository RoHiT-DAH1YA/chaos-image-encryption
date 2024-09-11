from PIL import Image
import chaotic_map as cm
import time
import numpy as np


def permutation(imageObject:Image,  key=None, controlParameter=2500, widthcap=1000, heightcap=1000):
    if key is None:
        key = cm.bifurcation(controlParameter, 0.657, 5184, 1)[1][0]
    nextkey = cm.bifurcation(controlParameter, key, 523, 1)[1][0]
    width, height = imageObject.size
    blockwidth, blockheight = width, height
    xBlocks = 1
    yblocks = 1
    while blockwidth > widthcap:
        xBlocks += 1
        blockwidth = width // xBlocks
    while blockheight > heightcap:
        yblocks += 1
        blockheight = height // yblocks

    newImage = Image.new(imageObject.mode, imageObject.size)
    for xnum in range(xBlocks):
        for ynum in range(yblocks):
            xstart = xnum * blockwidth
            ystart = ynum * blockheight
            xend = xstart + blockwidth - 1
            yend = ystart + blockheight - 1
            if xnum == xBlocks - 1:
                xend = width-1
            if ynum == yblocks - 1:
                yend = height-1

            randomNumbers = cm.bifurcation(controlParameter, nextkey, 0, (xend - xstart + 1) * (yend - ystart + 1))[1]
            nextkey = randomNumbers[-1]
            temp = np.array([[randomNumbers[i], i] for i in range(len(randomNumbers))])

            randomNumbers = np.argsort(temp[:, 0], kind='quicksort')
            bw = xend - xstart + 1
            bh = yend - ystart + 1
            for newIndex in range(len(randomNumbers)):
                originalIndex = randomNumbers[newIndex]
                originalPixelPosition = (xstart + originalIndex % bw, ystart + originalIndex//bw)
                newPixelPosition = (xstart + newIndex % bw, ystart + newIndex//bw)
                newImage.putpixel(newPixelPosition, imageObject.getpixel(originalPixelPosition))

    return newImage, key


def reversePermutation(imageObject:Image, permutationKey, controlParameter=2500, widthcap=1000,heightcap=1000):
    nextkey = cm.bifurcation(controlParameter, permutationKey, 523, 1)[1][0]
    width, height = imageObject.size
    blockwidth, blockheight = width, height
    xBlocks = 1
    yblocks = 1
    while blockwidth > widthcap:
        xBlocks += 1
        blockwidth = width // xBlocks
    while blockheight > heightcap:
        yblocks += 1
        blockheight = height // yblocks

    decyptedImage = Image.new(imageObject.mode, imageObject.size)
    for xnum in range(xBlocks):
        for ynum in range(yblocks):
            xstart = xnum * blockwidth
            ystart = ynum * blockheight
            xend = xstart + blockwidth - 1
            yend = ystart + blockheight - 1
            if xnum == xBlocks - 1:
                xend = width - 1
            if ynum == yblocks - 1:
                yend = height - 1

            randomNumbers = cm.bifurcation(controlParameter, nextkey, 0, (xend - xstart + 1) * (yend - ystart + 1))[1]
            nextkey = randomNumbers[-1]
            temp = np.array([[randomNumbers[i], i] for i in range(len(randomNumbers))])

            randomNumbers = np.argsort(temp[:, 0], kind='quicksort')
            bw = xend - xstart + 1
            bh = yend - ystart + 1
            for index in range(len(randomNumbers)):
                decyptedIndex = randomNumbers[index]
                encryptedPixelPosition = (xstart + index % bw, ystart + index // bw)
                decyptedPixelPosition = (xstart + decyptedIndex % bw, ystart + decyptedIndex // bw)
                decyptedImage.putpixel(decyptedPixelPosition, imageObject.getpixel(encryptedPixelPosition))

    return decyptedImage

if __name__ == "__main__":
    orgImage = Image.open("pool_balls.png")
    start = time.time()
    newImage, key = permutation(orgImage)
    end = time.time()
    newImage.save("./diffusedImage.png")
    print("key: ", key)
    print("completed")
    print("time: ", start, end, start - end)

    decryptedImage = reversePermutation(newImage, key)
    decryptedImage.save("./dediffusedImage.png")