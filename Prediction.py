from PIL import Image
import numpy
import csv
import time
import os.path

h = 64
w = 64
size = h,w
pathTrainFile = '../sample_submission4.csv'
pathModelInputFolder = '..\Model'
pathInputImagesForder = "../Image"
pathOutputResultFolder = "../Result"

def main():
    t0 = time.clock()

    matrixMedianNorthSouth = numpy.loadtxt(pathModelInputFolder + '/northsouthModel.txt', float)
    matrixStdNorthSouth = numpy.loadtxt(pathModelInputFolder + '/northsouthModelPonderation.txt', float)
    matrixMedianEastWest = numpy.loadtxt(pathModelInputFolder + '/eastwestModel.txt', float)
    matrixStdEastWest = numpy.loadtxt(pathModelInputFolder + '/eastwestModelPonderation.txt', float)
    matrixMedianFlat = numpy.loadtxt(pathModelInputFolder + '/flatModel.txt', float)
    matrixStdFlat = numpy.loadtxt(pathModelInputFolder + '/flatModelPonderation.txt', float)
    matrixMedianOther = numpy.loadtxt(pathModelInputFolder + '/otherModel.txt', float)
    matrixStdOther = numpy.loadtxt(pathModelInputFolder + '/otherModelPonderation.txt', float)

    with open(pathOutputResultFolder + "/PredictiveResult.csv", "w") as myfile:
        myfile.write("Id,label")

    trainFile = open(pathTrainFile,'rb')
    fichiercsvReader = csv.reader(trainFile, delimiter=',')
    next(fichiercsvReader, None)
    lines= [l for l in fichiercsvReader]
    k = 0
    for line in lines:
        imageName = line[0]
        category = line[1]
        k = k + 1
        print "Processing Image " + str(k) + " over " + str(len(lines))
        print "..." + str(k * 100 / len(lines)) + "%..."

        matrixI = readImage(imageName)
        countN = sumFinal(matrixI,matrixMedianNorthSouth,matrixStdNorthSouth)
        countE = sumFinal(matrixI,matrixMedianEastWest,matrixStdEastWest)
        countF = sumFinal(matrixI,matrixMedianFlat,matrixStdFlat)
        countO = sumFinal(matrixI,matrixMedianOther,matrixStdOther)
        result = 0
        if min(countN, countE, countF) == countN and countN < 0.22:
            result = 1
        elif min(countN, countE, countF) == countE and countE < 0.22:
            result = 2
        elif min(countN, countE, countF) == countF and countF < 0.22:
            result = 3
        else:
            #if min(countN, countE, countF, countO) == countO:
            result = 4

        with open(pathOutputResultFolder + "/PredictiveResult.csv", "a") as myfile:
            myfile.write("\n" + imageName+ "," + str(result))

    print("Processing time =" + str(time.clock() - t0))

def sumFinal(matrix1, matrix2, matrix3):
    matrix4 = numpy.multiply(numpy.abs(numpy.subtract(matrix1,matrix2)), matrix3)
    return numpy.sum(matrix4)

def readImage(imageNum):
    im = Image.open(pathInputImagesForder + "/" + imageNum + ".jpg")
    im = im.resize(size)
    pixel_array = numpy.array(im.convert('L'), 'f')
    pixel_array_norm = normalisation(1, pixel_array)
    # print pixel_array_norm
    return pixel_array_norm

def normalisation(type,pixel_array):
    if type == 1:
        max = numpy.max(pixel_array)
        min = numpy.min(pixel_array)
        for i in range(0,len(pixel_array)):
            for j in range(0,len(pixel_array[1])):
                pixel_array[i][j] = round(((pixel_array[i][j] - min)/(max - min)), 3)
        return pixel_array


main()
