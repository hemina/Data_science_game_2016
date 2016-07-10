import sys
import os
import numpy
import time

h = 64
w = 64
pathLoadImageInputForder = "..\FilesBeforeComputation"
pathModelOutputFolder = "..\Model"

def main():
    t0 = time.clock()
    print "NorthSouth Computation"
    matrixMedian = numpy.zeros((h, w))
    matrixStd = numpy.zeros((h, w))
    totalWeight = 0
    for j in range(0, h):
        print "..." + str(j * 0.39) + "%..."
        for k in range(0, w):
            with open(pathLoadImageInputForder + "\N_"+str(j)+"_"+str(k)+".txt", 'r') as f:
                data = f.readlines()
                for line in data:
                    numbers = map(float, line.split())
            matrixMedian[j][k] = numpy.median(numbers)
            matrixStd[j][k] = numpy.std(numbers)
            totalWeight = totalWeight + matrixStd[j][k]
    matrixStd = matrixStd / totalWeight
    numpy.savetxt(pathModelOutputFolder + "/northsouthModel.txt", matrixMedian)
    numpy.savetxt(pathModelOutputFolder + "/northsouthModelPonderation.txt", matrixStd)

    print "Flat Computation"
    matrixMedian = numpy.zeros((h, w))
    matrixStd = numpy.zeros((h, w))
    totalWeight = 0
    for j in range(0, h):
        print "..." + str(25 + j * 0.39) + "%..."
        for k in range(0, w):
            with open(pathLoadImageInputForder + "\F_" + str(j) + "_" + str(k) + ".txt", 'r') as f:
                data = f.readlines()
                for line in data:
                    numbers = map(float, line.split())
            matrixMedian[j][k] = numpy.median(numbers)
            matrixStd[j][k] = numpy.std(numbers)
            totalWeight = totalWeight + matrixStd[j][k]
    matrixStd = matrixStd / totalWeight
    numpy.savetxt(pathModelOutputFolder + "/flatModel.txt", matrixMedian)
    numpy.savetxt(pathModelOutputFolder + "/flatModelPonderation.txt", matrixStd)

    print "EastWest Computation"
    matrixMedian = numpy.zeros((h, w))
    matrixStd = numpy.zeros((h, w))
    totalWeight = 0
    for j in range(0, h):
        print "..." + str(50 + j * 0.39) + "%..."
        for k in range(0, w):
            with open(pathLoadImageInputForder + "\E_" + str(j) + "_" + str(k) + ".txt", 'r') as f:
                data = f.readlines()
                for line in data:
                    numbers = map(float, line.split())
            matrixMedian[j][k] = numpy.median(numbers)
            matrixStd[j][k] = numpy.std(numbers)
            totalWeight = totalWeight + matrixStd[j][k]
    matrixStd = matrixStd / totalWeight
    numpy.savetxt(pathModelOutputFolder + "/eastwestModel.txt", matrixMedian)
    numpy.savetxt(pathModelOutputFolder + "/eastwestModelPonderation.txt", matrixStd)

    print "Other Computation"
    matrixMedian = numpy.zeros((h, w))
    matrixStd = numpy.zeros((h, w))
    totalWeight = 0
    for j in range(0, h):
        print "..." + str(75 + j * 0.39) + "%..."
        for k in range(0, w):
            with open(pathLoadImageInputForder + "\O_" + str(j) + "_" + str(k) + ".txt", 'r') as f:
                data = f.readlines()
                for line in data:
                    numbers = map(float, line.split())
            matrixMedian[j][k] = numpy.median(numbers)
            matrixStd[j][k] = numpy.std(numbers)
            totalWeight = totalWeight + matrixStd[j][k]
    matrixStd = matrixStd / totalWeight
    numpy.savetxt(pathModelOutputFolder + "/otherModel.txt", matrixMedian)
    numpy.savetxt(pathModelOutputFolder + "/otherModelPonderation.txt", matrixStd)

    print("Processing time =" + str(time.clock() - t0))

main()
