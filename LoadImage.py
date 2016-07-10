from PIL import Image
import numpy
import csv
import time
import os.path

length = 64
width = 64
size = length,width
pathTrainFile = '../id_train.csv'
pathOutputFolder = "../FilesBeforeComputation"
pathInputImagesForder = "../Image"

def readImage(imageNum):
        im = Image.open(pathInputImagesForder + "/"+ imageNum + ".jpg")
        im = im.resize(size)
        pixel_array = numpy.array(im.convert('L'), 'f')
        pixel_array_norm = normalisation(1,pixel_array)
        #print pixel_array_norm
        return pixel_array_norm

def normalisation(type,pixel_array):
    if type == 1:
        max = numpy.max(pixel_array)
        min = numpy.min(pixel_array)
        for i in range(0,len(pixel_array)):
            for j in range(0,len(pixel_array[1])):
                pixel_array[i][j] = round(((pixel_array[i][j] - min)/(max - min)), 3)
        return pixel_array

def readTrainFile():
    t0 = time.clock()
    k = 0
    trainFile = open(pathTrainFile,'rb')
    fichiercsvReader = csv.reader(trainFile, delimiter=',')
    next(fichiercsvReader, None)
    lines= [l for l in fichiercsvReader]
    for line in lines:
        k = k + 1
        print "Processing Image " + str(k) + " over " + str(len(lines))
        print "..." + str(k*100/len(lines)) + "%..."
        num_image = line[0]
        category =line[1]
        if os.path.exists(pathInputImagesForder + "/"+ num_image + ".jpg"):
            pixel_array = readImage(num_image)
            for i in range(0, 64):
                for j in range(0, 64):
                    if int(category) == 1:
                        with open(pathOutputFolder + "/N_"+ str(i) + "_" + str(j) + ".txt", "a") as myfile:
                            myfile.write(" "+ str(pixel_array[i][j]))
                    elif int(category) == 2:
                       with open(pathOutputFolder + "/E_"+ str(i) + "_" + str(j) + ".txt", "a") as myfile:
                            myfile.write(" "+ str(pixel_array[i][j]))
                    elif int(category) == 3:
                        with open(pathOutputFolder + "/F_"+ str(i) + "_" + str(j) + ".txt", "a") as myfile:
                            myfile.write(" "+ str(pixel_array[i][j]))
                    elif int(category) == 4:
                        with open(pathOutputFolder + "/O_"+ str(i) + "_" + str(j) + ".txt", "a") as myfile:
                            myfile.write(" "+ str(pixel_array[i][j]))
    print("Processing time =" + str(time.clock() - t0))

readTrainFile()




