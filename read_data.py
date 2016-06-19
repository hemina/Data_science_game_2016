import pandas as pd
from skimage import io
import numpy as np
import os
import re

rootdir0="/home/mina/data_science_game/roof_images/"

count=0
dic={}
#print(os.path.dirname(os.path.abspath(__file__)))

for root, dirnames, filenames in os.walk( rootdir0 ):
    for filename in filenames:
	count=count+1

	# By using skimage.io.imread, we load the image into numpy.ndarray format.
	myfile=open(os.path.join(root,filename))
	img=io.imread(myfile)
#	print(img.shape) # From the shape we can see that each image is a RGB color image, however the size is different from one another. 

	# We keep only the id for each picture, so that we remove '.jpg' in the filename.
	filename=re.sub(r'.jpg','',filename)
	
	dic[filename]=img
#	data.append(dic)
	if count%5000==0:
		print(len(dic))
print(count)
ordered=sorted(dic.items())
img_data = pd.DataFrame(ordered)
img_data.columns = ['Id', 'data']
print(img_data)
#img_data.to_csv("img_data.csv", sep=';', index=False)

label=pd.read_csv("/home/mina/data_science_game/id_train.csv")
print(label)

train=pd.merge(label, img_data, on="Id")
print(train)

