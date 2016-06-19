import pandas as pd
from skimage import io
import os
import re

rootdir0="/home/mina/data_science_game/roof_images/"

count=0
dic={}
data=[]
#print(os.path.dirname(os.path.abspath(__file__)))

for root, dirnames, filenames in os.walk( rootdir0 ):
    for filename in filenames:
	count=count+1

	myfile=open(os.path.join(root,filename))
	img=io.imread(myfile)

	filename=re.sub(r'.jpg','',filename)

	dic={'id':filename,'data':img}
	data.append(dic)
	if count%100==0:
		print(dic)
print(count)
ordered=sorted(data)
img_data = pd.DataFrame(ordered)
img_data.to_csv("img_data.csv", sep=';', index=False)
