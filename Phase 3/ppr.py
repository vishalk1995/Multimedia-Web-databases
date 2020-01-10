import pandas as pd
import os
import webbrowser
import numpy as np
from sklearn.preprocessing import normalize

'''get the input from the user via command line'''

k= int(input('Enter the K values:'))
image1= input('Enter the image1:')
image2= input('Enter the image2:')
image3= input('Enter the image3:')
'''
image1='103569702'
image2='4765206950'
image3='5187089159'
'''

graph=pd.read_json("graph.json", convert_axes=False)
matrix = pd.DataFrame(index=graph.columns,
                                    columns=graph.columns)
matrix=matrix.fillna(0)
matrix=matrix.add(graph)
matrix=matrix.fillna(0)
#print(matrix)


'''get the index of the images'''

image1_index=(matrix.columns.get_loc(str(image1)))
image2_index=(matrix.columns.get_loc(str(image2)))
image3_index=(matrix.columns.get_loc(str(image3)))

all_imageids=list(matrix.columns.values)
matrix=normalize(matrix, axis=0, norm='l1')
Ai = np.array(matrix[:,1])
#print(matrix)
#print(sum(Ai))


'''create a teleportation vector'''
n=matrix.shape[0]
Ei = np.zeros(n,dtype=np.float64)
Ei[image1_index]=0.3333333333
Ei[image2_index]=0.3333333333
Ei[image3_index]=0.3333333333
'''
Ei = np.ones(n,dtype=np.float64)
Ei=Ei/n
print(Ei)
'''

#Below code can be used to retrieve induvidual similarity score of image-image pair
maxerr=0.01
s=0.85
sink=0
ro, r = np.zeros(n), np.ones(n)
iterations=0
#while np.sum(np.abs(r-ro)) > maxerr:
while iterations<20:
    ro = r.copy()
    iterations=iterations+1
    for i in range(0,n):
        Ai = np.array(matrix[:,i])
        #print("sum Ai= "+str(sum(Ai)))
        #Di = sink / float(n)
        r[i] = ro.dot( Ai*s + Ei*(1-s) )

#print("sum= "+str(sum(r)))
result=r/float(sum(r))
result_img=pd.DataFrame(result, index=all_imageids)

print("number of iterations="+str(iterations))
zipbObj = zip(all_imageids, result_img[0])
dictOfWords = dict(zipbObj)
# t2=sorted(dictOfWords.iteritems(),key=lambda x:-x[1],reverse=False)[:k]
# t2=sorted(dictOfWords.iteritems(),key=lambda x:-x[1],reverse=False)[:k]
# t2=ast.literal_eval(json.dumps(t2))

t2 = {}
count = 0
for image in sorted(dictOfWords, key=dictOfWords.get, reverse=True):
    t2[image] = dictOfWords[image]
    count += 1
    if count == k:
        break

def visualizeImages(t2):
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'config.ini'))
    path = config.get('visualization', 'image_path')
    htmlFile = "ppr.html"

    if os.path.exists(htmlFile):
        print("removing old html file...")
        os.remove(htmlFile)

    f = open(htmlFile, "a")
    header = open("htmlHeader.txt","r")
    f.write(header.read())

    print("generating html file...")

    f.write("<div align=\"center\"> <h1><u>PPR - PageRank Implementation. </u></h1></div>\n")
    f.write("<div class=\"row\">\n")

    f.write("<div align=\"center\"> <h2><u> Input Images </u></h2></div>\n")
    f.write("</div>\n\n<div class=\"row\">\n")
    for imageId in [image1, image2, image3]:
        f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + str(imageId) + ".jpg\" alt=\"Image Id - " + str(imageId) + "\"/><figcaption> Image Id - " + str(imageId) + "</figcaption>\n\t\t</figure> \n\t</div>\n")
    f.write("</div>\n\n<div class=\"row\">\n")
    f.write("<div align=\"center\"> <h2><u> Most relevant Images </u></h2></div>\n")

    i = 0
    for imageId in sorted(t2, key=t2.get, reverse=True):
        if i % 4 == 0:
            f.write("</div>\n\n<div class=\"row\"> \n")
            i = 0
        score = str(t2[imageId])
        # f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + "images\\" + str(dict_top8[imageId][0]) + ".jpg\" alt=\"Image Id - " + str(dict_top8[imageId][0]) + "&emsp;Score - " + labelValue + "\"/><figcaption> Image Id - " + str(dict_top8[imageId][0]) + "&emsp;Score - " + str(dict_top8[imageId][1]) + "</figcaption>\n\t\t</figure> \n\t</div>\n")
        f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + str(imageId) + ".jpg\" alt=\"Image Id - " + str(imageId) + "&emsp;Score - " + score + "\"/><figcaption> Image Id - " + str(imageId) + "&emsp;Score - " + str(score) + "</figcaption>\n\t\t</figure> \n\t</div>\n")
        i = i+1

    f.write("</div>\n</body>")
    webbrowser.open('file://' + os.path.realpath(htmlFile))

visualizeImages(t2)
