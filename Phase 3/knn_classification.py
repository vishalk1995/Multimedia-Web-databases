import os
import webbrowser
import time
import pandas
import shutil
from collections import Counter

def visualizeImages(imageLabelMatrix, inputImageIds, inputLabelValues):
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'config.ini'))
    path = config.get('visualization', 'image_path')
    htmlFile = "KnnImageLabels.html"

    if os.path.exists(htmlFile):
        print("removing old html file...")
        os.remove(htmlFile)

    f = open(htmlFile, "a")
    header = open("htmlHeader.txt","r")
    f.write(header.read())

    print("generating html file...")

    f.write("<div align=\"center\"> <h1><u>Image labelling using k-nearest neighbor based classification algorithm. </u></h1></div>\n")
    f.write("\n<div align=\"center\"> <h2><u>Input Image - Label Pairs. </u></h1></div>\n")
    f.write("<div class=\"row\">\n")

    i = 0
    j=0
    for imageId in inputImageIds:
        if i % 4 == 0:
            f.write("</div>\n\n<div class=\"row\"> \n")
            i = 0
        labelValue = str(inputLabelValues[j])
        j = j + 1
        f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + str(imageId) + ".jpg\" alt=\"Image Id - " + str(imageId) + "&emsp;Label - " + labelValue + "\"/><figcaption> Image Id - " + str(imageId) + "&emsp;Label - " + labelValue + "</figcaption>\n\t\t</figure> \n\t</div>\n")
        i = i+1

    f.write("</div>\n")
    f.write("<br>\n<div align=\"center\"> <h2><u>kNN output Image - Label Pairs. </u></h1></div>\n")
    f.write("<div class=\"row\">\n")

    i = 0
    for imageId in imageLabelMatrix.keys():
        if i % 4 == 0:
            f.write("</div>\n\n<div class=\"row\"> \n")
            i = 0
        labelValue = imageLabelMatrix[imageId]
        f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + str(imageId) + ".jpg\" alt=\"Image Id - " + str(imageId) + "&emsp;Label - " + labelValue + "\"/><figcaption> Image Id - " + str(imageId) + "&emsp;Label - " + labelValue + "</figcaption>\n\t\t</figure> \n\t</div>\n")
        i = i+1

    f.write("</div>\n</body>")

    #iexplore = os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"),
    #                     "Internet Explorer\\IEXPLORE.EXE")
    #browser = webbrowser.get(iexplore)
    webbrowser.open('file://' + os.path.realpath(htmlFile))


def main():
    k = int(input('Enter the k:'))

    print("============Started program================")
    start_time=time.time()

    print("reading similarity matrix...")
    matrix = pandas.read_csv("matrix.csv")
    matrix = pandas.DataFrame(matrix.values, index=matrix.columns, columns=matrix.columns)


    fields=['image', 'label']
    imageLabelPairs = pandas.read_csv("image_label_pairs.csv", skipinitialspace=True, engine='python')

    imagesFromFile = imageLabelPairs['image'].tolist()
    labelsFromFile = imageLabelPairs['label'].tolist()

    imageList = matrix.columns

    # Convert image and labels to dictionary
    #imageLabelPairsDict = dict(zip(imagesFromFile, labelsFromFile))
    #imageLabelPairsDict.pop('image')
    assignedImageLabels = {}

    #exit()

    print("fetching similarities and assigning labels...")
    for image in imageList:
        if image not in imagesFromFile:
            similarityScore = {}
            for image2 in imagesFromFile:
                similarityScore[image2] = matrix[image][str(image2)]

            # fetch top k from imageScores, identity labels and mark lable to this image
            kSimilarLables = []
            for i in range(k):
                max_key = max(similarityScore, key=similarityScore.get)
                kSimilarLables.append(labelsFromFile[imagesFromFile.index(max_key)])
                similarityScore.pop(max_key)

            #Assign the label which occurs maximum times in
            cnt = Counter(kSimilarLables)
            label = cnt.most_common(1)
            assignedImageLabels[image] = label[0][0]
            print("Image ID = " , image , "Assigned Label = " , label)

    # call visualize function
    visualizeImages(assignedImageLabels, imagesFromFile, labelsFromFile)
    print("==== End of program. Time Taken %s seconds ====" % (time.time() - start_time))

main()
