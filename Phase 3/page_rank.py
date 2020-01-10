import sys, os, webbrowser
import numpy
import json
import time
start=time.time()

def visualizeImages(t2):
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'config.ini'))
    path = config.get('visualization', 'image_path')
    htmlFile = "page_rank.html"

    if os.path.exists(htmlFile):
        print("removing old html file...")
        os.remove(htmlFile)

    f = open(htmlFile, "a")
    header = open("htmlHeader.txt","r")
    f.write(header.read())

    print("generating html file...")

    f.write("<div align=\"center\"> <h1><u> Most relevant Images using PageRank Implementation. </u></h1></div>\n")
    f.write("<div class=\"row\">\n")

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


def Pagerank_image(matrix,k):
    page={}
    p_rank={}
    old_val={}
    counter=0
    count1=0
    converge=0
    d = 0.85
    number_edges = len(matrix)
    b = next(iter(matrix.values()))
    for r in b:
        counter+=1
    number_out = counter
    for i in matrix:
        page[i] = 1/number_edges
        old_val[i] = page[i]
    while converge == 0:
        count1+=1
        for j in page:
            sum=0
            c=[x for x,v in matrix.items() if j in v]
            for i in c:
                if i != j:
                    sum = sum + page[i]
            page[j] = (1-d)/number_edges + d * (sum)/(number_out)
            if numpy.fabs(page[j] - old_val[j])<=0.00001:
                p_rank[j] = page[j]
                if len(p_rank) == len(page):
                    converge=1
                    print(numpy.sum([x for x in p_rank.values()]))
                    break
            else:
                old_val[j] = page[j]
    print("The convergence occured after ",count1,"iterations : ")
    #top=sorted(p_rank.items(),key=lambda x:-x[1],reverse=False)[:k]
    t2 = {}
    count = 0
    for image in sorted(p_rank, key=p_rank.get, reverse=True):
        t2[image] = p_rank[image]
        count += 1
        if count == k:
            break
    visualizeImages(t2)

if __name__ == '__main__':
    filename = 'graph.json'
    with open(filename) as load_f:
        matrix = json.load(load_f)
    print ("Hi! How you doing,\nPlease enter the value of K: ")
    k = int(input())
    print ("Applying the PageRank algorithm to each of the image nodes ....: ")
    b = Pagerank_image(matrix,k)
    print(b)
    end=time.time()
    print(end-start)
    sys.exit(0)
