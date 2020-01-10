import os,webbrowser
import operator
import time
import pandas, numpy
from pymongo import MongoClient
from scipy import spatial
from configparser import ConfigParser

start_time = time.time()

#===========================================Visulization================================================================
#========================================================================================================================

def visualizeImages(input_image, imageLabelMatrix):
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'config.ini'))
    path= config.get('visualization', 'image_path')    #Path where all the images stored to visualize
    htmlFile = "lsh.html"
    if os.path.exists(htmlFile):
        os.remove(htmlFile)
    f = open(htmlFile, "w")
    header = open("htmlHeader.txt","r")
    f.write(header.read())
    f.write("<div align=\"center\"> <h1><u>Image Searching using Locality Sensitive Hashing. </u></h1></div>\n")
    f.write("<div class=\"row\"></div>\n\n")
    f.write("<div align=\"center\"> <h2><u> Input Image </u></h2></div>\n")
    f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + str(input_image) + ".jpg\" alt=\"Image Id - " + str(input_image) + "\"/><figcaption> Image Id - " + str(input_image) + "</figcaption>\n\t\t</figure> \n\t</div>\n")
    f.write("<div class=\"row\"></div>\n\n")
    f.write("<div align=\"center\"> <h2><u> Top matching Images </u></h2></div>\n")
    f.write("<div class=\"row\">\n")
    i = 0
    for imageId in sorted(imageLabelMatrix, key=imageLabelMatrix.get):
        if i%4 == 0:
            f.write("</div>\n\n<div class=\"row\">\n")
            i=0
        labelValue = imageLabelMatrix[imageId]
        f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + str(imageId) + ".jpg\" alt=\"Image Id - " + str(imageId) + "&emsp;Label - " + str(labelValue) + "\"/><figcaption> Image Id - " + str(imageId) + "&emsp;Label - " + str(labelValue) + "</figcaption>\n\t\t</figure> \n\t</div>\n")
        i=i+1
    f.write("</div>\n</body>")
    webbrowser.open(htmlFile)
    print("---- completed in", time.time() - start_time, "seconds ----")

#=========================================tNN Images====================================================================
#========================================================================================================================
def find_nn(image_vector, count, lsh_table, lsh_fuctions, input_image_id):
    hash_codes = []
    for l in range(len(lsh_fuctions)):
       hash_codes.append([])
       for k in range(len(lsh_fuctions[l])):
           # table[image].append(0 if row.T.dot(h[i][hash_index]) < 0 else 1)
           hash_codes[l].append(0 if image_vector.dot(lsh_fuctions[l][k]) < 0 else 1)
    search_vector_hash_code = {}
    list_hash_codes = []
    combined_matrix = []
    for l in range(len(lsh_table)):
        list_hash_codes = hash_codes[l]
        # print(hash_codes[l])
        search_vector_hash_code = ''.join(str(c) for c in list_hash_codes)
        one_table  = lsh_table[l]
        # print(search_vector_hash_code)
        if search_vector_hash_code in one_table.keys():
            combined_matrix.append(one_table[search_vector_hash_code])
        else:
            pass
    c = 0
    for i in combined_matrix:
        for j in i:
            c=c+1
    print("No of overall images reffered: ", c)
    l = []
    for sublist in combined_matrix:
        for item in sublist:
            l.append(item)
    print("No of unique images referred: ",len(list(set(l)) ))
    all_distance = {}
    for image_id in l:
        img_vector = image_matrix.loc[image_id]
        all_distance[image_id]  = spatial.distance.euclidean(img_vector, image_vector)
    # top_t_distance = dict(sorted(all_distance.items(), key=operator.itemgetter(1))[:int(t)])
    top_t_distance = {}
    top_count = 0
    for image in sorted(all_distance, key=all_distance.get):
        top_t_distance[image] = all_distance[image]
        top_count += 1
        if top_count == count:
            break

    print('======================================================== TOP t similar images====================================================')
    # print(top_t_distance)
    for k,v in top_t_distance.items():
        print('IMAGE ID :',k,'---------------------','SCORE :',v)
    visualizeImages(input_image_id, top_t_distance)


#====================================================Get Objects from MongoDB===========================================
#========================================================================================================================
def getDataFrame(image, image_score, vector_model):
    columns = [vector_model + '-' + str(n) for n in range(len(image_score[0]))]
    image_score = numpy.array(image_score).reshape(len(image_score),
                                                   len(image_score[0]))
    return pandas.DataFrame(image_score, columns=columns,
                           index=sorted(image))

#===================================================Create a matrix=====================================================
#========================================================================================================================
def get_matrix():
    VDs = ['CM', 'CN', 'CM3x3', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
    image_matrix = pandas.DataFrame()
    for vd in VDs:
        config = ConfigParser()
        config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'config.ini'))
        mongo_hostname = config.get('MongoDB', 'hostname')
        mongo_database = config.get('MongoDB', 'database')
        location_vd_collection = config.get('MongoDB',
                                                 'location_vd_collection')
        
        client = MongoClient(mongo_hostname)
        collection = client.get_database(mongo_database).get_collection(location_vd_collection)
        result = collection.aggregate([{"$match": {}},
                                       {"$project":{
                                           "_id": "$_id",
                                           "location": "$location",
                                           "images": "$"+vd+".image",
                                           "scores": "$"+vd+".scores"
                                       }}])
        client.close()
        data = pandas.DataFrame(list(result))
        query_image_list = data['images']
        query_scores_list = data['scores']
        image_score_matrix = pandas.DataFrame()
        for index in range(len(query_image_list)):
            image_score_matrix = image_score_matrix.append(getDataFrame(query_image_list[index], query_scores_list[index], vd))
        image_matrix=pandas.concat([image_matrix, image_score_matrix], axis=1)
    image_matrix = image_matrix[~image_matrix.index.duplicated(keep='first')]
    return image_matrix


#=======================================================Main function===================================================
#========================================================================================================================
image_matrix = get_matrix()
print('======================TASK 5a===========================')
L = input("Enter number of layers, L: ")
L = int(L)
k = input("Enter number of hashes per layer, k: ")
k = int(k)
n = len(image_matrix.values[0])
h = {}
for i in range(L):
    h[i] = numpy.random.randn(k, n)
mega_dic = {}
for i in range(L):
    table = {}
    per_l_dic = {}
    for image in image_matrix.index:
        row = image_matrix.loc[image].values
        table[image] = []
        for hash_index in range(k):
            table[image].append(0 if row.T.dot(h[i][hash_index]) < 0 else 1)
    p = []
    for image, hash_list in table.items():
        hash_code = ''.join(str(c) for c in hash_list)
        per_l_dic.setdefault(hash_code, [])
        per_l_dic[hash_code].append(image)
    mega_dic[i] = per_l_dic
for layer in mega_dic.keys():
    for hash_code in mega_dic[i].keys():
        len(mega_dic[i][hash_code]), hash_code, mega_dic[i][hash_code]
print("Created an in-memory index structure containing set of vectors")
print('======================TASK 5b===========================')
image_id = input("Enter an image ID: ")
# image_id = '330096935'
image_vector = image_matrix.loc[image_id]
t = input("Enter number of top images to be dispayed, t: ")
find_nn(image_vector, int(t), mega_dic, h,image_id)
