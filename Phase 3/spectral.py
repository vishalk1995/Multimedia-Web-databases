import os, webbrowser
import numpy as np
import pandas as pd
import time
import json
from numpy.linalg import eig

k =int(input("\nPlease specify the number of neighbors:"))                                                   # take from task1
c =int(input("\nPlease specify the number of clusters:"))
s_time = time.time()

filename = 'graph.json'
with open(filename) as load_f:                                                                              #Taking graph from task1
    img_img_graph_dict = json.load(load_f)

image_name = list(img_img_graph_dict.keys())
img_img_zero_matrix = np.zeros((8912, 8912))
laplacian_df = pd.DataFrame(img_img_zero_matrix, index=image_name, columns=image_name)    #filling Laplacian with zeroes and images labels in index
del img_img_zero_matrix                                                                                       #of no use now

for index, row in laplacian_df.iterrows():
    k_neighbour_list = img_img_graph_dict[index]                                                                #filling the laplacian matrix
    for neighbour in k_neighbour_list:
        laplacian_df[index][neighbour] = -1
    row[index] = k

eigenvalues, eigenvectors = eig(laplacian_df)                                                       #calculating eiganvalues and eiganvectors of laplacian matrix
eiganvaluescopy = list(eigenvalues)

min_eigan = min(eiganvaluescopy)

eiganvaluescopy.remove(min_eigan)

min2_eigan = min_eigan = min(eiganvaluescopy)                                                             #getting second smallest eigan value

min2_index = list(eigenvalues).index(min2_eigan)                                                  #index of that eigan value

vector_min2_array = eigenvectors[:,min2_index]                                                      #getting desired eigan vector
vector_min2 = pd.DataFrame(list(vector_min2_array), index = image_name)                    #putting image names as index labels
sorted_vector_min2 = vector_min2.sort_values([0],ascending = False)                                 #Sorting the desired eigan vector

count = []
for i in range(0,c):                                                                                #Creating empty cluster and a list containg number of elements in each cluster
    globals()['group%s' % i] = []
    count.append(0)


def divisionele(array, avg):                                                                        #function to find element with eigan-vector value closest to the mean
    array = np.asarray(array)
    index_val = (np.abs(array - avg)).argmin()
    return array[index_val], index_val

temp_grp1 = []                                                                                      #temporary groups for hadling exchange and division
temp_grp2 = []
intial_pos_len = 0                                                                                  #variables to store length intial groups
intial_neg_len = 0


for element in sorted_vector_min2.index:
    if sorted_vector_min2[0][element] > 0 or sorted_vector_min2[0][element] == 0:
        temp_grp1.append(element)                                                                   #Initial division of elements as per the rules describes in report
    if sorted_vector_min2[0][element] < 0:
        temp_grp2.append(element)

intial_pos_len = len(temp_grp1)                                                                     #storing length of each group
intial_neg_len = len(temp_grp2)

if intial_pos_len > intial_neg_len:                                                                 #
    globals()['group%s' % 0] = temp_grp2                                                            #
    count[0] = intial_neg_len                                                                       #keeping group with smaller length
elif intial_pos_len < intial_neg_len:                                                               #
    globals()['group%s' % 0] = temp_grp1                                                            #
    count[0] = intial_pos_len                                                                       #


sorted_small = sorted_vector_min2
sorted_small = sorted_small.drop(globals()['group%s' % 0], axis = 0)                                #this sorted_small is our 'non-cluster' group, it is sorted and smaller than the original list
count.append(len(sorted_small))                                                                     #count[c] will be used to store the value of sorted_small set

for c_count in range(1,c):
    if not c_count == (c-1):                                                                        #if cluster is no the last cluser to be formed
        largest_grp_size = max(count)                                                               #largest gorup size
        largest_grp_no = count.index(largest_grp_size)                                              #Getting gorup number
        if not largest_grp_no == c:                                                                 #If lagest group is not the 'non-cluster' group then swap it with 'non-cluster' group
            temp_grp = globals()['group%s' % largest_grp_no]
            temp_count = count[largest_grp_no]
            globals()['group%s' % largest_grp_no] = list(sorted_small.index)
            count[largest_grp_no] = len(globals()['group%s' % largest_grp_no])
            sorted_small = sorted_vector_min2
            sorted_small = sorted_small.loc[temp_grp]
            count[c] = temp_count

        average = sorted_small[0].mean()                                                        #getting the mean of 'non-cluster' group
        divi_ele_value, divi_ele_index = divisionele(sorted_small[0], average)
        divi_ele_label = sorted_small.index[sorted_small[0] == divi_ele_value][0]               #getting the label(image name) of element with value closest to mean
        temp_grp1 = list(sorted_small.loc[:divi_ele_label].index)                               #diving into 2 groups as explained in documentation
        temp_grp2 = sorted_small
        temp_grp2 = list(temp_grp2.drop(temp_grp1, axis = 0).index)
        if len(temp_grp1) > len(temp_grp2):                                                     #
            globals()['group%s' % c_count] = temp_grp2                                          #
            count[c_count] = len(temp_grp2)                                                     #keeping the one with smaller length
        elif len(temp_grp1) < len(temp_grp2):                                                   #
            globals()['group%s' % c_count] = temp_grp1                                          #
            count[c_count] = len(temp_grp1)                                                     #
        sorted_small = sorted_small.drop(globals()['group%s' % c_count], axis = 0)              #sending all other to 'non-cluster' group
        count[c] = len(sorted_small)                                                            #keeping track on count

    elif c_count == (c-1):                                                                          #if the cluster is the las t cluster to be formed than
        globals()['group%s' % c_count] = list(sorted_small.index)                                   #put all remaiing elements in the cluster
        count[c_count] = len(globals()['group%s' % c_count])

count.pop()                                                                                     #removing the count of 'non-cluster' group,  unnecessary now.
print("\nTime taken till now - \n"+str(time.time()-s_time))

''' output is present in groups, named group0 to group(c-1)
    accessed using globals()['group%s'% i]
'''

for i in range(0,c):
    print("\n====================== GROUP "+str(i)+"=======================\n")                 #printing the output
    print(globals()['group%s'% i])
    print("\n==============================================================\n")

## Count for each group
for i in range(0,c):
    print("\nNumber of elements in Group "+str(i)+": "+str(count[i]))

def visualize():
    '''
    Create and open HTML file for visualizing clusters
    :param clusters: Dictionry representing clusters
    '''
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'config.ini'))
    html_file = 'spectral.html'
    path = config.get('visualization', 'image_path')
    if os.path.exists(html_file):
        print("removing old html file...")
        os.remove(html_file)
    f = open(html_file, "w")
    header = open('htmlHeader.txt','r')
    f.write(header.read())
    f.write("<div align=\"center\"> <h1><u>Image Clustering using spectral clustering algorithm. </u></h1></div>\n")
    f.write("<div class=\"row\">\n")
    for cluster in range(0,c):
        f.write("</div>\n")
        f.write("<br>\n<div align=\"center\"> <h2><u> Cluster "+ str(cluster + 1) +": </u></h1></div>\n")
        f.write("<div class=\"row\">\n")
        i = 0
        for imageId in globals()['group%s'% cluster]:
            if i % 4 == 0:
                f.write("</div>\n\n<div class=\"row\"> \n")
                i = 0
            f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + imageId + ".jpg\" alt=\"Image Id - " + imageId + "\"/><figcaption> Image Id - " + imageId + "</figcaption>\n\t\t</figure> \n\t</div>\n")
            i = i+1
    f.write("</div>\n</body>")
    webbrowser.open('file://' + os.path.realpath(html_file))

visualize()
