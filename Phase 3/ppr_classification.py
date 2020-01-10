import os
import webbrowser
import time
import pandas
import shutil
import numpy as np
from sklearn.preprocessing import normalize


def calculate_page_rank():
    start_time = time.time()

    # Read json file (Output of task1)
    print("Reading graph...")
    graph = pandas.read_json("graph.json", convert_axes=False)

    print("Reading input label pair csv...")
    image_matrix = pandas.DataFrame(index=graph.columns, columns=graph.columns)

    image_matrix = image_matrix.fillna(0)
    image_matrix = image_matrix.add(graph)
    image_matrix = image_matrix.fillna(0)

    # Extract image list
    graph_image_list = list(image_matrix.index)
    #print(graph_image_list)

    image_matrix = normalize(image_matrix, axis=0, norm='l1')

    N = image_matrix.shape[1]
    d = 0.85
    maxerr = 1.0e-8

    fields = ['image', 'label']
    imageLabelPairs = pandas.read_csv("image_label_pairs.csv", skipinitialspace=True, engine='python')

    input_images = imageLabelPairs.image.tolist()
    input_labels = imageLabelPairs.label.tolist()
    image_label_pair_dict = dict(zip(input_images, input_labels))

    #print("input images", input_images)

    unique_input_labels = set(input_labels)
    unique_input_labels = list(unique_input_labels)
    #print(unique_input_labels)

    final_ppr_dict = {}
    for label in unique_input_labels:
        # Extract keys from image_label pair list with
        print("applying PPR for label = ", label)
        current_label_images = [k for k, v in image_label_pair_dict.items() if str(v) == str(label)]
        #print(current_label_images)
        # personalized_vector = np.array([0]*N)
        personalized_vector = [0]*N
        # Fill corresponding personalized_vector indexes with appropriate personalized value
        input_image_count = len(current_label_images)
        #print("curr count=" , input_image_count)
        for i in current_label_images:
            personalized_vector[graph_image_list.index(str(i))] = 1/input_image_count

        personalized_vector = np.array(personalized_vector)[:, np.newaxis]
        rank_old = np.array([1/N]*N)[:, np.newaxis]
        #print(rank_old)

        iteration_count = 0
        # Compute pagerank r until it converges
        rank_new = np.matmul(((1 - d) * image_matrix), rank_old) + d * personalized_vector
        while np.linalg.norm(rank_new - rank_old, 2) > maxerr:
            rank_old = rank_new
            rank_new = np.matmul(((1-d)*image_matrix), rank_old) + d * personalized_vector
            iteration_count = iteration_count + 1
            print("Iteration = ", iteration_count)

        # print("Total iterations for label ", label, "= ", iteration_count)
        # print("rank vector dimensions = ", rank_new.shape)
        # print("page rank vector = ")
        # print(rank_new.tolist())

        for image, rank in zip(graph_image_list, rank_new):
            if image not in final_ppr_dict:
                final_ppr_dict[image] = []
                final_ppr_dict[image].append(rank[0])
            else:
                final_ppr_dict[image].append(rank[0])

    # Assign label of highest PPR
    print("Assigning labels to rest of the images from data set...")
    dataset_image_label_dict = {}
    for image in graph_image_list:
        if image not in input_images:
            max_index = final_ppr_dict[image].index(max(final_ppr_dict[image]))
            dataset_image_label_dict[image] = unique_input_labels[max_index]
            #print("Image = ", image, "label = ", unique_input_labels[max_index])

    visualizeImages(dataset_image_label_dict, input_images, input_labels)
    print("================== End of program. Time Taken is ", time.time() - start_time, "=========================")


def visualizeImages(dataset_image_label_dict, input_images, input_labels):
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'config.ini'))
    path = config.get('visualization', 'image_path')
    htmlFile = "ppr_based_lebels.html"

    if os.path.exists(htmlFile):
        print("removing old html file...")
        os.remove(htmlFile)

    f = open(htmlFile, "a")
    header = open("htmlHeader.txt","r")
    f.write(header.read())
    # Copy header to current html file
    shutil.copystat("htmlHeader.txt", htmlFile)

    print("generating html file...")

    f.write("<div align=\"center\"> <h1><u>Image labelling using PPR based classification algorithm. </u></h1></div>\n")
    f.write("\n<div align=\"center\"> <h2><u>Input Image - Label Pairs. </u></h1></div>\n")
    f.write("<div class=\"row\">\n")

    i = 0
    j = 0
    for imageId in input_images:
        if i % 4 == 0:
            f.write("</div>\n\n<div class=\"row\"> \n")
            i = 0
        labelValue = str(input_labels[j])
        j = j + 1
        f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + str(imageId) + ".jpg\" alt=\"Image Id - " + str(imageId) + "&emsp;Label - " + labelValue + "\"/><figcaption> Image Id - " + str(imageId) + "&emsp;Label - " + labelValue + "</figcaption>\n\t\t</figure> \n\t</div>\n")
        i = i+1

    f.write("</div>\n")
    f.write("<br>\n<div align=\"center\"> <h2><u>PPR based labelled Image - Label Pairs. </u></h1></div>\n")
    f.write("<div class=\"row\">\n")

    i = 0
    for imageId in dataset_image_label_dict.keys():
        if i % 4 == 0:
            f.write("</div>\n\n<div class=\"row\"> \n")
            i = 0
        labelValue = str(dataset_image_label_dict[imageId])
        f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + str(imageId) + ".jpg\" alt=\"Image Id - " + str(imageId) + "&emsp;Label - " + labelValue + "\"/><figcaption> Image Id - " + str(imageId) + "&emsp;Label - " + labelValue + "</figcaption>\n\t\t</figure> \n\t</div>\n")
        i = i+1

    f.write("</div>\n</body>")
    webbrowser.open('file://' + os.path.realpath(htmlFile))


if __name__=='__main__':
    # Example extracted from 'Introduction to Information Retrieval'
    calculate_page_rank()
