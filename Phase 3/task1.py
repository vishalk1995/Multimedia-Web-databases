import pandas as pd
import time
import json

k = int(input("\nPlease enter the value of K:\n"))
s_time = time.time()
img_img_sim=pd.read_csv("matrix.csv")                                                               #similarity matrix between images
print("\ntime in reading csv: "+str(time.time() - s_time)+"\n")
img_img_sim = pd.DataFrame(img_img_sim.values, index=img_img_sim.columns, columns=img_img_sim.columns)  #converting it into dataframe with image names as labels on index and columns

img_img_graph = {}                                                                              #this will be a 2d dictionary to stor graph

for index, row in img_img_sim.iterrows():                                           #running loop for each row with row label on hand
    graph_pair_dict = {}                                                            #dictionary that will contain neighbors of each a perticular image
    row_copy = row.copy()
    for k_iteration in range(0,k+1):                                                #k+1 iteration for each image to avoid self-similarity
        max_val = max(row_copy)
        index_list = list(row_copy.index[row_copy == max_val])
        index_val = index_list[0]
        if not index_val == index:                                                  #if no self-similarity then will the dictionary
            graph_pair_dict[index_val]=max_val
        row_copy.pop(index_val)                                                     #removing the closest image to avoid another encounter
        max_val = 0
    img_img_graph[index]=graph_pair_dict                                            #filling the 2D dictionary

#Saving to json
filename = 'graph.json'
with open(filename, 'w') as writing_f:                                              #saving the output as a json file
    json.dump(img_img_graph, writing_f)

#loading json
#with open(filename) as load_f:
#    img_img_sim_graph = json.load(load_f)                                       #img_img_sim and img_img_sim_graph are same

print("Total time taken: "+ str(time.time() - s_time)+"\n")
