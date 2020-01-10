# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 12:47:20 2018

@author: Vishal Kumar
ASU ID : 1215200480

comments are in such a manner that each comment sections explains the code on its left hand side.
Note that I've created sections of code using "-----------" just for the ease of looking at it in forms of different sections.
"""
                                                                               #importing all the libraries
import scipy                                                                   #scipy used for cosine function
from scipy import spatial                                                      #
import numpy                                                                   #importing numpy for some mathematical operations
from pymongo import MongoClient                                                #pymongo libraray is essential to establish connection between
#import pprint                                                                 #(For debugging purpose)pprint is used to print the json file data in a structured manner
import time                                                                    #Library to measure the runtime of most complex part of this code.

client = MongoClient()                                                         #Creating an object for MongoClient class to intialize our connection
db=client.MWDB_Phase1                                                          #db stands for database and MWDB_Phase1 is the name of my MongoDB datasetset
collection1 = db.location_mapping                                              #Collection which has location title corresponding to location number.        
collection = db.location_vd                                                    #Collection which has location data corresponding to location title.
 
#------------------------------------------------------------------------------ 
print("==========================INPUT START============================")     #Input section 
task = int(input("Enter the task number: "))                                   #Getting task number


if task==4 or task==5:                                                                                  #This file is only for task 4 and 5    
    location_no = int(input("Enter the Location number: "))                                             #location number input
    location_mapping_object = collection1.find_one({"number":location_no})                              #getting location title using location number.
    title_old = location_mapping_object["title"]                                                        #title_old variable is used to store the original title.
    title = title_old.replace("_"," ")                                                                  #title variable is used to store the title name i.e. the actual title without "_" character
    task_info = collection.find_one({"location":title_old})                                             #getting all the info of desired location using its title
    
    if task == 4:                                                                                       #TASK 4 - when task is related to one Model only 
        model_name = input("Enter the model you want to use: ")                                         #getting model name
        model_list=[]                                                                                   #logic of mainting a list is to do TASK 4 and 5 using same piece of code, we will iterate over this list to calculate distance.
        model_list.append(model_name)                                                                   #inserting that model name in our list

    elif task == 5:                                                                                     #If it is TASK 5
        model_list=["CM", "CM3x3", "CN", "CN3x3", "CSD", "GLRLM", "GLRLM3x3", "HOG", "LBP", "LBP3x3"]   #Over list will have all model names.
    
    k = int(input("Enter the value of K: "))                                                            #Getting K value irrespective of task
    print("==========================INPUT END============================")
#------------------------------------------------------------------------------
    
    
    
    distances = {}                                                             #Dictionary to store distance measured with each location in a "LOCATION_ID":"Distance" manner.
    distance_flag = 0                                                          #This will come handy for TASK 5 i.e. following loop runs more than 1 time, further explaination at usage point
    all_model_contributions ={}                                                #A 2D dictionary with intial key as the Model name and correspoding value is again a dictionary with "LOCATION_ID":"Distance as per that model" as key:value pair
    start_time = time.time()                                                   #Variable to store start time before the loop execution
    for model_object in model_list:                                            #running loop for every model present in model list
        print("Running on model: "+model_object)                               #printing on which model the loop is currently running.
        model = model_object                                                   #Storing model name in a variable (I modified my task 4 to task 5, and in task 4 I used "model" variable)
        all_model_contributions[model] = {}                                    #Sometime python needs to initialize the value corresponding to a key in dictionary.
        image_model_score = {}                                                 #Dictionary to store "image name":"VD as per model" in KEY:VALUE manner         
        for objects in task_info[model]:                                       #running loop on image objects under the model object (outer loop model value) for the query location
            image_model_score[objects["image"]]=objects["scores"]              #saving in the desired dictionary for later calculations

        #------------------------------------------------------------------------------
        all_location_names=[]                                                  #list to store all location names except query location
        all_locations_score = {}                                               #Dictionary with initial key as location name and value as a dictionary of "image":"VD as per model"
        all_location_data = collection.find({})                                #Getting data related to all VD of all locations
        for location in all_location_data:                                     #loop for each location (document) under all_location_data (i.e. locations_VD collection)
            location_model_score={}                                            #dictionary which will be value corresponding to key in all_locations_score dictionary
            if location["location"] != title_old:                              #when location is different as query location
                all_location_names.append(location["location"])                #storing location name in desired list
                for objects in location[model]:                                #loop for each object(which contains image name and score value) under model object of a location document
                    location_model_score[objects["image"]]=objects["scores"]   #storing image:vd value
            all_locations_score[location["location"]]=location_model_score     #storing data in this list as explained above duing its declaration
            
        for name in all_location_names:                                                                                                                     #loop to cal distance with each location
            distance_per_model = 0                                                                                                                          #variable to store final distance with some location.(Remember - loop running for certian value of model)
            cos_image_group = 0                                                                                                                             #variable to store cos distance of 1 image of query location with all other images of other location (Remember - loop running for certain value of model)
            for image_key in image_model_score.keys():                                                                                                      #running loop for each image in this list to calculate its distance with images of other location
                cos_image_image = 0                                                                                                                         #variable to store cos distance between to images
                for image_key2 in all_locations_score[name]:                                                                                                #Loop for each image in other location
                    cos_image_image = cos_image_image + scipy.spatial.distance.cosine(image_model_score[image_key],all_locations_score[name][image_key2])   #image to image cosine distance
                cos_image_group = cos_image_group + cos_image_image                                                                                         #aggregating values of all image-image distances.
            normalized_cos = cos_image_group/(len(image_model_score)*len(all_locations_score[name]))                                                        #taking average of of all different image-image scores
            distance_per_model = normalized_cos                                                                                                             #Now this average/normalized distance is our distance of query location with some other location under a certain model.
            all_model_contributions[model][name]=distance_per_model                                                                                         #storing above calculated distance in a dictionary to know for later usage.(This dictionary explained above)
            if distance_flag == 0:                                                                                                                          #Now this will run for both task 4 and 5.
                distances[name] = normalized_cos                                                                                                            #for Task 4, this is final distance, thus storing distance as value to location name as key 
            elif distance_flag != 0:                                                                                                                        #this will only run in case of Task 5 i.e when loop runs agin for another model
                distances[name] = distances[name] + normalized_cos                                                                                          #Here adding new distance (according to new model) to the distance value of location (as key)
        
        distance_flag += 1                                                     #increment of distance flag for usage when loop runs again
        
    for name in all_location_names:                                            #for every location
        distances[name]=distances[name]/10                                     #taking average of all 10 different distances(each as per one model)
    
    distances_backup=distances.copy()                                          #backup of distance dictionary
    
    k_closest_locations = {}                                                   #Dictionary to store K closest locations
    
    for i in range(0,k):                                                       #Loop running K times to fetch K closest locations.
        closest_location = min(distances, key=distances.get)                   #Returns location title of user with minimum distance.
        k_closest_locations[closest_location]=distances[closest_location]      #Saving that Location with its distance in k_closest_locations dictionary
        distances.pop(closest_location)                                        #poping r4esult we got in above steps to find next minimum

    if task == 4:                                                              #further requirement of Task 4
        top3gaps_k = {}                                                        #Dictionary to store top 3 contributing Image - Image pairs for each closest location
        all_distance_gap = {}                                                  #A dictionary with location as key and value is again a dictionary with "Image pair name":"Distance between them" as key:value pair
        for name in k_closest_locations.keys():                                #loop for each closest location
            top_3_gap = {}                                                     #dictionary to top 3 image pairs with minimum distance (this dictionary will be used as the value for top3gaps_k dictionary)
            all_distance_gap[name]={}                                          #empty initialization
            for image_key in image_model_score.keys():                                                                                              #
                cos_image_image = 0                                                                                                                 #
                for image_key2 in all_locations_score[name]:                                                                                        # This is the exact method as explained earlier to calculate cosine distance between 2 images
                    cos_image_image =  abs(scipy.spatial.distance.cosine(image_model_score[image_key],all_locations_score[name][image_key2]))       #
                    pair_name = str(str(image_key)+" "+str(image_key2))                                                                             #
                    all_distance_gap[name][pair_name]=cos_image_image                                                                               #storing that distance in thedesired array with "Image_name Image_name" being its key.
            
            for i in range(0,3):                                                                                                                    #running loop for top 3 contributing pairs.
                smallest_gap_image = min(all_distance_gap[name], key=all_distance_gap[name].get)                                                    #Finding pair which has minimum gap.
                top_3_gap[smallest_gap_image]=all_distance_gap[name][smallest_gap_image]                                                            #Storing that pair in required dictionary.
                all_distance_gap[name].pop(smallest_gap_image)                                                                                      #Poping that pair to find next most contributing pair.
        
            top3gaps_k[name]=top_3_gap                                                                                                              #Storing dictionary of top 3 contributing pair as VALUE corresponding the location of compared location as its KEY.
        end_time=time.time()                                                                                                                        #Getting time at this point of execution.
        time_taken = end_time-start_time                                                                                                            #time taken to execute till here.
        #----------------------------------------------------------------------
        print("\n")                                                            #
        print("===== QUERY LOCATION: "+title+"=====")                          #
        print("\n==============TASK 4 - OUTPUT START=====================\n")  #PRINTING OUTPUT FOR TASK 4
        for objects in k_closest_locations.keys():                             #Closest locations with their distances in k_closest_location
            title = objects.replace("_"," ")                                   #
            print("LOCATION NAME: "+title)                                     #
            print("SCORE: "+str(k_closest_locations[objects]))                 #
            print("\nTop 3 Image-Image pairs are: \n")                         #
            for location_pair in top3gaps_k[name]:                             #Top 3 image pair contributors in top3gaps_k                
                print(location_pair)                                           #
            print("\n")                                                        #
            print("---")                                                       #
        print("\nTime Taken for Task 4: "+str(time_taken)+" seconds")          #
        print("\n==============TASK 4 - OUTPUT END=====================\n")    #
            
    elif task == 5:                                                            #If Task is Task 5
        end_time=time.time()                                                   #Getting time at this point of execution.
        time_taken = end_time-start_time                                       #time taken to execute till here.
        print("\n===== QUERY LOCATION: "+title+"=====")                        #PRINTING OUTPUT FOR TASK 5
        print("\n==============TASK 5 - OUTPUT START=====================\n")  #
        for objects in k_closest_locations.keys():                             #Closest locations with their distances in k_closest_location
            title = objects.replace("_"," ")                                   #
            print("LOCATION NAME: "+title)                                     #
            print("OVERALL SCORE: "+str(k_closest_locations[objects]))         #
            print("\n")                                                        #
            print("Corresponding model contributions are: \n")                 #
            for model in all_model_contributions.keys():                       #Contribution of each model for each location in all_model_contribtuion
                value = all_model_contributions[model][objects]                #
                print("\n"+model+" : "+str(value))                             #
            print("\n---")                                                     #
        print("\nTime Taken for Task 5: "+str(time_taken)+" seconds")          #
        print("\n==============TASK 5 - OUTPUT END=====================\n")    #
            
    #------------------------------------------------------------------------------
else:
    print("\nRun another file for Task 1,2,3 and 4\n")
##================================================================================================================= 