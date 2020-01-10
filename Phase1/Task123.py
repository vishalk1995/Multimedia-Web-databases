# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:36:54 2018

@author: Vishal Kumar
ASU ID : 1215200480

comments are in such a manner that each comment sections explains the code on its left hand side.
Note that I've created sections of code using "-----------" just for the ease of looking at it in forms of different sections.
"""

                                                                               #importing all the libraries                                                                
import numpy                                                                   #importing numpy for some mathematical operations
from pymongo import MongoClient                                                ##pymongo libraray is essential to establish connection between MongoDB and python
#import pprint                                                                 #(FOR Debugging purposes)pprint is used to print the json file data in a structured manner

client = MongoClient()                                                         #Creating an object for MongoClient class to intialize our connection
db=client.MWDB_Phase1                                                          #db stands for database and MWDB_Phase1 is the name of my MongoDB datasetset

#------------------------------------------------------------------------------
print("==========================INPUT START============================")     #In this section we will take required input values and work on our dataset for further usage
Task = input("Specify the Task numer: ")                                       #Specifying the task number.

if Task == "1" or Task == "2" or Task == "3":                                  #This file is for Task 1, 2 and 3 therefor applying such condition
    if Task == "1":                                                            #Task - 1 is related to user_data,
        collection = db.user_data                                              #therefore loading user_data collection
        task_id = input("Enter the user Id: ")                                 #and asking for USER_ID of query user
        task_info = collection.find_one({"USER_ID":task_id})                   #retrieving data for query_user
        task_id_name = "USER_ID"                                               #We will use this variable later to compare other users with query user based on "USER_ID" object of different documents of user collection

    elif Task == "2":                                                          #Now if task is Task 2
        collection = db.image_data                                             #getting Image_data collection
        task_id = input("Enter the image Id: ")                                #Everything is similar to the above case, we will just take image_id of query_image
        task_info = collection.find_one({"IMAGE_ID":task_id})                  #data of query image
        task_id_name = "IMAGE_ID"                                              #and now this variable is related to IMAGE_ID  

    elif Task == "3":                                                          #If task is task 3
        collection1 = db.location_mapping                                      #our input is the location no. there this collection will be used to map location no. with location title
        collection = db.location_data                                          #This collection contains data related to location based on lacation name which are nothing location titles without the "_" character.
        location_no = int(input("Enter the Location number: "))                #taking location no.
        location_mapping_object = collection1.find_one({"number":location_no}) #From collection1 finding the document which has location no. object = query location no.
        title_old = location_mapping_object["title"]                           #extracting the title of query location
        title = title_old.replace("_"," ")                                     #forming location name by removing "_" character
        task_info = collection.find_one({"LOCATION_ID":title})                 #Now extracting the location data from location_data collection based using title of the query location
        task_id_name = "LOCATION_ID"                                           #and now this variable is related to LOCATION_ID 
        task_id = title                                                        #We will compare different image documents with document using name of query location
        
        
                                                                                  
    model = input("Enter the model you want to use: ")                         #Taking the model name
    k = int(input("Enter the value of K: "))                                   #Taking the K value
    print("==========================INPUT END============================")
#------------------------------------------------------------------------------
                                                                               #This section we are performing all the calculations
    
    task_terms=[]                                                              #An array to store all the terms used by query user, image or location
    tf_df_idf_terms=[]                                                         #An array to store all the TF or DF or TF_IDF values of all terms in above list
                                                                     
    for word_object in task_info["TEXT_DESC"]:                                 #This loop iterates through all the term objects related to query user, image or location
        task_terms.append(word_object["TERM"])                                 #stores different terms in task_terms list
        tf_df_idf_terms.append(word_object[model])                             #store tf, df or idf values of corresponding terms in tf_df_idf_terms list
    
    all_term_task_mapping = {}                                                 #This dictionary is used to store different users/images/locations related to different word, it is like a 2D 
                                                                               #dictionary where key is the "Term used by query user, location or image" and value corresponding to that key is 
                                                                               #again a dictionary with task_id_name of similar user, image or location with TF, DF or TF_IDF (of word used) being its value.
    all_distinct_thing_set = set()                                             #Same user, image or location may appear for different words, so created a set to store all distinct users/images/locations related to all words.
    
    
    for term in task_terms:                                                                         #Running loop for every term used by query user.
        task_word_model_mapping = {}                                                                #A dictionary will be created for each word with task_id_name:TF as the KEY:VALUE pair, this dict corresponds to value of key in all_term_task_mapping dict. 
        similar_things = collection.find({"TEXT_DESC.TERM":term})                                   #Finding all data of users/images/locations that used some term used by query user/image/location.
        for similar_thing in similar_things:                                                        #Running loop for every user/image/location that used that word.
            if not similar_thing[task_id_name] == task_id:                                          #I just included this condition so that details of query user/image/location are not used further.
                all_distinct_thing_set.add(similar_thing[task_id_name])                             #Adding USER_ID/IMAGE_ID/LOCATION_ID of related user to our set of distinct users/images/locations. 
                for word in similar_thing["TEXT_DESC"]:                                             #Loop for every word used by similar user/image/location (this is required cause we got all the details related to that user/image/location).
                    if (word["TERM"]==term) :                                                       #When word is the same word used by query user/image/location
                        task_word_model_mapping[similar_thing[task_id_name]]=word[model]            #storing id of that user/image/location as key with TF/DF/TF_IDF value as value corresponding to that key 
                                                                                        
        all_term_task_mapping[term] = task_word_model_mapping                                       #Saving above small dictionary as value of different words(as keys) in our big dictionary.
            
    distances = {}                                                                                  #Dictionary to store distance measured with each user/image/location in a "USER_ID/IMAGE_ID/LOCATION_ID":"Distance" manner.
    
    for thing in all_distinct_thing_set:                                                            #Running loop for each user/image/location which has used any word used by query user/image/location
        distance = 0                                                                                #Declaration of distance variable which will be used to calculate the distance of the user/image/location with query user/image/location
        tf_df_idf_terms_location = 0                                                                #Variable to store the index of tf/df/tfidf value of the word in use (word for which we will run loop in next step)
        for word in task_terms:                                                                     #Running loop for each word used by query user/image/location
            tf_df_idf_thing=0                                                                       #Initially assigning tf/df/tfidf_value=0 (of word in use) for user/image/location in comparision, to handle cases where the user/image/location in comparision havn't used that word.
            tf_df_idf_query_thing = tf_df_idf_terms[tf_df_idf_terms_location]                       #Fetching tf/df/tfidf value for query user/image/location.
            if thing in all_term_task_mapping[word].keys():                                         #condition to check whether user/image/location in comparision has used that word.
                tf_df_idf_thing = all_term_task_mapping[word][thing]                                #if user/image/location has used that word then update its tf/df/tfidf value from 0 to its actual value.
            distance = distance + numpy.square(tf_df_idf_query_thing-tf_df_idf_thing)               #Distance variable storing square of distances, to calculate euclidean distance later.
            tf_df_idf_terms_location += 1                                                           #Incrementing the index for locating tf/df/tfidf value of next word.
        distance = numpy.sqrt(distance)                                                             #calculating euclidean distance.
        distances[thing]=distance                                                                   #Storing distance calculated for a user/image/location in the distances dictionary with user's USER_ID/IMAGE_ID/LOCATION_ID as KEY and corresponding distance as VALUE.
    
    k_closest_things = {}                                                      #Dictionary to store K closest users/images/locations to query user/image/location
    
    for i in range(0,k):                                                       #Loop running K times to fetch K closest users/images/locations.
        closest_thing = min(distances, key=distances.get)                      #Returns USER_ID/IMAGE_ID/LOCATION_ID of user/image/location with minimum distance.
        k_closest_things[closest_thing]=distances[closest_thing]               #Saving that USER_ID/IMAGE_ID/LOCATION_ID with its distance in k_closest_thing dictionary
        distances.pop(closest_thing)                                           #Poping that minimum KEY:VALUE pair to find next closest user/image/location
        
    top3gaps_k = {}                                                            #Dictionary to store top 3 contributing words for each closest user/image/location
        
    for thing in k_closest_things.keys():                                      #Running loop for each user/image/location in K_closest_thing.
        all_term_gap = {}                                                      #Dictionary to store gap related to each word.
        top_3_gap = {}                                                         #Dictionary to store gap of top 3 contributing words.
        tfdfidf_term_location = 0                                              #Variable to store tf/df/tfidf_value index.
        for word in task_terms:                                                #Loop for each term used by query_user/image/location
            tfdfidf_user = 0                                                   #
            tfdfidf_query_user = tf_df_idf_terms[tfdfidf_term_location]        #Same logic as used in the previous loop to calculate euclidean distance.
            if thing in all_term_task_mapping[word].keys():                    #
                tfdfidf_user = all_term_task_mapping[word][thing]              #
            gap = abs(tfdfidf_query_user-tfdfidf_user)                         #Here calculating gap between words.
            all_term_gap[word]=gap                                             #Storing that gap corresponding to user/image/location in required dictionary.
            tfdfidf_term_location += 1                                         #Incrementing Index for next tf_value.
        
        for i in range(0,3):                                                   #running loop for top 3 contributing terms.
            smallest_gap_word = min(all_term_gap, key=all_term_gap.get)        #Finding word which has minimum gap.
            top_3_gap[smallest_gap_word]=all_term_gap[smallest_gap_word]       #Storing that word in required dictionary.
            all_term_gap.pop(smallest_gap_word)                                #Poping that word to find next most contributing word.
    
        top3gaps_k[thing]=top_3_gap                                            #Storing dictionary of top 3 contributing words as VALUE corresponding the the USER_ID/IMAGE_ID/LOCATION_ID of compared user/image/location as its KEY.
    
    
#------------------------------------------------------------------------------#
    print("==========================OUTPUT START============================")#
    if Task=="1":                                                              #
        print("\nK closest Users are: \n")                                     #
    elif Task=="2":                                                            #
        print("\nK closest Images are: \n")                                    #
    else:                                                                      #
        print("\n===== QUERY LOCATION: "+title+"=====")                        # PRINTING THE OUTPUT
        print("\nK closest Locations are: \n")                                 #
    for thing in k_closest_things.keys():                                      #K closest users/image/locations in k_closest_thing
        print("Name: "+thing+"      Distance: "+str(k_closest_things[thing]))  #
        print("\nTop 3 contributing terms are:\n")                             #
        for term in top3gaps_k[thing]:                                         #Top 3 contributing words in top3gaps_k
            print(term)                                                        #
        print("\n----")                                                        #
    print("==========================OUTPUT END============================")  #
#------------------------------------------------------------------------------
    
        
        
#==============================================================================
#==========================TASK 4 & 5==========================================
#==============================================================================
elif Task=="4" or Task=="5" :
    print("Please run another file for Task 4 and 5")
else : 
    print("There are no Tasks other than task 1, 2, 3, 4 and 5")