Author: Vishal Kumar, ASU ID: 1215200480
(description of each file is given at the end of this file)
-------------------------------------------------------------------------------------------------------------------------------
CSE 515 Multimedia and Web Databases
Project Phase: 1

Problem Description:
Phase 1 deals with the Textual and Visual Descriptors used by some users, images or locations. we are required to solve the problem of finding the similar users/images/location to some given user/image/location under certain conditions.

-------------------------------------------------------------------------------------------------------------------------------
Pre-requisite for my solution:

1. Python 3.7 (our base programming language):
Download available at - https://www.python.org/downloads/

2. MongoDB: This is my database management system.
Download available at - https://www.mongodb.com
 

3. Numpy and Scipy libraries.
Installation using pip command
-------------------------------------------------------------------------------------------------------------------------------
We wil store given data (text, XML and CSV files) in form of json in MongoDB, to be able to do that we first need to convert these files into json file. Two Team members collectively created shell scipts to do that, I have also included these shell scripts in "mwdb_scripts.zip"

How to use scripts:-
1. For text files:
converting text file for user textual descriptor:
$ bash scripts/parseUserText.sh <file_path_of_text_file>
Loading this Textual descriptor data in MongoDB:
$ bash scripts/loadData.sh <file_generated> <db_name> <collection_name>

similar for textual descriptor text file for location and images.

2. For XML file of location info(also loading into MongoDB):
$ python scripts/mongoInsertLocationInfo.py --xml-file <path_to_devset_topics> \
    --database <db_name> --collection <collection_name>

3. For CSV files of visual descriptors(also loading into mMngoDB):
$ bash scripts/load_VD.sh <path_to_visual_descriptor_folder> <db_name> <collection_name>

use "MWDB_Phase1" as your database so that solution python files can link with MongoDB.
(similarly use collection names as described in the following section)
-------------------------------------------------------------------------------------------------------------------------------
MongoDB usage:
(
My MongoDB details - 
Name of my database = MWDB_Phase1
Name of my collections under MWDB_Phase1 -
1. image_data         (corresponds to TD_images.bson and TD_images.metadata.json)
2. location_data      (corresponds to TD_location.bson and TD_locations.metadata.json)
3. user_data          (corresponds to TD_users.bson and TD_users.metadata.json)
4. location_mapping   (corresponds to location_info.bson and location_info.metadata.json)
5. location_vd        (corresponds to VD_locations.bson)
)

-------------------------------------------------------------------------------------------------------------------------------
This section is not important if you want to create your own JSON and store them in MongoDB(using commands explained in previous section), But If you want Use our JSON files then read ahead

I've provided all the required json files in a file named "json_files.zip"

Each metadata file contains the schema of corresponding bson files, so while creating collections in MongoDB please load both files in same collection, later you delte the document related to the metadata json file.

Following are quries required to load data:
1. open CMD, run: mongod
2. open another CMD window, run the follwing commands

(these json and bson files for a perticular collection are actually dumped files created my our team, that is why you need to import both)
query to import metadata:
mongoimport -d databasename -c collectionname <json_file_path>

query to import actual data:
mongorestore -d databasename -c collectionname <bson_file_path>

here databasename, collectionname will the name of the database, collection you want to create.

use "MWDB_Phase1" as your database so that solution python files can link with MongoDB.
(similarly use collection names as described at the top of the section)
-------------------------------------------------------------------------------------------------------------------------------

Executing files:
please make sure you have python 3.7 installed with numpy and scipy as additional libraries.
installing numpy and scipy:
1. pip install numpy
2. pip install scipy

installing pip using CMD:
python get-pip.py

Execution of solution files:
1. Open command prompt.
2. go to location where you have these solution file usind CD command (CD <File Path>).
3. To run program for Task 1,2 or 3, execute the following command
	python Task123.py
4. To run program for Task 4 & 5, execute the following command
	python Task45.py

The program will ask for your desired inputs and will give corresponding appropriate output.
-------------------------------------------------------------------------------------------------------------------------------
Acknowledgments:
Thanks to Professor K. Selçuk Candan for helping with concepts required to complete this phase. Also, 
thanks to all my team members for comingup with creative ideas and solutions required to complete this task.


-------------------------------------------------------------------------------------------------------------------------------
File Details:
1.  json_files.zip - 		contains all json and bson files.
2.  mwdb_scripts.zip -   	contains all script files.
3.  Report.pdf -		My report for phase 1.
4.  Task123.py - 		Python file for Task 1, 2 and 3.
5.  Task45.py - 		Python file for Task 4 and 5.
6.  Task_1_output.html -	Output of Task 1.
7.  Task_2_output.html -	Output of Task 2.
8.  Task_3_output.html -	Output of Task 3.
9.  Task_4_output.html -	Output of Task 4.
10. Task_5_output.html -	Output of Task 5.
-------------------------------------------------------------------------------------------------------------------------------
Thanks for reading.