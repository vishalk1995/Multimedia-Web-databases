from pymongo import MongoClient
import numpy
import tensorly as t
from utils import mongoUtils, configReader
'''get the input from the user via command line'''
'''location_id=int(input('Enter the locationID:'))'''
'''model=input('Enter the model:')'''

print('Please wait.... processing!!!\n')

'''function definition toaccept host name,
db name and collectionname and returns  the db
object'''

def get_mongodb_collection(mongodb_host, database, collection):
    client = MongoClient(mongodb_host)
    db = client.get_database(database)
    return db.get_collection(collection)

''' function definition to return the all locations data from mongo db.'''
def get_all_objects(quantity):
    config = configReader()
    mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                    quantity)
    client = mu.get_mongodb_client()
    obj = mu.get_mongodb_collection(client)
    objlist = obj.find()
    client.close()
    return objlist

def get_all_terms(item):
    termarray=[]
    for text in item['TEXT_DESC']:
        termarray.append(text['TERM'])
    return termarray

config =configReader()
userlist=get_all_objects(config.user_td_collection)
locationlist=get_all_objects(config.location_td_collection)
imagelist=get_all_objects(config.image_td_collection)
'''
usercount=len(userlist)
locationcount=len(locationlist)
imagecount=len(imagelist)
'''
userterms=dict()
locationterms=dict()
imageterms=dict()

userids=[]
locationids=[]
imageids=[]

tensor=[]
'''for user in userlist:
    userterms[user['USER_ID']]=get_all_terms(user)'''
for location in locationlist:
    locationterms[location['LOCATION_ID']]=get_all_terms(location)
    locationids.append(location['LOCATION_ID'])
for image in imagelist:
    imageterms[image['IMAGE_ID']]=get_all_terms(image)
    imageids.append(image['IMAGE_ID'])
i=0
j=0
k=0
userarray=[]
'''loop through the userlist '''
for user in userlist:
    '''if i==35:
        break'''
    userterms=get_all_terms(user)
    userids.append(user['USER_ID'])
    locationarray=[]
    #j=0
    '''loop through the location list to find the common terms'''
    for location in locationids:
        '''if j==35:
         break'''
        common=list(set(userterms).intersection(locationterms[location]))
        imagearray=[]
        #k=0
        ''' loop throught he image list to get the common terms'''
        for image in imageids:
            '''if k==10:
                break'''
            updatecommon=list(set(common).intersection(imageterms[image]))
            imagearray.append(len(updatecommon))
            #k=k+1
        locationarray.append(imagearray)
        #j=j+1
    userarray.append(locationarray)
    #i=i+1

# Decompose tensor using CP-ALS
'''P, fit, itr, exectimes = cp_als(userarray, 3, init='random')'''
'''print(userarray)'''
tensor = t.tensor(userarray)
''' Save the tensor into the local file '''
output_filename = 'save_data' + '.npy'
numpy.save(output_filename, tensor)
