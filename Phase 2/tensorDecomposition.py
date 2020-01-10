import sys, argparse, time
import numpy, pandas
from tensorly.decomposition import parafac
from sklearn.cluster import KMeans
from utils import mongoUtils, configReader


class tensorDecomposition:
    def __init__(self, latent_semantics):
        self.latent_semantics = latent_semantics

    def cpDecomposition(self):
        '''get the input from the user via command line'''

        '''Get the k input from command line'''
        k=self.latent_semantics

        print('Please wait.... processing!!!\n')

        ''' function definition to return the all locations data from mongo db.'''
        def get_all_objects(quantity):
            config = configReader()
            mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                            quantity)
            client = mu.get_mongodb_client()
            collection = mu.get_mongodb_collection(client)
            objlist = collection.find()
            return objlist

        '''get al the terms for a given user/location/image'''
        def get_all_terms(item):
            termarray=[]
            for text in item['TEXT_DESC']:
                termarray.append(text['TERM'])
            return termarray

        '''Get the Userlist'''
        config = configReader()
        userlist=get_all_objects(config.user_td_collection)
        '''Get the location list'''
        locationlist=get_all_objects(config.location_td_collection)
        ''' Get the image list'''
        imagelist=get_all_objects(config.image_td_collection)

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
        for user in userlist:
            userterms[user['USER_ID']]=get_all_terms(user)
            userids.append(user['USER_ID'])

        output_filename = 'save_data' + '.npy'
        #numpy.save(output_filename, tensor)

        ''' Load the file from local disk and process'''
        a=numpy.load(output_filename)

        '''Perform the rank k decomposition on the tensor'''
        factors=parafac(a,k)

        ''' Get the kmean for the users'''
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10,random_state=0).fit(factors[0])
        result=pandas.DataFrame(kmeans.labels_, index=userids)
        result.columns=['Group']

        '''' Get the kmean for the locations'''
        kmeans_location = KMeans(n_clusters=k, init='k-means++', n_init=10,random_state=0).fit(factors[1])
        result_loc=pandas.DataFrame(kmeans_location.labels_, index=locationids)
        result_loc.columns=['Group']

        '''' Get the kmean for the images'''
        kmeans_image = KMeans(n_clusters=k, init='k-means++', n_init=10,random_state=0).fit(factors[2])
        result_img=pandas.DataFrame(kmeans_image.labels_, index=imageids)
        result_img.columns=['Group']


        ''' loop throught the list and display the groups'''
        for i in range(0,k):
            print('Group '+str(i+1)+' Users')
            globals()['Group%s_User' %i]= result.index[result['Group']==i].tolist()
            print('number of users='+str(len(globals()['Group%s_User' %i])))
            print(globals()['Group%s_User' %i])

            print('Group '+str(i+1)+' Locations')
            globals()['Group%s_Location' %i]= result_loc.index[result_loc['Group']==i].tolist()
            print('number of Locations=' + str(len(globals()['Group%s_Location' % i])))
            print(globals()['Group%s_Location' %i])

            print('Group '+str(i+1)+' Images')
            globals()['Group%s_image' %i]= result_img.index[result_img['Group']==i].tolist()
            print('number of Images=' + str(len(globals()['Group%s_image' % i])))
            print(globals()['Group%s_image' %i])

            print('\n\n')


def readCommand(argv):
    parser = argparse.ArgumentParser(description=('Find latent semantics and'
                                                  + ' similar objects using'
                                                  + ' visual descriptors'))
    parser.add_argument('-k', '--latent-semantics', type=int, required=True,
                        help='Number of latent semantics to be found')
    return parser.parse_args(argv)


if __name__ == '__main__':
    start_time = time.time()
    args = readCommand(sys.argv[1:])
    latent_semantics = args.latent_semantics
    ls = tensorDecomposition(latent_semantics)
    ls.cpDecomposition()
    print("Execution completed in", time.time() - start_time, "seconds")
    sys.exit(0)
