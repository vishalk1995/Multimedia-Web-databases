import sys, argparse
import pandas, numpy
import time
from scipy.spatial import distance_matrix
from utils import mongoUtils, configReader, dimentionalityUtils

class locationSimilarity:
    def __init__(self, latent_semantics):
        '''
        Constuctor for class
        :param latent_semantics: Number of latent semantics to be returned
        '''
        self.latent_semantics = latent_semantics

    def getAllLocations(self):
        '''
        Function definition to return the all locations data from mongo db.
        :return: MongoDB cursor object
        '''
        config = configReader()
        mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                        config.location_td_collection)
        client = mu.get_mongodb_client()
        collection = mu.get_mongodb_collection(client)
        locationlist = collection.find()
        return locationlist

    def getLocationMatrix(self):
        '''
        Function to generate location-location similarity matrix
        :return:
        pandas DataFrame: location-location similarity matrix
        list: list of locations
        '''
        locationlst=self.getAllLocations()
        terms_locations = []
        loc_term_coll=[]
        loc_names=[]
        loc_term_coll_dist=dict()
        loc_term_coll_IDF_dist=dict()
        '''loopp through all the location to get distinct terms'''
        for obj in locationlst:
            loc_names.append(obj['LOCATION_ID'])
            terms_locations_temp1 = []
            terms_locations_IDF_temp1 = []
            for term in obj['TEXT_DESC']:
                terms_locations_temp1.append(term['TERM'])
                terms_locations_IDF_temp1.append(term['TF_IDF'])
                if not term['TERM'] in terms_locations:
                  terms_locations.append(term['TERM'])
            loc_term_coll_dist[obj['LOCATION_ID']]=terms_locations_temp1
            loc_term_coll_IDF_dist[obj['LOCATION_ID']]=terms_locations_IDF_temp1

        ''' Get the all location data'''
        locationlst=self.getAllLocations()

        ''' loop through the locations to construcrt the location term matrix'''
        for obj in locationlst:
            terms_locations_temp = []
            for term in terms_locations:
                if term in loc_term_coll_dist[obj['LOCATION_ID']]:
                    value_index = loc_term_coll_dist[obj['LOCATION_ID']].index(term)
                    terms_locations_temp.append(loc_term_coll_IDF_dist[obj['LOCATION_ID']][value_index])
                else:
                    terms_locations_temp.append(0)
            loc_term_coll.append(terms_locations_temp)

        ''' find the location to location similarity matrix'''
        loc_term_coll = numpy.array(loc_term_coll)
        df_col = pandas.DataFrame(loc_term_coll, columns=terms_locations, index=loc_names)
        df1=pandas.DataFrame(distance_matrix(df_col.values, df_col.values), index=df_col.index, columns=df_col.index)
        return df1, loc_names

    def computeSemantics(self, matrix, loc_names):
        '''
        Applies SVD on given location-location similarity matrix
        :return: None
        '''
        ''' SVD decomposition of matrix'''
        du = dimentionalityUtils('SVD', matrix, self.latent_semantics)
        U, s, VT = du.apply_transformation()
        df= pandas.DataFrame.from_records(U[:,0:self.latent_semantics],index=loc_names)

        length=len(loc_names)
        '''loop through the latent features to display the term weight pair'''
        for i in range(self.latent_semantics):
            print("Sementic Number "+str(i+1))
            zipbObj = zip(loc_names, df[i])
            dictOfWords = dict(zipbObj)
            t2 = sorted(dictOfWords.items(), key=lambda x: -x[1], reverse=False)[:length]
            df_result = pandas.DataFrame.from_records(t2, columns=['Location', 'Weight'])
            print(df_result)
            print("\n\n")

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
    ls = locationSimilarity(latent_semantics)
    matrix, location_names = ls.getLocationMatrix()
    ls.computeSemantics(matrix, location_names)
    print("Execution completed in", time.time() - start_time, "seconds")
    sys.exit(0)
