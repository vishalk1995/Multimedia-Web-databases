import sys, os
import configparser
from numpy import linalg
from pymongo import MongoClient
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import StandardScaler

class dimentionalityUtils():
    def __init__(self, method, matrix, dimensions=None):
        '''
        Constructor method for class
        :param method: Method for dimensionality reduction
        :param matrix: Data matrix
        :param dimensions: Number of dimensions to keep after reduction
        '''
        self.method = method
        self.matrix = matrix
        self.dimensions = dimensions

    def apply_transformation(self):
        '''
        Applies transformation based on method and returns relavent data
        :return:
            Resultant matrices of dimensionality reduction
        '''
        if self.method == 'PCA':
            sc = StandardScaler()
            matrix = sc.fit_transform(self.matrix)
            if self.dimensions is None:
                pca = PCA()
            else:
                pca = PCA(n_components=self.dimensions)
            u = pca.fit_transform(matrix)
            s = pca.explained_variance_
            vt = pca.components_
            return u, s, vt
        elif self.method == 'SVD':
            svd = TruncatedSVD(n_components=self.dimensions)
            u = svd.fit_transform(self.matrix)
            s = svd.explained_variance_
            vt = svd.components_
            return u, s, vt
            #u,s,vt = linalg.svd(self.matrix)
            #if self.dimensions is None:
            #    return u, s, vt
            #else:
            #    return u[:, :self.dimensions], s, vt[:self.dimensions, :]
        elif self.method == 'LDA':
            sc = StandardScaler()
            matrix = sc.fit_transform(self.matrix)
            if self.dimensions is None:
                lda = LatentDirichletAllocation()
            else:
                lda = LatentDirichletAllocation(n_components=self.dimensions)
            matrix_min = matrix.min()
            if matrix_min < 0:
                matrix = matrix - matrix_min
            u=lda.fit_transform(matrix)
            vt = lda.components_
            return u, None, vt

class configReader:
    '''
    Class for reading config.ini
    '''
    def __init__(self):
        '''
        Constructor for class. Initializes configuration variables
        '''
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'config.ini'))
        self.mongo_hostname = config.get('MongoDB', 'hostname')
        self.mongo_database = config.get('MongoDB', 'database')
        self.user_td_collection = config.get('MongoDB',
                                             'user_td_collection')
        self.image_td_collection = config.get('MongoDB',
                                              'image_td_collection')
        self.location_td_collection = config.get('MongoDB',
                                                 'location_td_collection')
        self.location_vd_collection = config.get('MongoDB',
                                                 'location_vd_collection')
        self.location_info_collection = config.get('MongoDB',
                                                   'location_info_collection')

class mongoUtils:
    def __init__(self, hostname, database, collection):
        '''
        Constructor method for class
        :param hostname: Hostname for MongoDB server
        :param database: MongoDB database name
        :param collection: MongoDB collection name
        '''
        self.hostname =  hostname
        self.database = database
        self.collection = collection

    def get_mongodb_client(self):
        """
        Initialize mongodb client
        :return:
            MongoDB client object with given server details
        """
        return MongoClient(self.hostname)

    def get_mongodb_collection(self, client):
        """
        Returns MongoDB collection object for specified collection
        :param client: MongoDB clien
        :return:
            collection object for specific db and collection
        """
        db = client.get_database(self.database)
        return db.get_collection(self.collection)

if __name__ == '__main__':
    sys.exit(0)
    pass
