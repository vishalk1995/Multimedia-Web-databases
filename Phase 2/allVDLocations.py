import pandas, numpy
from scipy import spatial
from utils import mongoUtils, configReader, dimentionalityUtils

class allVDLocations:
    def __init__(self, method, location_id, latent_semantics):
        '''
        Constructor method for the class
        :param method: Method for demensionality reductions
        :param location_id: Query location ID
        '''
        self.method = method
        self.location_id = location_id
        self.latent_semantics = latent_semantics

    def getLocationName(self):
        '''
        Returns location name respective to entered location ID
        :return:
            Location name
        '''
        config = configReader()
        mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                        config.location_info_collection)
        client = mu.get_mongodb_client()
        collection = mu.get_mongodb_collection(client)
        location_name = collection.distinct("title",
                                            {"number": int(self.location_id)})
        client.close()
        return location_name[0]

    def getLocationList(self):
        config = configReader()
        mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                        config.location_vd_collection)
        client = mu.get_mongodb_client()
        collection = mu.get_mongodb_collection(client)
        results = collection.aggregate([{"$match": {}},
                                        {"$project": {
                                            "location": "$location",
                                        }}])
        data = pandas.DataFrame(list(results))
        return data['location']


    def getDataFrame(self, image, image_score):
        '''
        Returns pandas dataframe for images and their visual discriptor model
        :param image: List of Image IDs
        :param image_score: List of visual descriptor score
        :return:
            Pandas dataframe
        '''
        image_score = numpy.array(image_score).reshape(len(image_score),
                                                       len(image_score[0]))
        return pandas.DataFrame(image_score, columns=range(len(image_score[0])),
                                index=image)

    def getQueryLocationData(self, location_name, vector_model):
        '''
        Get visual descriptor data for given location
        :return:
            Matrix of image scores for selected visual model of given location
        '''
        config = configReader()
        mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                        config.location_vd_collection)
        client = mu.get_mongodb_client()
        collection = mu.get_mongodb_collection(client)
        result = collection.aggregate([{"$match": {"location": location_name}},
                                       {"$project": {
                                           "_id": "$_id",
                                           "location": "$location",
                                           "images": "$" + vector_model + ".image",
                                           "scores": "$" + vector_model + ".scores"
                                       }}])
        data = pandas.DataFrame(list(result))
        client.close()
        query_image_list = data['images']
        query_scores_list = data['scores']
        image_score_matrix = pandas.DataFrame()
        for index in range(len(query_image_list)):
            image_score_matrix = image_score_matrix.append(
                self.getDataFrame(query_image_list[index], query_scores_list[index]))
        return image_score_matrix, query_image_list

    def getSimilarLocations(self):
        model_similarity = {}
        cos_location = []
        store = {}
        mat = {}
        location_name = self.getLocationName()
        model = ['CM', 'CN', 'CM3x3', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
        all_location_list = self.getLocationList()
        for i in model:
            image_score_matrix, query_image_list = self.getQueryLocationData(location_name, i)
            du = dimentionalityUtils(self.method, image_score_matrix, self.latent_semantics)
            u, s, vt = du.apply_transformation()
            new_feature_mat = image_score_matrix.dot(vt.T)
            for n in range(len(vt)):
                print("\nLatent Semantics", n + 1, "using Choice", self.method, "for Model", i, "are: ",
                      end=":")  # Displaying the Latent Semantics of Query location
                for score_index in range(len(vt[n])):
                    if score_index == len(vt[n]) - 1:
                        print(vt[n][score_index])
                    else:
                        print(vt[n][score_index], end=',')
                print()
            store[i] = vt
            mat[i] = new_feature_mat  # Storing the new feature matrix and all models score for further computation
        for r in range(len(all_location_list)):
            for i in model:
                other_image_score_matrix, image_list = self.getQueryLocationData(all_location_list[r], i)
                related_location_scores = numpy.matmul(other_image_score_matrix, numpy.transpose(
                    store[i]))  # generating Images of the location x Latent Semantics matrix
                location_scores_df = pandas.DataFrame(related_location_scores, index=other_image_score_matrix.axes[0])
                location_scores_df = location_scores_df[~location_scores_df.index.duplicated(keep='first')]
                cos = []
                mat[i]
                for e in image_list[0]:
                    current_image = list(location_scores_df.loc[e].values)
                    for j in mat[i].values:
                        cos.append(spatial.distance.euclidean(current_image, j.tolist()))  # Euclidean Distance Calculation
                cos_location.append(numpy.mean(cos))
            model_similarity[all_location_list[r]] = numpy.mean(cos_location)
        count = 0
        print("Most similar 5 locations are:")
        for location in sorted(model_similarity, key=model_similarity.get):
            print("Location", location, "with similarity score",
                  model_similarity[location])
            count += 1
            if count == 5:
                break
