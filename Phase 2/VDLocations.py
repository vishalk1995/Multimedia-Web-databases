import pandas, numpy
from scipy import spatial
from utils import mongoUtils, configReader, dimentionalityUtils

class VDLocations():
    def __init__(self, method, vector_model, location_id):
        '''
        Constructor method for the class
        :param method: Method for demensionality reductions
        :param vector_model: Visual descriptor model
        :param location_id: Query location ID
        '''
        self.method = method
        self.vector_model = vector_model
        self.location_id = location_id

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

    def getQueryLocationData(self):
        '''
        Get visual descriptor data for given location
        :return:
            Matrix of image scores for selected visual model of given location
        '''
        location_name = self.getLocationName()
        config = configReader()
        mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                        config.location_vd_collection)
        client = mu.get_mongodb_client()
        collection = mu.get_mongodb_collection(client)
        result = collection.aggregate([{"$match": {"location":location_name}},
                                       {"$project":{
                                           "_id": "$_id",
                                           "location": "$location",
                                           "images": "$" + self.vector_model + ".image",
                                           "scores": "$" + self.vector_model + ".scores"
                                       }}])
        data = pandas.DataFrame(list(result))
        client.close()
        query_image_list = data['images']
        query_scores_list = data['scores']
        image_score_matrix = pandas.DataFrame()
        for index in range(len(query_image_list)):
            image_score_matrix = image_score_matrix.append(
                self.getDataFrame(query_image_list[index], query_scores_list[index]))
        return image_score_matrix

    def computeQueryLocationSemantics(self, image_matrix, latent_semantics):
        '''
        Perform LSA on location data
        :param image_matrix: Image data matrix
        :param latent_semantics: Number of latent semantics
        :return:
            Projected data and vt from the decomposition
        '''
        du = dimentionalityUtils(self.method, image_matrix, latent_semantics)
        u, s, vt = du.apply_transformation()
        print("Latent Semantics are:")
        for index in range(len(vt)):
            print("Latent Semantic", (index + 1), end=": ")
            for score_index in range(len(vt[index])):
                if score_index == len(vt[index]) - 1:
                    print(vt[index][score_index])
                else:
                    print(vt[index][score_index], end=", ")
            print()
        print("------------------------------------------------------")
        pojected_data = image_matrix.dot(vt.T)
        return pojected_data, vt

    def getSimilarLocations(self, projected_data, vt):
        '''
        Compute most similar locations based on vector model data
        :param projected_data: projected data from LSA
        :param vt: vt from LSA
        '''
        config = configReader()
        mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                        config.location_vd_collection)
        client = mu.get_mongodb_client()
        collection = mu.get_mongodb_collection(client)
        results = collection.aggregate([{"$match": {}},
                                        {"$project":{
                                            "_id": "$_id",
                                            "location": "$location",
                                            "images": "$" + self.vector_model + ".image",
                                            "scores": "$" + self.vector_model + ".scores"
                                        }}])
        data = pandas.DataFrame(list(results))
        image_list = data['images']
        scores_list = data['scores']
        location_list = data['location']
        other_image_score_matrix = pandas.DataFrame()
        for index in range(len(image_list)):
            other_image_score_matrix = other_image_score_matrix.append(
                self.getDataFrame(image_list[index], scores_list[index]))
        related_location_scores=numpy.matmul(other_image_score_matrix,
                                             numpy.transpose(vt))
        location_scores_df = pandas.DataFrame(
            related_location_scores,index=other_image_score_matrix.axes[0])
        location_scores_df = location_scores_df[
            ~location_scores_df.index.duplicated(keep='first')]
        location_similarity = {}
        for index in range(len(location_list)):
            cos=[]
            for image in image_list[index]:
                current_image = list(location_scores_df.loc[image].values)
                for query_location_image in projected_data.values.tolist():
                    cos.append(1-spatial.distance.cosine(
                        current_image, query_location_image))
            location_similarity[location_list[index]]=numpy.mean(cos)
        count = 0
        print("Most similar 5 locations are:")
        for location in sorted(location_similarity,
                               key=location_similarity.get, reverse=True):
            print("Location", location, "with similarity score",
                  location_similarity[location])
            count += 1
            if count == 5:
                break
