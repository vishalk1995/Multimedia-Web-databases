import pandas, numpy
from scipy import spatial, mean
from utils import mongoUtils, configReader, dimentionalityUtils

class VDImages():
    def __init__(self, method, vector_model, image_id):
        '''
        Constructor method for class
        :param method: Method for dimensionality reduction
        :param vector_model: Visual descriptor model to be used
        :param image_id: Query image id
        '''
        self.method = method
        self.vector_model = vector_model
        self.image_id = image_id

    def getImageData(self):
        '''
        Fetches Image data from MongoDB for particular vector model
        :return: MongoDB cursor object
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
                                            "images": "$"+self.vector_model+'.image',
                                            "scores": "$"+self.vector_model+'.scores'
                                        }}])
        client.close()
        return results

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
        return pandas.DataFrame(image_score,
                                columns=range(len(image_score[0])),
                                index=image)

    def getImageModelMatrix(self):
        '''
        Create matrix based for data of image for particular visual descriptor
        :return:
            Image data matrix
            List of images
            List of locations
        '''
        results = self.getImageData()
        data = pandas.DataFrame(list(results))
        image_list = data['images']
        scores_list = data['scores']
        location_list = data['location']
        image_score_matrix = pandas.DataFrame()

        for index in range(len(image_list)):
            image_score_matrix = image_score_matrix.append(self.getDataFrame(
                image_list[index], scores_list[index]))
        return image_score_matrix, image_list, location_list

    def computeSemantics(self, image_matrix, image_list,
                         location_list, latent_semantics):
        '''
        Computes latent semantic analysis on the data matrix and print results
        :param image_matrix: The data matrix on which LSA is performed
        :param image_list: List of images
        :param location_list: List of locations
        :param latent_semantics: Number of latent semantics
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
        generated_original_matrix = image_matrix.values.dot(vt.T)
        LS_image_score_matrix = pandas.DataFrame(
            generated_original_matrix,index=image_matrix.axes[0].tolist())
        LS_image_score_matrix = LS_image_score_matrix[
            ~LS_image_score_matrix.index.duplicated(keep='first')]
        query_object = list(LS_image_score_matrix.loc[self.image_id].values)
        image_similarity = {}
        for index in image_matrix.axes[0].tolist():
            current_object = list(LS_image_score_matrix.loc[index].values)
            d =  1- spatial.distance.cosine(current_object, query_object)
            image_similarity[index] = d
        count = 0
        print("Most similar 5 images are:")
        for image in sorted(image_similarity,
                            key=image_similarity.get, reverse=True):
            print("Image", image, "with similarity score",
                  image_similarity[image])
            count += 1
            if count == 5:
                break

        location_similarity = {}
        for index in range(len(location_list)):
            distances = []
            for image in image_list[index]:
                current_object = list(LS_image_score_matrix.loc[image].values)
                d =  1- spatial.distance.cosine(current_object, query_object)
                distances.append(d)
            location_similarity[location_list[index]] = mean(distances)
        count = 0
        print("------------------------------------------------------")
        print("Most similar 5 locations are:")
        for location in sorted(location_similarity,
                               key=location_similarity.get, reverse=True):
            print("Location", location, "with similarity score",
                  location_similarity[location])
            count += 1
            if count == 5:
                break
