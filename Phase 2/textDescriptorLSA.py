import sys, argparse, time
import pandas, numpy, operator
from scipy import spatial
from utils import mongoUtils, configReader, dimentionalityUtils

class textDescriptorLSA:
    def __init__(self, method, vector_space):
        '''
        Constructor method for the class
        :param method: Method for dimensionality reduction
        :param vector_space: Vector space on which reduction is to be applied
        '''
        self.method = method
        self.vector_space = vector_space

    def getLocationName(self, location_id):
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
                                            {"number": int(location_id)})
        client.close()
        return location_name[0]

    def get_object_data(self):
        '''
        Returns matrix based on text descriptor for selected vector space
        :return:
            Pandas dataframe as matrix
            list of all terms
        '''
        config = configReader()
        if self.vector_space == 'user':
            mongo_collection = config.user_td_collection
            id_label = "USER_ID"
        elif self.vector_space == 'image':
            mongo_collection = config.image_td_collection
            id_label = "IMAGE_ID"
        elif self.vector_space == 'location':
            mongo_collection = config.location_td_collection
            id_label = "LOCATION_ID"
        mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                        config.user_td_collection)
        client = mu.get_mongodb_client()
        collection = mu.get_mongodb_collection(client)
        collection_data1 = collection.find({})
        collection_data_copy1 = collection.find({})
        client.close()
        all_task_term1 = {}
        all_task_idf1 =  {}
        id_label = 'USER_ID'
        for objects in collection_data1:
            word_dataframe = pandas.DataFrame(list(objects["TEXT_DESC"]))
            individual_word_list = word_dataframe["TERM"].values
            individual_idf_list = word_dataframe["TF_IDF"].values
            all_task_term1[objects[id_label]] = []
            all_task_term1[objects[id_label]].extend(individual_word_list)
            all_task_idf1[objects[id_label]] = []
            all_task_idf1[objects[id_label]].extend(individual_idf_list)
            del(individual_word_list)
            del(individual_idf_list)
            del(word_dataframe)

        dataframe1 = pandas.DataFrame(list(collection_data_copy1))
        all_terms_set = set()
        for objects in dataframe1["TEXT_DESC"]:
            all_terms_set = all_terms_set.union(set(pandas.DataFrame(objects)["TERM"].values))


        mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                        config.image_td_collection)
        client = mu.get_mongodb_client()
        collection = mu.get_mongodb_collection(client)
        collection_data2 = collection.find({})
        collection_data_copy2 = collection.find({})
        client.close()
        all_task_term2 = {}
        all_task_idf2 =  {}
        id_label = "IMAGE_ID"
        for objects in collection_data2:
            word_dataframe = pandas.DataFrame(list(objects["TEXT_DESC"]))
            individual_word_list = word_dataframe["TERM"].values
            individual_idf_list = word_dataframe["TF_IDF"].values
            all_task_term2[objects[id_label]] = []
            all_task_term2[objects[id_label]].extend(individual_word_list)
            all_task_idf2[objects[id_label]] = []
            all_task_idf2[objects[id_label]].extend(individual_idf_list)
            del(individual_word_list)
            del(individual_idf_list)
            del(word_dataframe)

        dataframe2 = pandas.DataFrame(list(collection_data_copy2))
        #all_terms_set = set()
        for objects in dataframe2["TEXT_DESC"]:
            all_terms_set = all_terms_set.union(set(pandas.DataFrame(objects)["TERM"].values))



        mu = mongoUtils(config.mongo_hostname, config.mongo_database,
                        config.location_td_collection)
        client = mu.get_mongodb_client()
        collection = mu.get_mongodb_collection(client)
        collection_data3 = collection.find({})
        collection_data_copy3 = collection.find({})
        client.close()
        all_task_term3 = {}
        all_task_idf3 =  {}
        id_label = "LOCATION_ID"
        for objects in collection_data3:
            word_dataframe = pandas.DataFrame(list(objects["TEXT_DESC"]))
            individual_word_list = word_dataframe["TERM"].values
            individual_idf_list = word_dataframe["TF_IDF"].values
            all_task_term3[objects[id_label]] = []
            all_task_term3[objects[id_label]].extend(individual_word_list)
            all_task_idf3[objects[id_label]] = []
            all_task_idf3[objects[id_label]].extend(individual_idf_list)
            del(individual_word_list)
            del(individual_idf_list)
            del(word_dataframe)

        dataframe3 = pandas.DataFrame(list(collection_data_copy3))
        #all_terms_set = set()
        for objects in dataframe3["TEXT_DESC"]:
            all_terms_set = all_terms_set.union(set(pandas.DataFrame(objects)["TERM"].values))

        if self.vector_space == 'user':
            id_label = 'USER_ID'
            all_object = list(dataframe1[id_label].values)
            all_task_idf = all_task_idf1
            all_task_term = all_task_term1
        elif self.vector_space == 'image':
            id_label = 'IMAGE_ID'
            all_object = list(dataframe2[id_label].values)
            all_task_idf = all_task_idf2
            all_task_term = all_task_term2
        elif self.vector_space == 'location':
            id_label = 'LOCATION_ID'
            all_object = list(dataframe3[id_label].values)
            all_task_idf = all_task_idf3
            all_task_term = all_task_term3

        all_terms = list(all_terms_set)
        # feature_object_matrix = []
        # for term in all_terms:
        #     feature_list = []
        #     for objects in all_object:
        #         if term in all_task_term[objects]:
        #             index_for_object = all_task_term[objects].index(term)
        #             feature_list.append(all_task_idf[objects][index_for_object])
        #         elif term not in all_task_term[objects]:
        #             feature_list.append(0)
        #     feature_object_matrix.append(feature_list)
        #     del(feature_list)
        # feature_object_matrix = numpy.array(feature_object_matrix)
        # object_feature_matrix = pandas.DataFrame(
        #     feature_object_matrix.T, columns=all_terms, index=all_object)
        output_filename = self.vector_space + 'matrix.npy'
        # numpy.save(output_filename, object_feature_matrix)
        # print("done")
        # sys.exit()
        object_feature_matrix = numpy.load(output_filename)
        return object_feature_matrix, all_terms, all_object

    def computeSemantics(self, object_matrix, latent_semantics, all_terms):
        '''
        Performs latent semantic analysis on matrix
        :param object_matrix: Data matrix
        :param latent_semantics: Number of latent semantics
        :param all_terms: List of all terms
        '''
        du = dimentionalityUtils(self.method, object_matrix, latent_semantics)
        u, s, vt = du.apply_transformation()
        final_frame = pandas.DataFrame(vt.reshape(-1, len(vt)), index=all_terms)

        for i in range(0, latent_semantics):
            globals()['final_l%s' % i] = final_frame[[i]].copy()
            globals()['final_l%s' % i] = globals()['final_l%s' % i].sort_values(by=[i], ascending = False)

        print("=============OUTPUT START====================\n")
        if self.method == 'SVD' or self.method == 'PCA':
            for i in range(0, latent_semantics):
                print("\nFor Latent symentic "+str(i+1)+" with eigen Value "+str(s[i])+"\n")
                print(globals()['final_l%s' % i])
        elif self.method == 'LDA':
            for i in range(0, latent_semantics):
                print("\nFor Topic"+str(i+1))
                print(globals()['final_l%s' % i])
        print("\n=============OUTPUT END====================")

    def getSimilarObjects(self, object_matrix, latent_semantics, query_space_objects, query_object, query_space_terms):
        du = dimentionalityUtils(self.method, object_matrix, latent_semantics)
        u, s, vt = du.apply_transformation()
        if self.vector_space == 'user':
            vector_space1 = 'location'
            vector_space2 = 'image'
            pass
        elif self.vector_space == 'image':
            vector_space1 = 'location'
            vector_space2 = 'user'
            pass
        elif self.vector_space == 'location':
            vector_space1 = 'user'
            vector_space2 = 'image'
            pass


        space1_lsa = textDescriptorLSA(self.method, vector_space1)
        space1_matrix, space1_terms, space1_objects = space1_lsa.get_object_data()
        space2_lsa = textDescriptorLSA(self.method, vector_space2)
        space2_matrix, space2_terms, space2_objects = space2_lsa.get_object_data()

        generated_original_matrix1 = object_matrix.dot(vt.T)
        generated_original_matrix1 = pandas.DataFrame(generated_original_matrix1, index=query_space_objects)
        generated_original_matrix1 = generated_original_matrix1[
            ~generated_original_matrix1.index.duplicated(keep='first')]

        generated_original_matrix2 = space1_matrix.dot(vt.T)
        generated_original_matrix2 = pandas.DataFrame(generated_original_matrix2, index=space1_objects)
        generated_original_matrix2 = generated_original_matrix2[
            ~generated_original_matrix2.index.duplicated(keep='first')]

        generated_original_matrix3 = space2_matrix.dot(vt.T)
        generated_original_matrix3 = pandas.DataFrame(generated_original_matrix3, index=space2_objects)
        generated_original_matrix3 = generated_original_matrix3[
            ~generated_original_matrix3.index.duplicated(keep='first')]

        def printTop5Similar(object_type, similarity):
            count = 0
            print("Most similar 5", object_type+'s', "are:")
            for similar_object in sorted(similarity,
                                   key=similarity.get, reverse=True):
                print(object_type, similar_object, "with similarity score",
                      similarity[similar_object])
                count += 1
                if count == 5:
                    break

        query_space_distances = {}
        query_object = list(generated_original_matrix1.loc[query_object].values)
        for index in generated_original_matrix1.axes[0].tolist():
            current_object = list(generated_original_matrix1.loc[index].values)
            d1 = 1 - spatial.distance.cosine(current_object, query_object)
            query_space_distances[index] = d1
        printTop5Similar(self.vector_space, query_space_distances)
        print("============================================================================================================")

        space1_distances = {}
        generated_original_matrix2 = generated_original_matrix2[~generated_original_matrix2.index.duplicated(keep='first')]
        for index in generated_original_matrix2.axes[0].tolist():
            current_object = list(generated_original_matrix2.loc[index].values)
            d2 = 1 - spatial.distance.cosine(current_object, query_object)
            space1_distances[index] = d2
        printTop5Similar(vector_space1, space1_distances)
        print("============================================================================================================")
        space2_distances = {}
        generated_original_matrix3 = generated_original_matrix3[~generated_original_matrix3.index.duplicated(keep='first')]
        for index in generated_original_matrix3.axes[0].tolist():
            current_object = list(generated_original_matrix3.loc[index].values)
            d3 = 1 - spatial.distance.cosine(current_object, query_object)
            space2_distances[index] = d3
        printTop5Similar(vector_space2, space2_distances)


def readCommand(argv):
    '''
    Method to parse command line arguments
    :param argv: List of command line arguments
    :return:
        Parsed arguments
    '''
    parser = argparse.ArgumentParser(description=('Find latent semantics and'
                                                  + ' similar objects using'
                                                  + ' visual descriptors'))
    parser.add_argument('--task', type=int, required=True, choices=range(1,3),
                        help='Enter the number of task to be run')
    parser.add_argument('--method', type=str, required=True,
                        choices=['SVD', 'PCA', 'LDA'],
                        help='Enter method for finding latent semantics')
    parser.add_argument('--vector-space', type=str, required=True,
                        choices=['user', 'image', 'location'],
                        help='Enter vector space to be used for the task')
    parser.add_argument('-k', '--latent-semantics', type=int, required=True,
                        help='Number of latent semantics to be found')
    parser.add_argument('--object-id', type=str, required=False, default=None,
                        help='(Only required for task 2) Query object ID')
    return parser.parse_args(argv)

if __name__ == '__main__':
    start_time = time.time()
    args = readCommand(sys.argv[1:])
    task = args.task
    method = args.method
    vector_space = args.vector_space
    latent_semantics = args.latent_semantics
    object_id = args.object_id
    if task == 1:
        lsa = textDescriptorLSA(method, vector_space)
        object_term_matrix, all_terms, all_objects = lsa.get_object_data()
        lsa.computeSemantics(object_term_matrix, latent_semantics, all_terms)
    elif task == 2:
        if object_id is None:
            raise Exception("Missing query object id")
        lsa = textDescriptorLSA(method, vector_space)
        if vector_space == 'location':
            object_id = lsa.getLocationName(object_id)
            object_id = object_id.replace("_", " ")
        query_object_term_matrix, all_terms, all_query_space_objects = lsa.get_object_data()
        lsa.getSimilarObjects(query_object_term_matrix, latent_semantics, all_query_space_objects, object_id, all_terms)
    print("Execution completed in", time.time() - start_time, "seconds")
    sys.exit(0)
