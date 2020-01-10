import sys, argparse, time
import VDImages, VDLocations, allVDLocations


def readCommand(argv):
    '''
    Method to parse command line arguments
    :param argv: List of command line arguments
    :return:
        Parsed arguments
    '''
    parser = argparse.ArgumentParser(description=('Find latent semantics and'
                                                  + ' similar objects using'
                                                  + ' text descriptors'))
    parser.add_argument('-k', '--latent-semantics', type=int, required=True,
                        help='Number of latent semantics to be found')
    parser.add_argument('--task', type=int, required=True, choices=range(3,6),
                        help='Enter the number of task to be run')
    parser.add_argument('--method', type=str, required=True,
                        choices=['SVD', 'PCA', 'LDA'],
                        help='Enter method for finding latent semantics')
    parser.add_argument('--vector-model', type= str, required=False, default=None,
                        choices=['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM',
                                 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3'],
                        help='(Optional) Enter vector model for the task')
    parser.add_argument('--object-id', type=str, required=True,
                        help='Enter query object')
    return parser.parse_args(argv)


if __name__ == '__main__':
    start_time = time.time()
    args = readCommand(sys.argv[1:])
    latent_semantics = args.latent_semantics
    task = args.task
    method = args.method
    vector_model = args.vector_model
    object_id = args.object_id
    if task == 3:
        if vector_model is None:
            raise Exception("Missing/Invalid visual descriptor model")
        lsa = VDImages.VDImages(method, vector_model, object_id)
        image_matrix, image_list, location_list = lsa.getImageModelMatrix()
        lsa.computeSemantics(image_matrix, image_list, location_list, latent_semantics)
    if task == 4:
        if vector_model is None:
            raise Exception("Missing/Invalid visual descriptor model")
        lsa = VDLocations.VDLocations(method, vector_model, object_id)
        image_matrix = lsa.getQueryLocationData()
        projected_data, vt = lsa.computeQueryLocationSemantics(image_matrix, latent_semantics)
        lsa.getSimilarLocations(projected_data, vt)
    if task == 5:
        las = allVDLocations.allVDLocations(method, object_id, latent_semantics)
        las.getSimilarLocations()
        pass
    print("Execution completed in", time.time() - start_time, "seconds")
    sys.exit(0)
