import os, sys, webbrowser, random, time, argparse
import pandas, numpy
from scipy.spatial import distance

def visualize(clusters):
    '''
    Create and open HTML file for visualizing clusters
    :param clusters: Dictionry representing clusters
    '''
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'config.ini'))
    html_file = 'kmeans.html'
    path = config.get('visualization', 'image_path')
    if os.path.exists(html_file):
        print("removing old html file...")
        os.remove(html_file)
    f = open(html_file, "w")
    header = open('htmlHeader.txt','r')
    f.write(header.read())
    f.write("<div align=\"center\"> <h1><u>Image Clustering using k-means algorithm. </u></h1></div>\n")
    f.write("<div class=\"row\">\n")
    for index in clusters.keys():
        f.write("</div>\n")
        f.write("<br>\n<div align=\"center\"> <h2><u> Cluster "+ str(index + 1) +": </u></h1></div>\n")
        f.write("<div class=\"row\">\n")
        i = 0
        for imageId in clusters[index]:
            if i % 4 == 0:
                f.write("</div>\n\n<div class=\"row\"> \n")
                i = 0
            f.write("\t<div> \n\t\t<figure class=\"thumbnail\">\n\t\t\t<img src=\"" + path + imageId + ".jpg\" alt=\"Image Id - " + imageId + "\"/><figcaption> Image Id - " + imageId + "</figcaption>\n\t\t</figure> \n\t</div>\n")
            i = i+1
    f.write("</div>\n</body>")
    webbrowser.open('file://' + os.path.realpath(html_file))

def k_means(num_clusters, graph_file):
    '''
    K-means clustering algorithm
    :param num_clusters: Number of clusters
    :param graph_file: Path to graph file
    '''
    start = time.time()
    image_graph = pandas.read_json(graph_file, convert_axes=False)
    image_graph = image_graph.fillna(0)
    image_list = image_graph.columns

    print('Graph read in', time.time() - start, 'seconds')
    read = time.time()

    # Create Adjacency matrix for converting graph to vector space
    adjacency_matrix = pandas.DataFrame(index=image_list, columns=image_list)
    for image in image_list:
        outgoing = image_graph.index[image_graph[image] != 0].tolist()
        for node in outgoing:
            weight = image_graph[image].loc[node]
            adjacency_matrix.loc[image].loc[node] = weight

    similarity_matrix = adjacency_matrix.fillna(0)

    print('Adjacency matrix created in', time.time() - read, 'seconds')
    km1 = time.time()

    image_list = similarity_matrix.index

    initial_centroids = []
    centroids = {}

    random_point = image_list[random.randrange(len(image_list))]

    dist_mat = distance.squareform(distance.pdist(similarity_matrix.values))
    dist_mat = pandas.DataFrame(dist_mat, index=image_list, columns=image_list)
    max_index = dist_mat[random_point].idxmax()
    initial_centroids.append(max_index)

    point1 = initial_centroids[0]
    max_index = dist_mat[point1].idxmax()
    initial_centroids.append(max_index)

    for index in range(2, num_clusters):
        max_dist_sum = 0
        for candidate in image_list:
            dist_sum = 0
            for image in initial_centroids:
                dist_sum += dist_mat[image][candidate]
            if dist_sum > max_dist_sum:
                max_index = candidate
                max_dist_sum = dist_sum
        initial_centroids.append(max_index)

    print('Initial centroids', initial_centroids)

    for index in range(len(initial_centroids)):
        centroids[index] = similarity_matrix.loc[initial_centroids[index]].values

    del(dist_mat)

    def knn_eucl(image_list, centroids):
        '''
        Function to assign images to clusters represented by centroids
        :param image_list: List of image IDs
        :param centroids: Dictionary of vectors for current centroid
        :return: Dictionary representing image clusters
        '''
        image_clusters = {}
        centroids_matrix = pandas.DataFrame(centroids).T
        dist_mat = distance.cdist(similarity_matrix.values, centroids_matrix)
        dist_mat = pandas.DataFrame(dist_mat, index=image_list, columns=centroids.keys())
        for image in image_list:
            min_index = dist_mat.loc[image].idxmin()
            if min_index in image_clusters.keys():
                image_clusters[min_index].append(image)
            else:
                image_clusters[min_index] = [image]
        return image_clusters

    converged = False
    while not converged:
        converged = True
        image_clusters = knn_eucl(image_list, centroids)
        centroid_count = 0
        new_centroids = {}
        for centroid, cluster in image_clusters.items():
            current_similarity = similarity_matrix.loc[cluster]
            new_centroids[centroid_count] = current_similarity.mean().values
            centroid_count += 1
        for index in (centroids.keys() & new_centroids.keys()):
            if not numpy.allclose(centroids[index], new_centroids[index]):
                converged = False
                break
        centroids=new_centroids

    print("="*30)
    print("Clusters are:")
    for index in image_clusters.keys():
        print("Cluster: ", index+1,"Images count: ", len(image_clusters[index]))
        print(image_clusters[index])

    visualize(image_clusters)
    print('Execution completed in', time.time() - km1, 'seconds')

def readCommand(argv):
    '''
    Function to parse command line arguments
    :param argv: command line arguments
    :return: Parser object containing all arguments
    '''
    parser = argparse.ArgumentParser(description=('Find clusters in a graph'
                                                  + ' using k-means algorithm'))
    parser.add_argument('-c', '--clusters', type=int, required=True,
                        help='Number of clusters to be formed')
    parser.add_argument('-g', '--graph-file', type=str, required=True,
                        help='Path to the graph file in json format')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = readCommand(sys.argv[1:])
    num_clusters = args.clusters
    graph_file = args.graph_file
    if num_clusters < 2:
        raise Exception("Number of clusters should be greater than or equal to 2")
    k_means(num_clusters, graph_file)
    sys.exit(0)
