# MIT 6.034 Lab 6: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    #stop case
    if id_tree.is_leaf():
        return id_tree.get_node_classification()
    child = id_tree.apply_classifier(point)
    return id_tree_classify_point(point,child) 



#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    dic = {}
    for point in data:
        result = classifier.classify(point)
        if result not in dic:
            dic[result] = []
        dic[result].append(point)
    return dic


#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    sample_size = len(data)
    types = {}
    s = set()
    for point in data:
        c = target_classifier.classify(point)
        s.add(c)
        if c not in types:
            types[c] = 0 
        types[c] += 1

    if len(s) == 1:
        return 0 
    
    disorder = 0
    for key in types.keys():
        disorder += - (float(types[key])/sample_size)*log2((float(types[key])/sample_size))
    return disorder

def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    branches = split_on_classifier(data, test_classifier)
    sample_size = len(data)
    total_disorder = 0
    for branch in branches:
        weight = len(branches[branch])/sample_size 
        branchdisorder = branch_disorder(branches[branch], target_classifier)
        total_disorder += weight * branchdisorder

    return total_disorder


## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab6.py:

# for classifier in tree_classifiers:
#     print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    
    min_disorder = float('inf')
    best_classifier = None
    for cur_classifier in possible_classifiers:
        cur_disorder = average_test_disorder(data, cur_classifier, target_classifier)
        if cur_disorder < min_disorder:
            best_classifier = cur_classifier
            min_disorder = cur_disorder
    if min_disorder == 1:
        raise NoGoodClassifiersError
    return best_classifier



## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node == None:
        id_tree_node = IdentificationTreeNode(target_classifier)
    
    if len(list(split_on_classifier(data, target_classifier).keys()))==1: 
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))
    else:
        try:
            classifier = find_best_classifier(data, possible_classifiers, target_classifier)
            features = split_on_classifier(data, classifier)
            id_tree_node = id_tree_node.set_classifier_and_expand(classifier, features)
            branches = id_tree_node.get_branches()
            for branch in branches:
                cur_branch = branches[branch]
                data = features[branch]
                construct_greedy_id_tree(data, possible_classifiers, target_classifier, cur_branch)
        except NoGoodClassifiersError:
            return id_tree_node    
    return id_tree_node





## To construct an ID tree for 2014 Q2, Part A:
# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))
# tree1 = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
# tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
# print(id_tree_classify_point(tree_test_point, tree_tree))

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
# print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
# print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))


#### Part 1E: Multiple choice ##################################################

ANSWER_1 = "bark_texture"
ANSWER_2 = "leaf_shape"
ANSWER_3 = "orange_foliage"

ANSWER_4 = [2,3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = 'No'
ANSWER_9 = 'No'


#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    dot_product = 0
    n = len(u)
    for i in range(n):
        dot_product += u[i]*v[i]
    return dot_product

def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    n = len(v)
    sum_sqr = 0
    for i in range(n):
        sum_sqr += v[i]**2
    length = math.sqrt(sum_sqr)
    return length

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    v1 = point1.coords
    v2 = point2.coords
    n = len(v1)
    u = []
    for i in range(n):
        u.append(v1[i] - v2[i])
    return norm(u)


def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    v1 = point1.coords
    v2 = point2.coords
    n = len(v1)
    distance = 0
    for i in range(n):
        distance += abs(v1[i] - v2[i])
    return distance

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    v1 = point1.coords
    v2 = point2.coords
    n = len(v1)
    distance = 0
    for i in range(n):
        if v1[i] != v2[i]:
            distance += 1
    return distance

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    v1 = point1.coords
    v2 = point2.coords
    distance = 1 - dot_product(v1,v2)/(norm(v1)*norm(v2))
    return distance
    


#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    dist_points = []
    
    for point2 in data:
        dist = distance_metric(point,point2)
        dist_points.append((dist, point2))

    sorted_lexi = sorted(dist_points, key=lambda tuple_: tuple_[1].coords)
    sorted_value = sorted(sorted_lexi, key=lambda tuple_: tuple_[0])

    k_nearest = []
    for i in range(k):
        k_nearest.append(sorted_value[i][1])
    return  k_nearest


def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    k_nearest = get_k_closest_points(point, data, k, distance_metric)
    k_classification = []
    for i in range(k):
        cur_point = k_nearest[i]
        cur_classification = cur_point.classification
        k_classification.append(cur_classification)
    classes = list(set(k_classification))

    mode = classes[0]
    max_count = 0
    for j in range(len(classes)):
        count = k_classification.count(classes[j])
        if count > max_count:
            mode = classes[j]
            max_count = count 
    return mode



## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    n = len(data)
    correct = 0
    for i in range(n):
        val = data[i]
        train = data[:i] + data[i+1:]
        pred = knn_classify_point(val, train, k, distance_metric)
        if val.classification == pred:
            correct += 1
    return correct/n


def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    distance_metrics = [euclidean_distance, manhattan_distance, hamming_distance, cosine_distance]
    ks = [1,2,3,4]
    accs = []
    for distance_metric in distance_metrics:
        for k in ks:
            acc = cross_validate(data, k, distance_metric)
            accs.append((k, acc, distance_metric))
    
    sorted_acc = sorted(accs, key=lambda tuple_: tuple_[0])
    best_params = (sorted_acc[-1][0], sorted_acc[-1][2])
    return best_params
    



## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = "Overfitting"
kNN_ANSWER_2 = "Underfitting"
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3


#### SURVEY ####################################################################

NAME = "Xiaohan Yang"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = "9-10 hrs"
WHAT_I_FOUND_INTERESTING = "I think part 1 is very useful for me to understand the process of building an ID tree. Moreover, part 2A also help me to distinguish the concept of k-nearest neighbors and decision tree. "
WHAT_I_FOUND_BORING = "Nothing."
SUGGESTIONS = ""
