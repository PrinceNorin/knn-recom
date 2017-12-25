from numpy import *
from matplotlib import pyplot
import operator

def load_data(filename):
    f = open(filename)
    lines = f.readlines()
    size = len(lines)
    
    labels = []
    m_feats = zeros((size, 2))
    label_dict = {'not_get_along': 0, 'might_get_along': 1, 'get_along': 2}
    
    for i, line in enumerate(lines):
        line = line.strip()
        parts = line.split('\t')
        m_feats[i, :] = parts[0:2]
        labels.append(label_dict[parts[-1]])
        
    return m_feats, labels

def auto_norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    
    n = dataset.shape[0]
    n_dataset = dataset - tile(min_vals, (n, 1))
    n_dataset = n_dataset / tile(ranges, (n, 1))
    
    return n_dataset, ranges, min_vals

def classify(in_a, dataset, labels, k):
    n = dataset.shape[0]
    diff_m = tile(in_a, (n, 1)) - dataset
    sq_diff_m = diff_m ** 2
    sq_distance = sq_diff_m.sum(axis=1)
    distance = sq_distance ** 0.5
    sorted_indices = distance.argsort()
    
    label_count = {}
    for i in range(k):
        vote_label = labels[sorted_indices[i]]
        label_count[vote_label] = label_count.get(vote_label, 1) + 1
        
    sorted_label_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_count[0][0]

def test_classify():
    feats, labels = load_data('dataset.txt')
    n_feats, ranges, min_vals = auto_norm(feats)
    n = n_feats.shape[0]
    
    err_count = 0.0
    test_size = int(n * 0.1) # 10% test data
    for i in range(test_size):
        result = classify(n_feats[i, :], n_feats[test_size:n, :], labels[test_size:n], 20)
        if result != labels[i]:
            err_count += 1.0
            
    return err_count / float(test_size)

def get_input(message):
    return float(input(message))

def classify_person():
    display_text = ["don't get along", 'might get along', 'do get along']

    gaming_hours = get_input('Hours spent on MMORPG per week: ')
    animes_num = get_input('Number of anime with romance/comedy/action that watched so far: ')

    feats, labels = load_data('dataset.txt')
    n_feats, ranges, min_vals = auto_norm(feats)

    in_a = array([animes_num, gaming_hours])
    norm_input = (in_a - min_vals) / ranges
    k = int(n_feats.shape[0] * 0.05) # Taking 5% Nearest Neighbors

    result = classify(norm_input, n_feats, labels, k)
    return display_text[result]

print("You probably %s with this person!" % classify_person())

# print("error rate: %f" % test_classify())

# m_feats, labels = load_data('dataset.txt')
# colors = array(labels) * 12.5 # random color

# fig = pyplot.figure()
# ax = fig.add_subplot(111)
# ax.scatter(m_feats[:, 0], m_feats[:, 1], c=colors)
# pyplot.show()
