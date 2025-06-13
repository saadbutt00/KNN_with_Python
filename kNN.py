import numpy as np
import collections

num_feature = int(input('Enter Number of Features - '))
k = int(input('Select K - '))
print('')

X = []

for b in range(num_feature):
    a = list(map(float, input(f'Enter feature X{b} values (space-separated): ').split()))
    X.append(a)

Y = input('Enter feature Y values (space-separated): ').split()

print('\nDistance Metrics:')
print('1. Euclidean Distance')
print('2. Manhattan Distance')
print('3. Minkowski Distance')
print('')
dist_opt = int(input('Select Distance Metric (1 or 2 or 3): '))
print('')
updated_X = []

for x in X:
    int_arr = []
    str_arr = []
    
    for y in x:
        try:
            num = float(y)
            int_arr.append(num)
        except:
            str_arr.append(y)

    if int_arr:
        updated_X.append(int_arr)
    if str_arr:
        updated_X.append(str_arr)

updated_X = np.transpose(X).tolist()

user_sample = list(map(float, input('Enter your Sample (space-separated): ').split()))
print('')
if len(user_sample) != num_feature:
    print("Error: Sample size must match number of features.")
    exit()

def euclidean_dist(x1, x2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

def manhattan_dist(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

def minkowski_dist(x1, x2, p):
    return (sum(abs(a - b) ** p for a, b in zip(x1, x2))) ** (1 / p)

distance = []

for xi, yi in zip(updated_X, Y):
    if dist_opt == 1:
        dist = euclidean_dist(user_sample, xi)
    elif dist_opt == 2:
        dist = manhattan_dist(user_sample, xi)
    elif dist_opt == 3:
        p = float(input('Enter p value for Minkowski Distance - '))
        dist = minkowski_dist(user_sample, xi, p)
    else:
        print('Error - Invalid option')
        exit()
    distance.append((dist, yi))

sorted_neighbors = sorted(distance, key=lambda x: x[0])
k_neighbors = sorted_neighbors[:k]

labels = []
for _, label in k_neighbors:
    labels.append(label)

pred = collections.Counter(labels).most_common(1)[0][0]

print('Prediction -', pred)