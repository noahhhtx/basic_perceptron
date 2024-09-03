# Written by Noah Harrison

import numpy as np

def sigmoid(s):
    return 1 / (1 + np.exp(0 - s))

def weight_update(w, c, j, o):
    return w + c * j * o

def j_output(t, o):
    return (t - o) * o * (1 - o)

def j_hidden(j_o, o, w):
    return (j_o * w) * o * (1 - o)

#default values
c = 1
target = 0
no_of_nodes = 999

weights = {}
inps = {}
output = []
hidden = []
biases = {}

s=""
while s != "done":
    s = input().lower()
    if s == "done":
        break
    arr = s.split(" ")
    if arr[0] == "input":
        arr[1] = int(arr[1])
        arr[2] = float(arr[2])
        inps[arr[1]] = arr[2]
    elif arr[0] == "weight":
        arr[1] = int(arr[1])
        arr[2] = int(arr[2])
        arr[3] = float(arr[3])
        if arr[1] not in weights:
            weights[arr[1]] = {}
        weights[arr[1]][arr[2]] = arr[3]
    elif arr[0] == "c" or arr[0] == "learningrate":
        c = int(arr[1])
    elif arr[0] == "target":
        target = int(arr[1])
    elif arr[0] == "hidden":
        hidden.append(int(arr[1]))
    elif arr[0] == "output":
        output.append(int(arr[1]))
        no_of_nodes = int(arr[1])
    elif arr[0] == "bias":
        arr[1] = int(arr[1])
        arr[2] = float(arr[2])
        biases[arr[1]] = arr[2]

for k in range(1,6):

    #calculate outputs

    outputs = {}
    
    for j in range(1, no_of_nodes+1):
        if j in hidden:
            temp = 0
            for node in weights:
                if node in inps and j in weights[node]:
                    temp += (inps[node] * weights[node][j])
                elif node in biases and j in weights[node]:
                    temp += (biases[node] * weights[node][j])
            temp = sigmoid(temp)
            outputs[j] = temp
        if j in output:
            temp = 0
            for node in weights:
                if node in hidden and j in weights[node]:
                    temp += (outputs[node] * weights[node][j])
                elif node in biases and j in weights[node]:
                    temp += (biases[node] * weights[node][j])
            temp = sigmoid(temp)
            outputs[j] = temp
    
    #update weights

    j_s = {}

    for i in range(no_of_nodes,0,-1):
        if i in output:
            j_s[i] = j_output(target, outputs[i])
        if i in hidden:
            j_s[i] = j_hidden(j_s[list(weights[i].keys())[0]], outputs[i], weights[i][list(weights[i].keys())[0]])

    for begin in weights:
        for end in weights[begin]:
            if begin in inps:
                weights[begin][end] = weight_update(weights[begin][end], c, j_s[end], inps[begin])
            if begin in biases:
                weights[begin][end] = weight_update(weights[begin][end], c, j_s[end], biases[begin])
            if begin in hidden:
                weights[begin][end] = weight_update(weights[begin][end], c, j_s[end], outputs[begin])

    #print weights

    print("Weights (After Iteration %s):" % k)
    for begin in weights:
        for end in weights[begin]:
            print(begin, end, ": ", weights[begin][end], sep="")

