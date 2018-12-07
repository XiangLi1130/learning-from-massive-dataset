import pandas as pd
from pandas import DataFrame
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import csv
import random as random
from itertools import combinations

#all constant 
R = 199933
b_ = 100
r_ = 10
m_ = b_ * r_
large_prime = 90001567

#Q1
def Q1():
	file = open("Netflix_data.txt", "r") 
	movie_size = 0
	movie_ID = 0
	# build dictionary
	d = {}
	for line in file: 
	    if ':' in line:
	        movie_id = line[:-2]
	        movie_size = movie_size + 1
	    else:
	        info = line.strip().split(",")
	        userID = info[0]
	        movieID = int (movie_id)
	        rating = int (info[1])
	        if int (info[1]) > 2:
	            d.setdefault(userID, []).append(movieID)
	    # get user size
	user_size = 0
	for k in d:
	    if len(d[k]) <= 20:
	        user_size = user_size + 1
	    # build matrix
	matrix = np.zeros([movie_size,user_size]);
	index = 0
	for k in d:
	    if len(d[k]) <= 20:
	        for movie_id in d[k]:
	            matrix[movie_id - 1][index] = 1
	        index = index + 1
	return matrix, movie_size, user_size





#Q2
def Q2(user_size, movie_size, matrix):
	distances = []
	for i in range(0,10000):
	    user1_index = random.randint(0,user_size -1)
	    user2_index = random.randint(0,user_size -1)
	    user1 = matrix[:,user1_index]
	    user2 = matrix[:,user2_index]
	    intersection = 0
	    union = 0
	    for j in range (0,movie_size) :
	        intersection = intersection + (int(user1[j]) & int(user2[j]))
	        union = union + (int(user1[j]) | int(user2[j]))
	    distance = 1 - (float(intersection) / float(union))
	    distances.append(distance)
	average = np.average(distances)
	print("average distance is: ", average)
	distances.sort()
	print("the lowest distance is: " , distances[0])
	#draw graph
	plt.xlim(0.35,1)
	plt.hist(distances, bins=100)



#Q3
def Q3(matrix):
    matrix1 = csc_matrix(matrix)
    return matrix1


#Q4
def getAllValidCandidate(matrix1) :
	randomMat = np.random.randint(low=1, high=movie_size, size=(m_, 2))

	for user in range(0, user_size):
	    oneUserInfo = np.array(matrix1[:, user].indices)
	    #ones = np.ones([1,len(oneUserInfo)])
	    ones = np.array(len(oneUserInfo) * [1])
	    x_matrix = np.vstack((oneUserInfo, ones))
	    # (ax+b)mod R
	    oneUserSig = np.dot(randomMat, x_matrix) % R
	    minSig = list(oneUserSig.min(axis=1))
	    if user == 0:
	        signatureMat = minSig.copy()
	    else:
	        signatureMat.extend(minSig)
	        
	signatureMat = np.transpose(np.reshape(signatureMat, (user_size, m_)))
	print("the signature matrix is: ")
	print(signatureMat)
	for b in range (0,b_):
    #generate ai and bi
	    hashMatrix = np.random.randint(low = 1, high = large_prime, size = (r_,2))
	    oneBandMatrix = np.array(signatureMat[b * r_:(b + 1) * r_, :])
	    ai = np.reshape(hashMatrix[:,0],(r_,1))
	    bi = np.reshape(hashMatrix[:,1],(r_,1))
	    oneBandMatrix = np.add(ai*oneBandMatrix,bi) % large_prime
	    sumMatrix = oneBandMatrix.sum(axis = 0)
	    #find out the users in same bucket
	    sumList = sumMatrix.tolist()
	    index = np.arange(1, len(sumList) + 1)
	    #linked user index with its hash value
	    userWithIndex = list(zip(sumList,index))
	    userWithIndex.sort()
	    i = 0
	    pair = set()
	    #the list of users in the same bucket
	    same = []
	    while (i < len(sumList) - 2) :
	        same.clear()
	        while(userWithIndex[i][0] == userWithIndex[i+1][0] and i < len(sumList) - 2):
	            same.append(userWithIndex[i][1])
	            i = i + 1
	        same.append(userWithIndex[i][1])
	        pair.update(set(combinations(same, 2)))
	        i = i + 1
	    if b == 0:
	        allPairs = pair
	    else :
	        allPairs.update(pair)
    #print("all pair number: " + str(len(allPairs)))
	return randomMat, signatureMat, allPairs

def Q4(matrix):
	randomMat, signatureMat, allPairs = getAllValidCandidate(matrix)
	validPairs = set()
	for pair in allPairs:
		user1 = matrix[:, pair[0] - 1].indices
		user2 = matrix[:, pair[1] - 1].indices
		#since there are too many pairs, using  Compressed Sparse Column format to calculate
		#is much more efficient than compare each row in sparse matrix
		intersection = len(np.intersect1d(user1, user2))
		union = len(np.union1d(user1,user2))
		distance = 1 - float(intersection)/ float(union)
		#print(distance)
		if distance < 0.35:
			validPairs.add(pair)
	print("pair numbers with distance less than 0.35: " + str(len(validPairs)))
	return randomMat, signatureMat, validPairs

def writeToFile(similarPairs) :
    with open('similarPair_project.csv','w') as writeFile:
        similarWriter = csv.writer(writeFile, delimiter=',')
        for pair in similarPairs:
            similarWriter.writerow([pair[0], pair[1]])

#Q5
#generate movie_size*1 matrix for queried user
def generateMatrix(movies):
    arr = np.zeros([movie_size,1])
    for m in movies:
        arr[m - 1][0] = 1
    return arr

#generate signature column for queried user
def generateSig(arr,randomMat) :
    arr_compress = csc_matrix(arr)
    likedMovie = np.array(arr_compress[:, 0].indices)
    #print(likedMovie)
    ones = np.array(len(likedMovie) * [1])
    matForHashing = np.vstack((likedMovie, ones))
    sigCol = np.dot(randomMat, matForHashing)
    sigCol = sigCol % R
    sigCol_ret = np.reshape(sigCol.min(axis=1), (m_, 1))
    return sigCol_ret, arr_compress

def getCandidateUser(sigCol,sigMat,arr_compress,arr):
    wholeSig = np.hstack((sigCol,sigMat))
    row_ = r_
    band_ = b_
    m = m_
    candidateUser = set()
    for i in range(0,band_):
        hashMat = np.random.randint(low = 1, high = large_prime, size = (row_,2))
        oneBandMat = np.array(wholeSig[i * row_:(i + 1) * row_, :])
        ai = np.reshape(hashMat[:,0],(row_,1))
        bi = np.reshape(hashMat[:,1],(row_,1))
        oneBandHash = np.add(ai*oneBandMat,bi) % large_prime
        sumMat = oneBandHash.sum(axis = 0)
        sumList = sumMat.tolist()
        targetSum = sumList[0]
        for j in range (1, len(sumList)):
            if (sumList[j] == targetSum):
                candidateUser.add(j)
    return candidateUser

def findNearest (movie_size, matrix, arr, candidateUser):
    nearestUsers = set()
    for c in candidateUser:
        user1 = matrix[:,c-1]
        q_user = arr[:,0]
        intersection = 0
        union = 0
        for k in range(0,movie_size):
            intersection = intersection + (int(q_user[k]) & int(user1[k]))
            union = union + (int(q_user[k]) | int(user1[k]))
        distance = 1 - float(intersection)/ float(union)
        if (distance < 0.35):
            nearestUsers.add(c)
    return nearestUsers

def findOneNearestNeighbor (sigCol,sigMat,arr_compress,arr):
    wholeSig = np.hstack((sigCol,sigMat))
    rowInOneBand = 10
    validCandidate = 10000
    nearestUser = -1
    validUsers = set()
    low = 5
    high = 20
    while validCandidate > 200 or validCandidate == 0:
        rowInOneBand = int ((low+high)/2)
        bandRow = int (m_/rowInOneBand)
        validCandidate = 0
        validUsers.clear()
        for b1 in range (0,bandRow):
            hashMat = np.random.randint(low = 1, high = large_prime, size = (rowInOneBand,2))
            oneBandMat = np.array(wholeSig[b1 * rowInOneBand:(b1 + 1) * rowInOneBand, :])
            ai = np.reshape(hashMat[:,0],(rowInOneBand,1))
            bi = np.reshape(hashMat[:,1],(rowInOneBand,1))
            oneBandHash = np.add(ai*oneBandMat,bi) % large_prime
            sumMat = oneBandHash.sum(axis = 0)
            for i in range (1,len(sumMat) - 1):
                if sumMat[i] == sumMat[0]:
                    validCandidate += 1
                    validUsers.add(i)
        if(validCandidate == 0):
        	if(low == high):
        	    low = low - 2
        	else:
        		high = rowInOneBand
        elif(validCandidate > 200):
        	if (low == high):
        	    high = high + 2
        	else:
        		low = rowInOneBand
        #it is hard to find out only one validCandidate, so I set the threshold to be 100, then if we find less than 100 
        #candidate, the nearest one is the user we want
        else :
            min_dis = 1
            queried_user = arr_compress[:,0].indices
            q_user = arr[:,0]
            for user in validUsers:
                user2 = matrix[:,user - 1]
                intersection = 0
                union = 0
                for k in range(0,movie_size):
                    intersection = intersection + (int(q_user[k]) & int(user2[k]))
                    union = union + (int(q_user[k]) | int(user2[k]))
                distance = 1 - float(intersection) / float(union)
                if(distance < min_dis):
                    nearestUser = i
                    min_dis = distance
    #all users have Jaccard distance 1
    if (nearestUser == -1) :
        nearestUser = 1
    return nearestUser

def Q5(movies,randomMat,signatureMat) :
    arr_q5 = generateMatrix(movies)
    sigCol_ret, arr_compress = generateSig(arr_q5,randomMat)
    candidateUser = getCandidateUser(sigCol_ret, signatureMat, arr_compress, arr_q5)
    nearestUsers = findNearest(movie_size, matrix, arr_q5, candidateUser)
    print("the nearest users are (the users with Jaccard distance less than 0.35): \n")
    print(nearestUsers)
    print("note:the first user number we use is 1 not 0\n")
    if (len(nearestUsers) == 0):
    	print("no users are within Jaccard distance 0.35 of the queried user, now finding the user closest to queried user\n")
    	nearestOne = findOneNearestNeighbor(sigCol_ret, signatureMat, arr_compress,arr_q5)
    	print("the nearest one to queried user is: " + str(nearestOne))
    return nearestUsers

def getInput(randomMat,signatureMat) :
    enter_more = True
    while (enter_more) :
        print("please input the movie list in the format of movie1, movie2,......")
        print("for example:")
        print("1,3,864,234")
        input_movie = raw_input("input queried user now.\n")
        print("what you input is: {" + input_movie + "}")
        movies = input_movie.split(",")
        movie_list = {}
        for m in movies:
        	movie_list.add(int(m))
        Q5(movie_list,randomMat,signatureMat)
        again = input("do you want to input another queried user?\ny for yes, n for no")
        enter_more = again.lower().startswith('y')



if __name__ == "__main__":
	matrix, movie_size, user_size = Q1()
	print("matrix for Q1 is: ")
	print (matrix)
	Q2(user_size,movie_size,matrix)
	compress_sparse_matrix = Q3(matrix)
	randomMat, signatureMat, validPairs = Q4(compress_sparse_matrix )
	print("writing all pairs with distance less than 0.35 to files......")
	writeToFile(validPairs)
	print("done writing!")
	movies = {1307}
	print("here is an example input, queried user with liked movie 1307, calculating nearest users......")
	#movies = {1,3,2,5,7,20,50,354,15,80,3000,69,919}
	Q5(movies,randomMat,signatureMat)





