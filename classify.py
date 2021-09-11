#Nicholas Stewart nss67
import os
import sys
import math
import random

def gradiantDescent(TrainingSet, stepSize, Epsilon, MAX_COST, Gvector, flag, restarts):
	prevCost = float('inf')
	Epsilon = float(Epsilon)
	prevAccuracy = calcAccuracy(Gvector, TrainingSet)
	iteration = 0
	while(True):
		print("iteration:", iteration, "Current Accuracy:", prevAccuracy)
		if iteration >= int(restarts):
			print("Maximum Iterations Reached, exiting...")
			print("------------------------------------------------------")
			return (Gvector,prevAccuracy)
		iteration += 1
		totalCost = 0.0
		gradiantDict = {}
		for v in TrainingSet.keys():
			gradiantDict[v] = 0
		for K, category in TrainingSet.items():
			for T in category:
				classifiedAS = findClosest(Gvector, T)
				if T not in TrainingSet[classifiedAS]:
					Cost = calcDistance(Gvector[K], T) - calcDistance(Gvector[classifiedAS], T)
					if Cost < float(MAX_COST):
						newGv = []
						newGw = []
						Gv = Gvector[K]
						Gw = Gvector[classifiedAS]
						for i in range(len(T)):
							diffGv = float(T[i]) - float(Gv[i])
							diffGw = float(Gw[i]) - float(T[i])
							newGv.append(diffGv)
							newGw.append(diffGw)
						newGvsum = sum(newGv)
						newGwsum = sum(newGw)
						gradiantDict[K] = newGvsum
						
						gradiantDict[classifiedAS] = newGwsum
						totalCost += Cost
					else:
						totalCost += float(MAX_COST)
		if flag:
			for K in sorted(Gvector.keys()):
				print("Exemplar", K, Gvector[K])	
		if totalCost < Epsilon:
			print("TRAINING DONE")
			print("------------------------------------------------------")
			return (Gvector,prevAccuracy)
		if totalCost > (1-Epsilon)*prevCost:
			print("Improvement Too Small, returning")
			print("------------------------------------------------------")
			return (Gvector,prevAccuracy)
		Hvector = {}
		for v in TrainingSet.keys():
			old = Gvector[v]
			gradiant = gradiantDict[v]
			new = []
			for i in range(len(old)):
				h = float(old[i]) + float(stepSize)*gradiant
				new.append(h)
			Hvector[v] = new					
		newAccuracy = calcAccuracy(Hvector, TrainingSet)
		if newAccuracy < prevAccuracy:
			print("Accuracy is worse, returning previous")
			print("------------------------------------------------------")
			return (Gvector,prevAccuracy)
		for v in TrainingSet.keys():
			Gvector[v] = Hvector[v]
		prevCost = totalCost
		prevAccuracy = newAccuracy
		print("------------------------------------------------------")	
	
def calcAccuracy(Gvector, TrainingSet):
	correctClassify = 0
	Tcount = 0
	for category in TrainingSet.values():
		for T in category:
			Tcount += 1
			classifiedAS = findClosest(Gvector,T)
			if T in TrainingSet[classifiedAS]:
				correctClassify += 1
			
	accuracy = correctClassify/Tcount
	return accuracy
				
def findClosest(Gvector, T):
	distDict = {}
	for K, exemplar in Gvector.items():
		dist = calcDistance(exemplar, T)
		distDict[K] = dist
	closestVector = min(distDict.values())
	for v in distDict.keys():
		if distDict[v] == closestVector:
			classifiedAS = v
			return classifiedAS
					

def calcDistance(exemplar, point):
	distanceSQ = 0
	for i in range(len(point)):
		diff = float(exemplar[i]) - float(point[i])
		diff = diff ** 2
		distanceSQ += diff
	return distanceSQ

def calcCentroids(TrainingSet):
	sums = {}
	setCount = {}		
	for v in TrainingSet.keys():
		if v not in sums.keys():
			sums[v] = []
		if v not in setCount.keys():
			setCount[v] = 0
		sumList = sums[v]	
		for x in TrainingSet[v]:
			setCount[v] = setCount[v] + 1
			if len(sumList) != len(x):
				for i in range(len(x)):
					sumList.append(0)
			for i in range(len(x)):
				sumList[i] += float(x[i])
				
	for v in sums.keys():
		argList = sums[v]
		for i in range(len(argList)):
			argList[i] = argList[i]/setCount[v]			
	return sums

def main():
	inputSize = len(sys.argv)
	if inputSize == 6 or inputSize == 7:
		flag = False
		TrainingSet = {}
		input_file_name = sys.argv[1]
		input_file = open(input_file_name, 'r')
		for line in input_file:
			line = line.split(',')
			size = len(line)
			category = line[size-1]
			category = category.strip('\n')
			line = line[0:size-1]
			if category in TrainingSet:
				TrainingSet[category].append(line)
			else:
				TrainingSet[category] = []
				TrainingSet[category].append(line)
		stepSize = sys.argv[2]
		Epsilon = sys.argv[3]
		MAX_COST = sys.argv[4]
		random_restarts = sys.argv[5]
		if inputSize == 7:
			output = sys.argv[6]
			if output == '-v':
				flag = True
		GVector = calcCentroids(TrainingSet)
		result = gradiantDescent(TrainingSet, stepSize, Epsilon, MAX_COST, GVector, flag, random_restarts)
		resultVector = result[0]
		accuracy = result[1]
		if flag:
			print("Final Exemplars:")
			for K in sorted(resultVector.keys()):
				print("Exemplar", K, resultVector[K])
		print("Best Accuracy Found with Centroid:", accuracy)
		
	else:
		print("Invalid input, exiting...")
		sys.exit(1)	
	
	
if __name__ == "__main__":
    main()
		
				
	
