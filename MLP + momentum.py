import math
import random
import csv

data = [[]]
with open('clean.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
	data = list(spamreader)
	
numInputs = 8
numOutputs = 1
numNeurons = 11
numTrainingRows = 353
numValidationRows = 118
numTestRows = 117
numNodes = numInputs + numOutputs + numNeurons
numEpochs = 2250
mse = 0

w = [ [0]*(numNodes + 1) for i in range(numNodes + 1) ]
wNew = [ [0]*(numNodes + 1) for i in range(numNodes + 1) ]
wDelta = [ [0]*(numNodes + 1) for i in range(numNodes + 1) ]
wMean = 0
s = [0]*(numNodes + 1)
u = [0]*(numNodes + 1)
fS = [0]*(numNodes + 1)
delta = [0]*(numNodes + 1)
alpha = 0.9
step = 0.2

for i in range (0, numNodes):
	for j in range(numInputs + 1, numNodes + 1):
		w[i][j] = random.uniform(-(numInputs/2), numInputs/2)

# TRAINING
x = 0
while x < numEpochs:
	for i in range (0, numTrainingRows):
		for j in range (numInputs + 1, numNodes):
			s[j] = w[0][j]
			for node in range(1, numInputs + 1):
				s[j] += w[node][j] * data[i][node - 1]
			u[j] = 1/(1 + math.exp(-s[j]))
		s[numNodes] = w[0][numNodes]
		for node in range(numInputs + 1, numInputs + numNeurons + 1):
			s[numNodes] += (w[node][numNodes]*u[node]) 
		u[numNodes] = 1/(1 + math.exp(-s[numNodes]))
		fS[numNodes] = u[numNodes] * (1 - u[numNodes])
		delta[numNodes] = (data[i][numInputs] - u[numNodes]) * fS[numNodes]
		for j in range (numInputs + 1, numNodes):
				fS[j] = u[j] * (1 - u[j])
				delta[j] = w[j][numNodes] * delta[numNodes] * fS[j]
#WEIGHTS UPDATE
		for i in range (0, numNodes): 
			for j in range (numInputs + 1, numNodes + 1):
				if i == 0:
						w[i][j] = w[i][j] + (step * delta[j])
				else:
						w[i][j] = w[i][j] + (step * delta[j] * u[j])
				wDelta[i][j] = wNew[i][j] - w[i][j]
		w = wNew.copy()
	x = x + 1
# VALIDATION
for i in range (numTrainingRows, numTrainingRows + numValidationRows):
		for j in range (numInputs + 1, numNodes):
			s[j] = w[0][j]
			for node in range(1, numInputs + 1):
				s[j] += w[node][j] * data[i][node -1]
			u[j] = 1/(1 + math.exp(-s[j]))
		s[numNodes] = w[0][numNodes]
		for node in range(numInputs + 1, numInputs + numNeurons + 1):
			s[numNodes] += (w[node][numNodes]*u[node]) 
		u[numNodes] = 1/(1 + math.exp(-s[numNodes])) 
		wMean = wMean + ((data[i][numInputs] - u[numNodes])**2)
#TEST
for i in range (numTrainingRows + numValidationRows, numTrainingRows + numValidationRows + numTestRows):
	for j in range (numInputs + 1, numNodes):
		s[j] = w[0][j]
		for node in range (1, numInputs + 1):
			s[j] += (w[node][j] * data[i][node - 1])
		u[j] = 1/(1 + math.exp(-s[j]))
	s[numNodes] = w[0][numNodes]
	for node in range (numInputs + 1, numInputs + numNeurons + 1):
		s[numNodes] += (w[node][numNodes] * u[node])
	u[numNodes] = 1/(1 + math.exp(-s[numNodes]))
	wMean = wMean + ((data[i][numInputs] - u[numNodes])**2)

mse = wMean / numTestRows
print(mse)