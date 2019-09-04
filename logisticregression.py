import numpy as np
import pandas
import math


#This File is used to modularize Logisitc Regressin related Code 

#As done for Linear, we will divide Input Data into Training:80, Testing:10, Validation:10
def CreateTargetTraining(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    return t

def CreateInputTraining(InputDataSet, TrainingPercent = 80):
    T_len = int(math.ceil(len(InputDataSet)*0.01*TrainingPercent))
    d2 = InputDataSet[0:T_len,:]
    return d2

def CreateValidationPair(InputDataSet, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(InputDataSet)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = InputDataSet[TrainingCount+1:V_End,:]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def CreateTargetValidation(InputDataSet, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(InputDataSet)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =InputDataSet[TrainingCount+1:V_End]
    return t

	
#This calls internally Weight reduction at each steps along with the Cost function
def fit(features, labels, weights, lr, iters):
    cost_history = []
    for i in range(iters):
        weights = weightReduction(features, labels, weights, lr)
        cost = costAnalysis(features, labels, weights)
        cost_history.append(cost)

    return weights, cost_history

#As in logistic Regression, we will take initail gradient and keep on decreasing weight using items
#to reach optimal weight point.
def weightReduction(features, labels, weights, lr):
    N = len(features)

    evaluateFinalions = evaluate(features, weights)
    gradient = np.dot(features.T,  evaluateFinalions - labels)
    gradient /= N
    gradient *= lr
    weights -= gradient

    return weights

	
#This will keep calculating the cost at each step and return the final cost iteratirvely
def  costAnalysis(features, labels, weights):
    observations = len(labels)
    evaluateFinalions = evaluate(features, weights)
    class1_cost = -labels*np.log(evaluateFinalions)
    class2_cost = (1-labels)*np.log(1-evaluateFinalions)
    cost = class1_cost - class2_cost
    cost = cost.sum()/observations

    return cost

#The Activation function we use is Sigmoidal. We are implementing it simply with below formula
def sigmoidalOutput(z):
    return 1 / (1 + np.exp(-z))

#WE feed Sigmoidal function input verctor multiplied by loss vector  
def evaluate(X, loss):
    return sigmoidalOutput(np.dot(X, loss))

def evaluateFinal(X, loss):
    return evaluate(X, loss).round()




#This is the central method and our entry point
#We will inittailize opur weights and then we calcualte loss function
#Our AIm is to Reduce the Loss function based on the weights.
#So we will keep on reducing the weight values step by step finally when we reach optimal weight
def logisticRegression(InputDataSet,OutPutDataSet):
	print(len(InputDataSet))
	InputDataSet=np.array(InputDataSet)
	TrainingPercent = 80
	ValidationPercent = 10
	TestPercent = 10
	lr = 0.1
	num_iter = 100
	TrainingTarget = np.array(CreateTargetTraining(OutPutDataSet,TrainingPercent))
	TrainingData   = CreateInputTraining(InputDataSet,TrainingPercent)

	ValDataAct = np.array(CreateTargetValidation(OutPutDataSet,ValidationPercent, (len(TrainingTarget))))
	ValData = CreateValidationPair(InputDataSet,ValidationPercent, (len(TrainingTarget)))

	TestDataAct = np.array(CreateTargetValidation(OutPutDataSet,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
	TestData = CreateValidationPair(InputDataSet,TestPercent, (len(TrainingTarget)+len(ValDataAct)))

	weights = np.zeros(TrainingData.shape[1])
	w,c =fit(TrainingData, TrainingTarget, weights, lr, num_iter)
	AccuracyBefore = evaluate(ValData, w)
	AccuracyBefore = evaluateFinal(TestData, w)
	AccuracyFinal = (AccuracyBefore == TestDataAct).mean()
	print ("----------Final Accuracy in Logistic Regression-----"+str(AccuracyFinal))