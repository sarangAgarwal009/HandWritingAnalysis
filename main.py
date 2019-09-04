import numpy as np
import pandas as pd
import csv
import math
import neuralnetwork as nw
import logisticregression as logistic
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense


#This is imported and used so that all the Print statements are redirected to a file namely Output.txt instead of console
import sys
sys.stdout=open("outPut.txt","w")

#I will create a dictionary of the Image Id as KEy with the Feature List as value. 
#This dictionary will be used when i will be later on Concatenating or subtracting the values to compare based on Image Key
def CreateFeatureDictionary(filePath, isHumanInspection):
    featureDictionary = {}
    if isHumanInspection:			#isHumanInspection is a flag based on which i know column range in which the feature lies.
        featureList = range(2,11)	# IF the HUman data is read, then colum 2-11 should be read
        with open(filePath, 'rU') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:  
                featureDictionary.update({str(row[1]): list(int(row[i]) for i in featureList)}) 
    else:
        featureList = range(1,513)	# if gsc dATA IS READ, then feature from 1-512 columns should be read
        with open(filePath, 'rU') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:  
                featureDictionary.update({str(row[0]): list(int(row[i]) for i in featureList)})

    return featureDictionary



#In case of subtraction of Feature values, we will have either 9 or 512 feature columns
#Based on above Dictionary that i created, i will query the same and diffn file and will fetch the feature value from each
#row by row and keep subtracting and storing the final feature value
def CreateSubtractionFrame(filePath, featureDictionary, isHumanInspection, xFeatureData, yOutputVector):
    
    count = 0
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        next(reader)
        for row in reader:
            xFeatureData.append(np.subtract(featureDictionary[str(row[0])],featureDictionary[str(row[1])]))
            yOutputVector.append(int(row[2]))
            count += 1
            if isHumanInspection and count>=512:
                break
            elif not isHumanInspection and count >=2500:
                break;
    #print ("Data Matrix Generated..")
    return xFeatureData, yOutputVector


#We will take 80% training target. So for this purpose, our training matrix will be 
#divided into 80-10-10 Row percentages.
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    
    return t



#similar to subtraction done above, iam concatenating the feature value.
#Now the Feature column will be double in size, i.e either `8 or 1024
#Approach is similar as above, first created a dictionary of Image id, then match that key with corresponding 
#same and diffn pair files A and B observer, adn then concatenated it
def CreateConcatenationFrame(filePath, featureDictionary, isHumanInspection, xFeatureData, yOutputVector):
  
    count = 0
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        next(reader)
        for row in reader:
            xFeatureData.append(featureDictionary[str(row[0])]+featureDictionary[str(row[1])])
            yOutputVector.append(int(row[2]))
            count += 1
            if isHumanInspection and count>=512:
                break
            elif not isHumanInspection and count >=2500:
                break;
    
    return xFeatureData, yOutputVector

# To obtain 80% training Input data, we are just pruning the rows according to that percentage.
def CreateTrainingInputDataSet(FinalFeatureDataInput, TrainingPercent = 80):
    T_len = int(math.ceil(len(FinalFeatureDataInput[0])*0.01*TrainingPercent))
    dataSet = FinalFeatureDataInput[:,0:T_len]
    return dataSet

	

#To generate the 10% Validation data, we are using below function
def CreateValidationDataSet(FinalFeatureDataInput, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(FinalFeatureDataInput[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    xFeatureData = FinalFeatureDataInput[:,TrainingCount+1:V_End]
    return xFeatureData

# Define validation target size to be 10% of the original target size
def CreateOutputValidationDataSet(FinalFeatureDataInput, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(FinalFeatureDataInput)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    target =FinalFeatureDataInput[TrainingCount+1:V_End]
    return target

#This function is used to generate the Co-variance matrix.
def GenerateBigSigma(Data,TrainingPercent):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])
        # Compute the variance for each row vector
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        # Put the variance of each row in the entry with same row and column ids
        BigSigma[j][j] = varVect[j]

        
    BigSigma = np.dot(200,BigSigma)
    return BigSigma

#This function gets the scalar values of the design matrix that we created.
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

# This fills our design matrix on the based of Scalar values that we obtain above
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

# We get the initital setting of Weights that will
# be used in SGD solution later on.
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

# Generate the design matrix using all the values we have calculated from above so far.
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    
    return PHI

# Get the actual Validation phase Output matrix generated
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    
    return Y

# We are calcualting the Root Mean Square Error from below function
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    
    accuracy = 0.0
    counter = 0
    
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    # print ("Accuracy Generated is "+str(accuracy))
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))







#Linear Regression Code
#Will take input 2 parameters : 
#	First : Whether we have to test Human data set or GSC-Features
#	Second : Whether we have to test Concatenation or Subtraction
#Based on these 2 flags below, what i will do is decide which files i need to process ultimately.
#From here , i will fetch the Input Feature Vector and the corresponding Target Feature Vector.
#After that what i am doing is firstly doing 
# Linear Regression using SGD
#Post that i am doing Logistic Regression using separate Python file logisticregression.python
#At last, i am doing estimation using Neural Networks using neuralnetwork.py file

def linearRegressionCode(isHumanInspection,isConcat):
	
	xFeatureData=[]
	yOutputVector=[]
	if(isHumanInspection):
		firstPruning=CreateFeatureDictionary('HumanObserved-Features-Data\HumanObserved-Features-Data.csv', True)
		if(isConcat):
			xFeatureData, yOutputVector = CreateConcatenationFrame('HumanObserved-Features-Data\same_pairs.csv',firstPruning, isHumanInspection, xFeatureData, yOutputVector)
			xFeatureData, yOutputVector = CreateConcatenationFrame('HumanObserved-Features-Data\diffn_pairs.csv',firstPruning, isHumanInspection, xFeatureData, yOutputVector)
		else:
			xFeatureData, yOutputVector = CreateSubtractionFrame('HumanObserved-Features-Data\same_pairs.csv',firstPruning, isHumanInspection, xFeatureData, yOutputVector)
			xFeatureData, yOutputVector = CreateSubtractionFrame('HumanObserved-Features-Data\diffn_pairs.csv',firstPruning, isHumanInspection, xFeatureData, yOutputVector)
#If its human and whether its Concatenation or Subtraction
#Otherwise go to below code and fetch GSC files	
	else:
		firstPruning=CreateFeatureDictionary('GSC-Features-Data\GSC-Features.csv', False)
		if(isConcat):
			xFeatureData, yOutputVector = CreateConcatenationFrame('GSC-Features-Data\same_pairs.csv',firstPruning, isHumanInspection, xFeatureData, yOutputVector)
			xFeatureData, yOutputVector = CreateConcatenationFrame('GSC-Features-Data\diffn_pairs.csv',firstPruning, isHumanInspection, xFeatureData, yOutputVector)
		else:
			xFeatureData, yOutputVector = CreateSubtractionFrame('GSC-Features-Data\same_pairs.csv',firstPruning, isHumanInspection, xFeatureData, yOutputVector)
			xFeatureData, yOutputVector = CreateSubtractionFrame('GSC-Features-Data\diffn_pairs.csv',firstPruning, isHumanInspection, xFeatureData, yOutputVector)

#This code i have written to introduce some random component in the Matrix obtained.It will do a random shuffling of them
	current_state = np.random.get_state()
	np.random.shuffle(xFeatureData)
	np.random.set_state(current_state)
	np.random.shuffle(yOutputVector)
	FinalFeatureDataInput = np.transpose(xFeatureData)
		
	FinalTargetValue=yOutputVector
	
	#Dividing the code into segments of percentages by Training, Testing and validation
	TrainingPercent=80
	ValidationPercent=10
	TestPercent=10
	

	#Actually dividing the Inputs now
	TrainingTarget = np.array(GenerateTrainingTarget(FinalTargetValue,TrainingPercent))
	TrainingData   = CreateTrainingInputDataSet(FinalFeatureDataInput,TrainingPercent)
	print(TrainingTarget.shape)
	print(TrainingData.shape)
	ValDataAct = np.array(CreateOutputValidationDataSet(FinalTargetValue,ValidationPercent, (len(TrainingTarget))))
	ValData    = CreateValidationDataSet(FinalFeatureDataInput,ValidationPercent, (len(TrainingTarget)))
	print(ValDataAct.shape)
	print(ValData.shape)
	TestDataAct = np.array(CreateOutputValidationDataSet(FinalTargetValue,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
	TestData = CreateValidationDataSet(FinalFeatureDataInput,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
	print(ValDataAct.shape)
	print(ValData.shape)
	kmeans = KMeans(n_clusters=10, random_state=0).fit(np.transpose(TrainingData))
	Mu = kmeans.cluster_centers_
	BigSigma     = GenerateBigSigma(FinalFeatureDataInput, TrainingPercent) # Get Variance matrix
	print(BigSigma.shape)
	for i in range(len(BigSigma)):
		BigSigma[i][i] += 0.001
	TRAINING_PHI = GetPhiMatrix(FinalFeatureDataInput, Mu, BigSigma, TrainingPercent)
	W = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(0.01))
	TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100)
	VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)
	print(Mu.shape)

	print(TRAINING_PHI.shape)
	print(W.shape)
	print(VAL_PHI.shape)
	print(TEST_PHI.shape)

	W_Now        = np.dot(220, W)
	La           = 2
	learningRate = 0.03
	L_Erms_Val   = []
	L_Erms_TR    = []
	L_Erms_Test  = []
	W_Mat        = []

#No of epochs for which the weights will be updated
	for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
		Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
		La_Delta_E_W  = np.dot(La,W_Now)
		Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
		Delta_W       = -np.dot(learningRate,Delta_E)
		W_T_Next      = W_Now + Delta_W
    #Updating weight matrix
		W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
		TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
		Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
		L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
		VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
		Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
		L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
		TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
		Erms_Test = GetErms(TEST_OUT,TestDataAct)
		L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    
	print ('----------Gradient Descent Solution--------------------')
	print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
	print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
	print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
	



	print ("Input dimension"+str(np.size(np.array(xFeatureData),1)))

	
	
	#Print put the logistic Regression Value Now 
	print ("------------------Logistic Regression Starting--------")
	 
# Logistic Regression Code i have written in other file, I am invoking that logic from here
	logistic.logisticRegression(xFeatureData,yOutputVector);
	
	
#   Neural Network solution i have written in another python file. That i am invoking and importing here and running
	nw.runNeuralNetwork(xFeatureData,yOutputVector,np.size(np.array(xFeatureData),1))

	
	



	
	
#Run Linear Algebra Solution--------------------

print("Running Linear Algebra Solution Now")

#This is Himan data with Subtraction Dataset
print("------Running Human Data With Subtraction-----")
linearRegressionCode(True,False)


#This is Human Data with Concatenating
print("------Running Human Data With Concatenation-----")

linearRegressionCode(True,True)

print("------Running GSC data with Subtraction-------")

#This is GSC Data with Subtraction
linearRegressionCode(False,False)
print("------Running GSC data with Concatenation------")


#This is GSC Data with Concatenation
linearRegressionCode(False,True)

#closing the Output file
sys.stdout.close()







