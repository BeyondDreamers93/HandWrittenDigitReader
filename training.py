import cv2
import numpy as np

class trainModel:
    def __init__(self):
        self.trainData = None
        self.testData = None
        self.trainLabels = None
        self.testLabels = None
        self.inputImagePath = './train/digits.png'
        return

    def trainAndTestKNN(self,kValue):
        inputImage = cv2.imread(self.inputImagePath)
        # Initiate kNN, train the data, then test it with test data for k=3
        knn = cv2.ml.KNearest_create()
        self.__processInput(inputImage, 70)
        #print(self.trainData.shape)
        #print(self.trainLabels.shape)

        knn.train(self.trainData,cv2.ml.ROW_SAMPLE, self.trainLabels)
        ret, result, neighbors, distance = knn.findNearest(self.testData, k=kValue)

        # Now we check the accuracy of classification
        # For that, compare the result with test_labels and check which are wrong
        matches = result == self.testLabels
        correct = np.count_nonzero(matches)
        accuracy = correct * (100.0 / result.size)
        print("Accuracy is = %.2f" % accuracy + "%")
        return knn

    def __processInput(self,inputImage,trainPerc):
        gray = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)
        small = cv2.pyrDown(inputImage)

        # Split the image to 5000 cells, each 20x20 size
        # This gives us a 4-dim array: 50 x 100 x 20 x 20
        cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

        # Convert the List data type to Numpy Array of shape (50,100,20,20)
        x = np.array(cells)

        # Split the full data set into two segments
        # One will be used fro Training the model, the other as a test data set
        self.trainData = x[:,:trainPerc].reshape(-1,400).astype(np.float32) # Size = (3500,400)
        self.testData = x[:,trainPerc:100].reshape(-1,400).astype(np.float32) # Size = (1500,400)

        # Create labels for train and test data
        k = [0,1,2,3,4,5,6,7,8,9]
        self.trainLabels = np.repeat(k,350)[:,np.newaxis]
        self.testLabels = np.repeat(k,150)[:,np.newaxis]
        return 