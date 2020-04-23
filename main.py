from training import trainModel
from HandwrittenScan import handwrittenScan
import cv2

trainModelObj = trainModel()
handwrittenScanObj = handwrittenScan()

knn_Model = trainModelObj.trainAndTestKNN(kValue = 3)

inputPath = './input/002.jpg'
img=cv2.imread(inputPath)

result,mainImage,rect = handwrittenScanObj.extractPredictDigits(inputPath,knn_Model)
handwrittenScanObj.displayDigit(result, mainImage, rect)

cv2.waitKey(0)
cv2.destroyAllWindows()


