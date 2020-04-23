import cv2
import numpy as np

class handwrittenScan:
    def __init__(self):
        self.full_number = []
        return
    def x_cord_contour(self,contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates

        M = cv2.moments(contour)
        return (int(M['m10']/M['m00']))
        

    def makeSquare(self,not_square):
        # This function takes an image and makes the dimenions square
        # It adds black pixels as the padding where needed
        
        BLACK = [0,0,0]
        img_dim = not_square.shape
        height = img_dim[0]
        width = img_dim[1]
        #print("Height = ", height, "Width = ", width)
        if (height == width):
            square = not_square
            return square
        else:
            doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
            height = height * 2
            width = width * 2
            #print("New Height = ", height, "New Width = ", width)
            if (height > width):
                pad = int((height - width)/2)
                #print("Padding = ", pad)
                doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=BLACK)
            else:
                pad = int((width - height)/2)
                #print("Padding = ", pad)
                doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,
                                                    cv2.BORDER_CONSTANT,value=BLACK)
        doublesize_square_dim = doublesize_square.shape
        #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
        return doublesize_square


    def resize_to_pixel(self,dimensions, image):
        # This function then re-sizes an image to the specificied dimenions
        
        buffer_pix = 4
        dimensions  = dimensions - buffer_pix
        squared = image
        r = float(dimensions) / squared.shape[1]
        dim = (dimensions, int(squared.shape[0] * r))
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        img_dim2 = resized.shape
        height_r = img_dim2[0]
        width_r = img_dim2[1]
        BLACK = [0,0,0]
        if (height_r > width_r):
            resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
        if (height_r < width_r):
            resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
        p = 2
        ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
        img_dim = ReSizedImg.shape
        height = img_dim[0]
        width = img_dim[1]
        #print("Padded Height = ", height, "Width = ", width)
        return ReSizedImg

    def extractPredictDigits(self,inputImage,knn):
        mainImage = cv2.imread(inputImage)
        refsizex = 872
        refsizey = 532
        fx = refsizex/mainImage.shape[1]
        fy = refsizey/mainImage.shape[0]
        #print(fx,fy)
        resizedImage = cv2.resize(mainImage,None,fx = fx,fy=fy)
        gray = cv2.cvtColor(resizedImage,cv2.COLOR_BGR2GRAY)
        # Blur image then find edges using Canny 
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        #Find Contours
        _, contours, _ = cv2.findContours(edged.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        #print(len(contours))
        contoursSelected = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                contoursSelected.append(contour)

        if len(contoursSelected) == 0:
            print("No Digits Found/Digits are Small")
            return 
        #Sort out contours left to right by using their x cordinates
        contourSelected = sorted(contoursSelected, key = self.x_cord_contour, reverse = False)
        count = 0
        for c in contourSelected:
        # compute the bounding box for the rectangle
            (x, y, w, h) = cv2.boundingRect(c)    
            
            #cv2.drawContours(image, contours, -1, (0,255,0), 3)
            #cv2.imshow("Contours", image)
            if w >= 5 and h >= 25:
                roi = blurred[y:y + h, x:x + w]
                ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
                squared = self.makeSquare(roi)
                final = self.resize_to_pixel(20, squared)
                count = count + 1
                final_str = "final" + str(count)
                #cv2.imshow(final_str, final)
                final_array = final.reshape((1,400))
                final_array = final_array.astype(np.float32)
                ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
                self.displayDigit(result,resizedImage,(x, y, w, h))
                if(cv2.waitKey(1) == 13):
                    cv2.destroyAllWindows()
        return result,resizedImage,(x, y, w, h)

    def displayDigit(self,result,image,rect):
        number = str(int(float(result[0])))
        self.full_number.append(number)
        # draw a rectangle around the digit, the show what the
        # digit was classified as
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
        cv2.putText(image, number, (rect[0] , rect[1] + 155),
        cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image)
        return

