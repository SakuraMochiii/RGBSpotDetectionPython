import cv2
import numpy as np
import os

def detectDots(orig_file, file_name):
    directory_path = os.path.dirname(__file__)
    orig_file_path = os.path.join(directory_path, orig_file) #img path
    orig_img = cv2.imread(orig_file_path) #read img

    file_path = os.path.join(directory_path, file_name) #img path

    img = cv2.imread(file_path) #read img
    
    # cv2.imshow("cropped image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=16, sigmaY=0)
    divide = cv2.divide(gray, blur, scale=1)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2] #contours

    border = morph.copy()
    cv2.drawContours(border, cnts, -1, (0, 0, 0), 2)
    
    s1 = 3
    s2 = 25 #dot size
    xcnts = [] 
    
    for cnt in cnts: 
        if s1<cv2.contourArea(cnt)<s2: 
            xcnts.append(cnt) 
            cv2.drawContours(orig_img, [cnt], -1, (255,105,180), 2)
    
    str = "Number of spots: {}".format(len(xcnts)) + "\n" #output

    avg_cnt = []
    for x in xcnts:
        avg_cnt.append(np.round(x.mean(axis=0)[0]))

    str += "Spot locations: {}".format(avg_cnt)

    cv2.imshow("grayscale", morph)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("detected image", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return str

def detectRect(file_name, color):
    colors = {'red': [np.array([20, 255, 255]), np.array([0, 200, 180])],
              'green': [np.array([100, 255, 255]), np.array([40, 200, 180])],
              'blue': [np.array([120, 255, 255]), np.array([90, 130, 180])]}

    image = cv2.imread(file_name) #read img
    cv2.imshow("original image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    buf = cv2.addWeighted(image, 1.1, image, 0, 0.8) #add contrast
    hsv = cv2.cvtColor(buf, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, colors[color][1], colors[color][0])
    thresh = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,9)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #contours
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #draw outline
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    area_treshold = 10000

    blank = np.zeros(image.shape, dtype=np.uint8)
    blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    cropped = blank.copy()
    for c in cnts:
        if cv2.contourArea(c) > area_treshold:
            epsilon = 0.1*cv2.arcLength(c,True)
            c2 = cv2.approxPolyDP(c,epsilon,True)
            cropped = cv2.drawContours(blank, [c2], -1, (255, 255, 255), cv2.FILLED)
            # cropped = cv2.drawContours(blank, [c2], -1, (255, 255, 255), 5)
    
    # cv2.imshow("whatisthiseven", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow('thresh', thresh)
    filtered = cv2.bitwise_and(~image, ~image,mask = cropped)
    border = cv2.findContours(blank, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered = cv2.drawContours(filtered, border[0], -1, (255, 255, 255), 5)
    cv2.imwrite("filtered.jpg", ~filtered)
    print(detectDots(file_name, "filtered.jpg"))
    # cv2.imshow('opening', (255 - opening))
    # cv2.imshow('image', filtered)
    # cv2.waitKey()

# detectRect("redtest.jpg", "red")
# detectRect("red_test.jpg", "red")
detectRect("greentest2.jpg", "green")
# detectRect("green_test.jpg", "green")
# detectRect("bluetest.jpg", "blue")
# detectRect("blue_test.jpg", "blue")
