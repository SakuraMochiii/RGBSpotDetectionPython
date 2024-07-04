import cv2
import numpy as np
import os

'''Displays the specified image, press any key to close'''
def display_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
'''main function to detect'''
def detectDots(orig_file, file_name):

    '''read image from path'''
    directory_path = os.path.dirname(__file__)
    orig_file_path = os.path.join(directory_path, orig_file) #img path
    orig_img = cv2.imread(orig_file_path) #read img

    file_path = os.path.join(directory_path, file_name) #filtered img path
    img = cv2.imread(file_path) #read filtered img

    '''process image'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=16, sigmaY=0)
    divide = cv2.divide(gray, blur, scale=1)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2] #contours

    '''draw contours and filter by area'''
    border = morph.copy()
    cv2.drawContours(border, cnts, -1, (0, 0, 0), 2)
    
    s1 = 3
    s2 = 25 #dot size
    xcnts = []
    
    for cnt in cnts:
        if s1<cv2.contourArea(cnt)<s2:
            xcnts.append(cnt)
            cv2.drawContours(orig_img, [cnt], -1, (255,105,180), 2)
    
    '''print out results'''
    str = "Number of spots: {}".format(len(xcnts)) + "\n" #output

    avg_cnt = []
    for x in xcnts:
        avg_cnt.append(np.round(x.mean(axis=0)[0]))

    str += "Spot locations: {}".format(avg_cnt)

    '''display final image with detected spots'''
    # display_img("grayscale", morph)
    display_img("detected image", orig_img)

    return str

'''helper function to filter by color'''
def detectRect(file_name, color):
    colors = {'red': [np.array([100, 255, 255]), np.array([0, 200, 180])],
              'green': [np.array([100, 255, 255]), np.array([40, 200, 180])],
              'blue': [np.array([120, 255, 255]), np.array([90, 130, 180])]}
            #   'red': [np.array([190,255,255]), np.array([160,20,70])], works for redtest2 not redtest1

    image = cv2.imread(file_name) #read img
    display_img("original image", image)
    buf = cv2.addWeighted(image, 1.1, image, 0, 0.8) #add contrast
    # display_img("buf", buf)
    hsv = cv2.cvtColor(buf, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, colors[color][1], colors[color][0])
    thresh = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,9)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #contours
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #draw outline
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    area_treshold = 5000

    blank = np.zeros(image.shape, dtype=np.uint8)
    blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    cropped = blank.copy()
    for c in cnts:
        if cv2.contourArea(c) > area_treshold:
            epsilon = 0.05*cv2.arcLength(c,True)
            c2 = cv2.approxPolyDP(c,epsilon,True)
            if len(c2) == 4:
                cropped = cv2.drawContours(blank, [c2], -1, (255, 255, 255), cv2.FILLED)
            else:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cropped = cv2.drawContours(blank,[box],-1,(255,255,255),cv2.FILLED)

    # display_img("thresh", thresh)
    filtered = cv2.bitwise_and(~image, ~image,mask = cropped)
    border = cv2.findContours(blank, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered = cv2.drawContours(filtered, border[0], -1, (255, 255, 255), 5)
    cv2.imwrite("filtered.jpg", ~filtered)
    print(detectDots(file_name, "filtered.jpg"))
    # display_img('opening', (255 - opening))
    # display_img('image', filtered)

detectRect("redtest.jpg", "red")
detectRect("greentest2.jpg", "green")
detectRect("bluetest.jpg", "blue")
detectRect("blue_test.jpg", "blue")
