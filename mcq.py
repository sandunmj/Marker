import cv2 as cv
import imutils
import numpy as np
import base64

def correcting(img):

    def order_points(pts):
        # initialize a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect


    def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped


    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "left-to-right":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        bounding_boxes = [cv.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        #return the list of sorted contours and bounding boxes
        return cnts, bounding_boxes


    def show_image(image):
        cv.imshow('Image', image)
        cv.waitKey(0)
        cv.destroyWindow('Image')


    def get_center(cont):
        M = cv.moments(cont)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY


    SCALE_PERCENT = 90
    answers = [2, 3, 1, 5, 1, 3, 4, 3, 5, 2, 1, 4, 3, 2, 1, 2, 5, 3, 3, 1, 4, 2, 5, 3, 2, 2, 1, 3, 5, 4, 1, 3, 4, 5, 4, 2, 1, 3, 5, 2, 4, 5, 4, 2, 2, 5, 3, 1, 2, 4]

    # originalImage = cv.imread('papers/1.jpg')
    originalImage = cv.imdecode(np.frombuffer(img, np.uint8), -1)
    # originalImage = img.astype('uint8')

    width = int(originalImage.shape[1] * SCALE_PERCENT / 100)
    height = int(originalImage.shape[0] * SCALE_PERCENT / 100)
    dim = (width, height)

    resizedImage = cv.resize(originalImage, dim, interpolation=cv.INTER_AREA)
    grayImage = cv.cvtColor(resizedImage, cv.COLOR_BGR2GRAY)
    blurredImage = cv.GaussianBlur(grayImage, (5, 5), 0)
    edgedImage = cv.Canny(blurredImage, 75, 200)

    # find contours in the edge map, then initialize the contour that corresponds to the document
    borderContours = cv.findContours(edgedImage.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    borderContours = imutils.grab_contours(borderContours)
    docContours = None

    # ensure that at least one contour was found
    if len(borderContours) > 0:
        # sort the contours according to their size in
        # descending order
        borderContours = sorted(borderContours, key=cv.contourArea, reverse=True)

        # loop over the sorted contours
        for c in borderContours:
            # approximate the contour
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                docContours = approx
                break

    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    paperImage = four_point_transform(resizedImage, docContours.reshape(4, 2))
    warpedImage = four_point_transform(grayImage, docContours.reshape(4, 2))

    # For border removal
    height, width = warpedImage.shape
    if height % 2 != 0:
        warpedImage = np.concatenate((warpedImage, np.zeros((1, width), dtype='uint8')), axis=0)
        height = warpedImage.shape[0]
    if width % 2 != 0:
        warpedImage = np.concatenate((warpedImage, np.zeros((height, 1), dtype='uint8')), axis=1)
        width = warpedImage.shape[1]

    top_border = int(height * 0.03) * 2
    bottom_border = int(height * 0.008) * 2
    left_border = int(width * 0.022) * 2
    right_border = int(width * 0.025) * 2

    print(warpedImage.shape)
    warpedImage = warpedImage[top_border:height-bottom_border, left_border:width-right_border]

    # show_image(warpedImage)
    # apply Otsu's thresholding method to binarize the warped piece of paper
    thresholdImage = cv.threshold(warpedImage, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    # thresholdImage = np.zeros(warpedImage.shape, dtype='uint8')
    # for i in range(warpedImage.shape[0]):
    #     for j in range(warpedImage.shape[1]):
    #         if warpedImage[i, j] < 150:
    #             thresholdImage[i, j] = 255
    #         else:
    #             thresholdImage[i, j] = 0

    # height, width = thresholdImage.shape  # removing the wide border
    # if width % 2 == 0:
    #     croppedImage = thresholdImage[BORDER_WIDTH:height - BORDER_WIDTH, BORDER_WIDTH:width - BORDER_WIDTH]
    # else:
    #     croppedImage = thresholdImage[BORDER_WIDTH:height - BORDER_WIDTH, BORDER_WIDTH:width - BORDER_WIDTH - 1]

    midPoint = int(warpedImage.shape[1] / 2)
    concatenatedImage = np.concatenate((thresholdImage[:, :midPoint], thresholdImage[:, midPoint:]))

    # find contours in the thresholded image, then initialize the list of contours that correspond to questions
    anyContours = cv.findContours(concatenatedImage.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    anyContours = imutils.grab_contours(anyContours)
    questionContours = []

    # loop over the contours
    for c in anyContours:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 80 and h >= 80 and 0.8 <= ar <= 1.2:
            questionContours.append(c)

    questionColoredImage = cv.threshold(concatenatedImage, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    questionColoredImage = cv.cvtColor(questionColoredImage, cv.COLOR_GRAY2RGB)
    # cv.drawContours(questionColoredImage, questionContours, -1, (0, 255, 0), 3)

    # sort the question contours top-to-bottom, then initialize the total number of correct answers
    questionContours = sort_contours(questionContours, method="top-to-bottom")[0]

    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    correct = 0
    for i in range(0, len(questionContours), 5):
        sortedContours = sort_contours(questionContours[i:i + 5], method="left-to-right ")[0]
        answerMarked = None
        numBubbled = 0
        for (j, c) in enumerate(sortedContours):
            mask = np.zeros(concatenatedImage.shape, dtype="uint8")
            cv.drawContours(mask, [c], -1, 255, -1)

            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv.bitwise_and(concatenatedImage, concatenatedImage, mask=mask)
            total = cv.countNonZero(mask)
            if total > 3000:
                numBubbled += 1
                if numBubbled > 1:
                    answerMarked = None
                    break
                answerMarked = j+1
        color = (0, 0, 255)
        answer = answers[int(i/5)]
        if answer == answerMarked:
            color = (0, 255, 0)
            correct += 1
        x, y = get_center(sortedContours[answer-1])
        cv.circle(questionColoredImage, (x, y), 25, color, -1)
        # cv.drawContours(questionColoredImage, sortedContours[answer-1], -1, color, 3)

    print(questionColoredImage.shape)
    midPointFinal = int(questionColoredImage.shape[0] / 2)
    concatenatedImageFinal = np.concatenate((questionColoredImage[:midPointFinal, :, :], questionColoredImage[midPointFinal:, :, :]), axis=1)

    print(answers)

    SCALE_PERCENT_END = 25
    width = int(concatenatedImageFinal.shape[1] * SCALE_PERCENT_END / 100)
    height = int(concatenatedImageFinal.shape[0] * SCALE_PERCENT_END / 100)
    dim = (width, height)

    concatenatedImageFinal = cv.resize(concatenatedImageFinal, dim, interpolation=cv.INTER_AREA)
    # cv.imwrite('papers/1a-{0}-{1}.jpg'.format(correct, SCALE_PERCENT), concatenatedImageFinal)
    return {"image": base64.b64encode(concatenatedImageFinal).decode("utf-8")}
