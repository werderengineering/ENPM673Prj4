import numpy as np
import cv2
import imagesProcessing as ip


def affline(p):
    assert type(p) == np.ndarray
    assert p.shape == (6, 1)
    p1, p2, p3, p4, p5, p6 = p.reshape(6)
    return np.array([[1+p1, p3, p5], [p2, 1+p4, p6], [0, 0, 1]])


def afflineInv(p):
    assert type(p) == np.ndarray
    assert p.shape == (6, 1)
    p = affline(p)
    return np.linalg.inv(p)


def affline_Jacobian(rect):
    """
    Parameters
    rect : list <int>
        the bounding box that marks the template region in tmp, first point is upper left, second is lower right
    return
        Jacobian_W: np.ndarray<int>
    """
    assert type(rect) == np.ndarray
    point_upperLeft = rect[0:2, 0].astype(int)
    point_upperRight = rect[0:2, 1].astype(int)
    point_lowerRight = rect[0:2, 2].astype(int)

    width = int(point_lowerRight[0] - point_upperLeft[0])    # width of bounding box
    height = int(point_lowerRight[1] - point_upperLeft[1])   # height of bounding box
    Jacobian_W = np.zeros(((width+1)*(height+1), 2, 6))   # initialize the Jacobian of affine transformation matrix
    index = 0
    for y in range(point_upperLeft[1], point_lowerRight[1]+1):    # loop over range of y: y is second element in a variable vector
        for x in range(point_upperLeft[0], point_lowerRight[0]+1):    # loop over range of x: x is first element in a variable vector
            Jacobian_W[index, :] = np.array([[x, 0, y, 0, 1, 0],
                                             [0, x, 0, y, 0, 1]])
            index = index + 1
    assert Jacobian_W.shape == ((width+1)*(height+1), 2, 6)
    return Jacobian_W   #np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])


def hessiaMatrixInverse(SD):
    assert type(SD) == np.ndarray
    assert SD.shape[1] == 6
    H = np.dot(SD.T, SD)
    assert H.shape == (6, 6)    # check output shape
    assert np.linalg.det(H) > 0.0
    return np.linalg.inv(H)


def compositionTwoAffine(W1, W2):
    assert W1.shape == (3, 3) and W2.shape == (3, 3)
    W_combined = np.dot(W1, W2)
    return W_combined


def steepestDescent(grad_template, Jacobian_W):
    assert type(grad_template) == np.ndarray
    assert type(Jacobian_W) == np.ndarray
    assert grad_template.shape[0] == Jacobian_W.shape[0], "The template gradient template have shape of " + str(grad_template.shape) + ", Jacobian_W has a shape of " + str(Jacobian_W.shape)
    assert grad_template.shape[1] == 2
    assert Jacobian_W.shape[1] == 2 and Jacobian_W.shape[2] == 6
    shape_grad = (grad_template.shape[0], 2, 1)     # shape: (400, 2, 1)
    grad_template_reshape = grad_template.reshape(shape_grad)   # shape: (400, 2, 1)
    SD_simi = np.multiply(grad_template_reshape, Jacobian_W)     # shape: (400, 2, 6) = (400, 2, 1) x (400, 2, 6)
    SD = np.sum(SD_simi, axis=1)     # shape (400, 6) = sum((400, 2, 6))
    assert SD.shape == (grad_template.shape[0], 6)
    return SD


def inverseCompositional(frame_current, template, rect_template, p_prev, W_prev, cache, flag_showItera=False):
    """
    Parameters
    ----------
    frame_current : np.ndarray <int>
        a grayscale image of the current frame
    template : np.ndarray <int>
        a grayscale image of the template
    rect : np.ndarray <float>
        a bounding box that marks the template region in img, the rectangle coordinates that intend to define the
        tracking feature location at the image
    W_prev :  np.ndarray <float>
        the matrix of previous affine transformation warping
    """
    threshold = 0.1
    iteration = 10000
    # check type
    assert type(frame_current) == np.ndarray
    assert type(template) == np.ndarray
    assert type(rect_template) == np.ndarray
    assert type(W_prev) == np.ndarray
    # check shape
    assert len(frame_current.shape) == 2
    assert rect_template.shape == (3, 3)
    assert W_prev.shape == (3, 3)
    # check element type
    assert type(frame_current[0, 0]) == np.uint8
    assert type(rect_template[0][0]) == float or np.int32, "The bounding box upper right x coordinates is a " + str(type(rect_template[0][0]))
    assert type(W_prev[0, 0]) == np.float64, "The initial p variable is a " + str(type(W_prev[0, 0]))
    """pre-processing the data from the tracking of last frame"""
    grad_template_x, grad_template_y, grad_template, Jacobian_W, SD, Hinv = cache
    """calculate the pixels inside the rectangle bounding box that highlight the matching feature from previous frame"""
    if flag_showItera:
        frame_current_dewarped = cv2.warpAffine(ip.floatToUint8(frame_current), np.linalg.inv(W_prev)[0:2, :], (frame_current.shape[1],
                                                                                       frame_current.shape[
                                                                                           0]))  # dewarp the current frame to go back to the view of first frame
        template_frame_current_dewarped = ip.subImageInBoundingBoxAndEq(ip.floatToUint8(frame_current_dewarped), rect_template, histEqualize=True)
        cv2.imshow("Dewarped current frame before update", ip.floatToUint8(frame_current_dewarped))
        cv2.imshow("Feature cropped from dewarped current frame before update", ip.floatToUint8(template_frame_current_dewarped))
        cv2.imshow("Feature template", ip.floatToUint8(template))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    """The new affine warp transformation"""
    W_update = W_prev.copy()
    p_update = p_prev.copy()
    error = 0

    """Iteration of The Inverse Compositional Algorithm"""
    for i in range(iteration):
        W_prev_inv = np.linalg.inv(W_prev)
        # dewarp the current frame to go back to the view of first frame
        frame_current_dewarped = cv2.warpAffine(frame_current, W_prev_inv[0:2, :], (frame_current.shape[1], frame_current.shape[0]))
        template_frame_current_dewarped = ip.subImageInBoundingBoxAndEq(frame_current_dewarped, rect_template, histEqualize=True)
        template_frame_current_dewarped = ip.uint8ToFloat(template_frame_current_dewarped)
        error = template - template_frame_current_dewarped  # compute the error between new feature and previous frame, shape: (20x20)
        error_column = ip.imgToArray(error)   # reshape error array to be (400, 1), checked

        delta_p = np.dot(-Hinv, np.dot(SD.T, error_column)) # shape: (6x1) = (6x6) (6x400)(400x1) ??????????????????????????????????????
        W_delta_p = affline(delta_p)    # get the affine transformation, shape is (3, 3)
        W_delta_p = np.linalg.inv(W_delta_p)

        p_update = p_update + delta_p

        W_update = compositionTwoAffine(W_update, W_delta_p)   # update the affine transformation, shape is (3x3)
        if np.linalg.norm(delta_p) < threshold:
            print("Converged at iteration #" + str(i))
            break
    print("The average error is \n" + str(np.mean(error)))
    print("Delta p is \n" + str((p_update-p_prev).T))
    """show Lucas-Kanade tracking result"""
    if flag_showItera:
        # highlight where big error is
        # cv2.circle(frame_current_dewarped)
        frame_current_dewarped = cv2.warpAffine(ip.floatToUint8(frame_current), np.linalg.inv(W_prev)[0:2, :], (frame_current.shape[1],
                                                                                       frame_current.shape[
                                                                                           0]))  # dewarp the current frame to go back to the view of first frame
        template_frame_current_dewarped = ip.subImageInBoundingBoxAndEq(ip.floatToUint8(frame_current_dewarped), rect_template, histEqualize=True)
        cv2.imshow("Dewarped current frame after update", ip.floatToUint8(frame_current_dewarped))
        cv2.imshow("Feature cropped from dewarped current frame after update", ip.floatToUint8(template_frame_current_dewarped))
        cv2.imshow("Feature template", ip.floatToUint8(template))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    return np.dot(W_update, rect_template), p_update, W_update  # return the rectangle upperLeft and lowerRight corners


def affineLKFeatureRegnization(frame_current, rect, template, rect_template, p_prev, W_prev, cache, algorithm="inverse Compositional", flag_itera=False):
    """
        Parameters
        ----------
        frame_current : np.ndarray <int>
            a grayscale image of the current frame
        rect : np.ndarray <float>
            a bounding box that marks the feature region in previous frame, the rectangle coordinates that close to the
            tracking feature location at the image
        template : np.ndarray <int>
            a grayscale image of the template
        rect_template: np.ndarray <int>
            a bounding box that marks the template region in first frame, the rectangle coordinates that defines the
            tracking feature location at the first image
        W_prev : np.ndarray <float>
            previous warp function, will be a affine transformation in this project
        algorithm:
            algorithm you would like to run the Lucas-Kanade algorithm
        Returns
        ----------
        rect: new bounding box in bew frame
    """
    # check type
    assert type(frame_current) == np.ndarray and type(template) == np.ndarray and type(rect) == np.ndarray
    # check shape
    assert len(frame_current.shape) == 2, "frame shape is " + str(frame_current.shape)
    assert rect.shape == (3, 2) or (3, 3)
    # check element type
    assert type(frame_current[0, 0]) == np.uint8
    assert type(rect[0][0]) == float or np.int32, "The bounding box upper right x coordinates is a " + str(type(rect[0][0]))

    p_update, W_update = p_prev.copy(), W_prev.copy()

    """get a new rect that locate the feature on current frame"""
    if algorithm == "inverse Compositional":
        if len(cache) == 0:
            """Pre-compute of The Inverse Compositional Algorithm"""
            # Evaluate the gradient of template
            grad_template_x = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=3)  # template gradients along x, shape: (20x20)
            grad_template_y = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize=3)  # template gradients along y, shape: (20x20)
            grad_template = np.concatenate((ip.imgToArray(grad_template_x), ip.imgToArray(grad_template_y)), axis=1) # shape: (400x2)
            Jacobian_W = affline_Jacobian(rect_template)  # Evaluate the jacobian, shape: (400, 2, 6)   ???????????????????????
            SD = steepestDescent(grad_template, Jacobian_W)  # Compute the steepest descent image, shape: (400, 6) = (400, 2, 6) x (400, 2, 1)  checked
            Hinv = hessiaMatrixInverse(SD)  # compute the inverse of Hessian matrix based on SD, shape: (6, 6) = (6x400) dot (400x6)
            cache = grad_template_x, grad_template_y, grad_template, Jacobian_W, SD, Hinv

        rect, p_update, W_update = inverseCompositional(frame_current, template, rect_template, p_prev, W_prev, cache, flag_showItera=flag_itera)
    else:
        Exception("No such algorithm: " + str(algorithm))
    return rect, p_update, W_update, cache



def LKRegisteration(frames, template, rect_template, flag_showFeatureRegisteration = True):
    """
        Parameters
        ----------
        frames : np.ndarray <int>
            a grayscale image of the current frame
        template : np.ndarray <int>
            a grayscale image of the template
        rect_template: np.ndarray <int>
            a bounding box that marks the template region in first frame, the rectangle coordinates that defines the
            tracking feature location at the first image
    """
    template = ip.uint8ToFloat(template)
    rect = rect_template.copy()
    p_prev = np.zeros((6, 1))   # assume the first frame is exact same with the frame where template cropped from
    transformation_affine = np.eye(3,
                                   3)  # since it's the warp transformation of first image itself, it should change the coordinates
    cache = []  # some constant for each iteration
    for frame in frames:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        rect, p_prev, transformation_affine, cache = affineLKFeatureRegnization(frame_gray, rect, template, rect_template, p_prev,
                                                                        transformation_affine, cache,
                                                                        algorithm="inverse Compositional",
                                                                        flag_itera=True)  # rect update
        if flag_showFeatureRegisteration:
            ip.drawRect(frame, rect, flag_showImgFeatureWithMarked=True)

def debug():
    # affineLKtracker(img, tmp, rect, p_prev, flag_frameCrop=True)
    return 0