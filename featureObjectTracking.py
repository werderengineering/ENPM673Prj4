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


def imgToArray(img):
    """change the (n x m) shape array to (n*m, 1) array"""
    assert type(img) == np.ndarray
    width, height = img.shape
    return img.reshape((width*height, 1))


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
    point_lowerRight = rect[0:2, 1].astype(int)
    width = int(point_lowerRight[0] - point_upperLeft[0])    # width of bounding box
    height = int(point_lowerRight[1] - point_upperLeft[1])   # height of bounding box
    Jacobian_W = np.zeros((width*height, 2, 6))   # initialize the Jacobian of affine transformation matrix
    index = 0
    for y in range(point_upperLeft[1], point_lowerRight[1]):    # loop over range of y: y is second element in a variable vector
        for x in range(point_upperLeft[0], point_lowerRight[0]):    # loop over range of x: x is first element in a variable vector
            Jacobian_W[index, :] = np.array([[x, 0, y, 0, 1, 0],
                                             [0, x, 0, y, 0, 1]])
            index = index + 1
    assert Jacobian_W.shape == (width*height, 2, 6)
    return Jacobian_W   #np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])


def hessiaMatrixInverse(SD):
    assert type(SD) == np.ndarray
    assert SD.shape[1] == 6
    H = np.dot(SD.T, SD)
    assert H.shape == (6, 6)    # check output shape
    return np.linalg.inv(H)


def compositionTwoAffine(W1, W2):
    assert W1.shape == (3, 3) and W2.shape == (3, 3)
    W_combined = np.dot(W1, W2)
    return W_combined


def steepestDescent(grad_template, Jacobian_W):
    assert type(grad_template) == np.ndarray
    assert type(Jacobian_W) == np.ndarray
    assert grad_template.shape[0] == Jacobian_W.shape[0]
    assert grad_template.shape[1] == 2
    assert Jacobian_W.shape[1] == 2 and Jacobian_W.shape[2] == 6
    shape_grad = (grad_template.shape[0], 2, 1)
    grad_template_reshape = grad_template.reshape(shape_grad)
    SD = np.sum(np.multiply(grad_template_reshape, Jacobian_W), axis=1)
    assert SD.shape == (grad_template.shape[0], 6)
    return SD


def inverseCompositional(frame_current, template, rect, p_prev, flag_frameCrop=False):
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
    p_prev :  np.ndarray <float>
        the parameters  of the previous affine transformation warping
    """
    # check type
    assert type(frame_current) == np.ndarray
    assert type(template) == np.ndarray
    assert type(rect) == np.ndarray
    assert type(p_prev) == np.ndarray
    # check shape
    assert len(frame_current.shape) == 2
    assert rect.shape == (3, 2)
    assert p_prev.shape == (6, 1)
    # check element type
    assert type(frame_current[0, 0]) == np.uint8
    assert type(rect[0][0]) == float or np.int32, "The bounding box upper right x coordinates is a " + str(type(rect[0][0]))
    assert type(p_prev[0, 0]) == np.float64, "The initial p variable is a " + str(type(p_prev[0, 0]))

    W_x_p = affline(p_prev)    # initialize a affine transformation that map a vector to itself
    rect = np.dot(W_x_p, rect)  # new rect
    img_warpped = ip.subImageByBoundingBox(frame_current, rect)    # get a crop from this frame
    if flag_frameCrop:
        ip.drawRect(frame_current, rect, True)
        cv2.imshow("Initial Warped Image", img_warpped)
        cv2.imshow("Template", template)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    # img_cropped = cv2.warpAffine(img_cropped, p_prev, dsize=img_cropped.shape)

    """Pre-compute of The Inverse Compositional Algorithm"""
    # Evaluate the gradient of template
    grad_template_x = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=3)  # template gradients along x
    grad_template_y = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize=3)  # template gradients along x
    grad_template = np.concatenate((imgToArray(grad_template_x), imgToArray(grad_template_y)), axis=1)
    Jacobian_W = affline_Jacobian(rect)  # Evaluate the jacobian
    SD = steepestDescent(grad_template, Jacobian_W)    # Compute the steepest descent image
    Hinv = hessiaMatrixInverse(SD)  # compute the inverse of Hessian matrix based on SD
    """Iteration of The Inverse Compositional Algorithm"""
    for i in range(10):
        error = img_warpped - template  # compute the error between new feature and previous frame, shape: (400x1)
        delta_p = -np.dot(Hinv, np.dot(SD.T, imgToArray(error))) # shape: (6x1) = (6x6) (6x400)(400x1)
        W_delta_p = affline(delta_p)    # get the affine transformation, shape is (3, 3)
        W_x_p = compositionTwoAffine(W_x_p, np.linalg.inv(W_delta_p))   # update the affine transformation, shape is (3x3)
    if flag_frameCrop:
        cv2.imshow("Initial Warped Image", ip.subImageByBoundingBox(frame_current, rect))
        if cv2.waitKey(0):
            cv2.destroyAllWindows()

    return np.dot(W_x_p, rect)  # return the rectangle upperLeft and lowerRight corners


def affineLKtracker(frame_current, template, rect, algorithm="inverse Compositional"):
    """
        Parameters
        ----------
        frame_current : np.ndarray <int>
            a grayscale image of the current frame
        template : np.ndarray <int>
            a grayscale image of the template
        rect : np.ndarray <float>
            a bounding box that marks the template region in previous frame, the rectangle coordinates that intend to define the
            tracking feature location at the image
        param algorithm:
            algorithm you would like to run the Lucas-Kanade algorithm
        Returns
        ----------
        rect: new bounding box in bew frame
    """
    # check type
    assert type(frame_current) == np.ndarray and type(template) == np.ndarray and type(rect) == np.ndarray
    # check shape
    assert len(frame_current.shape) == 2, "frame shape is " + str(frame_current.shape)
    assert rect.shape == (3, 2)
    # check element type
    assert type(frame_current[0, 0]) == np.uint8
    assert type(rect[0][0]) == float or np.int32, "The bounding box upper right x coordinates is a " + str(type(rect[0][0]))
    p = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).reshape(6, 1)
    """get a new rect that locate the feature on current frame"""
    if algorithm == "inverse Compositional":
        rect = inverseCompositional(frame_current, template, rect, p, flag_frameCrop=False)
    else:
        Exception("No such algorithm: " + str(algorithm))
    return rect


def debug():
    # affineLKtracker(img, tmp, rect, p_prev, flag_frameCrop=True)
    return 0