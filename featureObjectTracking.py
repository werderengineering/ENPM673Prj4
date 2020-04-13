import numpy as np
import cv2


def affline_Jacobian(x, y):
    return np.array([[x, 0, y, 0, 1, 0],[0, x, 0, y, 0, 1]])


def affline(p1, p2, p3, p4, p5, p6):
    return np.array([1+p1, p3, p5, p2, 1+p4, p6])


def steepestDescent(grad_template):


def inverseCompositional(frame_current, template, rect, p_prev, flag_frameCrop=False):
    """
    Parameters
    ----------
    frame_current : np.ndarray <int>
        a grayscale image of the current frame
    template : np.ndarray <int>
        a grayscale image of the template
    rect : list <int>
        the bounding box that marks the template region in tmp
    p_prev :  np.ndarray <float>
        the parameters  of the previous affine transformation warping
    """
    # check type
    assert type(frame_current) == np.ndarray
    assert type(template) == np.ndarray
    assert type(rect) == list
    assert type(rect[0]) == tuple
    assert type(p_prev) == np.ndarray
    # check shape
    assert frame_current.shape[2] == 1
    assert frame_current.shape[2] == 1
    assert len(rect) == 4
    assert p_prev.shape == (2, 6)
    # check element type
    assert type(frame_current[0, 0]) == np.uint8
    assert type(frame_current[0, 0]) == np.uint8
    assert type(rect[0][0]) == int
    assert type(p_prev[0, 0]) == float

    """get rectangle coordinates and cropped image"""
    x_left, x_right, y_upper, y_lower = rect[0]
    img_cropped = frame_current[x_left:x_right, y_upper:y_lower]    # get a crop from this frame
    if flag_frameCrop:
        cv2.imshow("Sub-image on Grayscale Frame", cv2.rectangle(frame_current, (x_left,y_upper),(x_right,y_lower),(0, 255,0),3))
        cv2.imshow("Warped Image", img_cropped)
        cv2.imshow("Template", template)
    # img_cropped = cv2.warpAffine(img_cropped, p_prev, dsize=img_cropped.shape)

    """Pre-compute of The Inverse Compositional Algorithm"""
    # Evaluate the gradient of template
    grad_template_x = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=3)  # template gradients along x
    grad_template_y = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=3)  # template gradients along x
    grad_template = np.concatenate((grad_template_x, grad_template_y), axis=1)
    Jacobian_W = affline_Jacobian()  # Evaluate the jacobian
    steepestDescentimage =  # Compute the steepest descent image
    p_new = p_prev.copy()   # copy the previous to new one

    #
    error = img_cropped - template  # compute the error between new feature and previous frame

    return p_new


def affineLKtracker():



def debug():
    # affineLKtracker(img, tmp, rect, p_prev, flag_frameCrop=True)


debug()