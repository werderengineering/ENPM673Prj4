from __main__ import *

def tracker(frame,Pframe):

    framG=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    PFrameG = cv2.cvtColor(Pframe, cv2.COLOR_BGR2GRAY)

    #Find center

    LX=frame.shape[0]
    LY=frame.shape[1]
    CY=int(LX/2)
    CX=int(LY/2)

    NormF1=framG/255
    NormFP=PFrameG/255


    ##################################################
    # w=int(NormF1.shape[0]/2)
    #
    # kernel_x = np.array([[-1., 1.], [-1., 1.]])
    # kernel_y = np.array([[-1., -1.], [1., 1.]])
    # kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    #
    # mode = 'same'
    # fx = signal.convolve2d(NormF1, kernel_x, boundary='symm', mode=mode)
    # fy = signal.convolve2d(NormF1, kernel_y, boundary='symm', mode=mode)
    # ft = signal.convolve2d(NormFP, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(NormF1, -kernel_t, boundary='symm', mode=mode)
    #
    #
    # u = np.zeros(NormF1.shape)
    # v = np.zeros(NormF1.shape)
    #
    # for i in range(w, NormF1.shape[0] - w):
    #     for j in range(w, I1g.shape[1] - w):
    #         Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
    #         Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
    #         It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
    #         # b = ... # get b here
    #         # A = ... # get A here
    #         # if threshold τ is larger than the smallest eigenvalue of A'A:
    #         nu = ...  # get velocity here
    #         u[i, j] = nu[0]
    #         v[i, j] = nu[1]


    #Output should be desired trackable XY coordinates

    return CX,CY
################################################