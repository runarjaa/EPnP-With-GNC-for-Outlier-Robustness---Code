import numpy as np
import open3d as o3d

# ---------- Math functions ----------
def skewm(r):
    return np.array([[0,-r[2],r[1]], [r[2],0,-r[0]], [-r[1],r[0],0]])

def expso3(u):
    S = skewm(u); un = np.linalg.norm(u)
    return np.eye(3) + np.sinc(un/np.pi)*S \
        + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S


# For GNC
def w_from_r(r, eps, mu):
    if abs(r) < eps * np.sqrt(mu/(mu+1)):
        w = 1
    elif abs(r) > eps * np.sqrt((mu+1)/mu):
        w = 0
    else:
        w = (eps * np.sqrt(mu*(mu+1)))/abs(r) - mu
    return w




# ---------- File loading functions ----------
def load_points_from_file(file_loc):
    if file_loc[-4:] == '.off':
        CAD_off = o3d.io.read_triangle_mesh(file_loc)
        return make_points_homogenous(np.asarray(CAD_off.vertices))

def make_points_homogenous(points):
    if points.shape[1] == 3:
        return np.c_[points, np.ones(points.shape[0])]
    elif points.shape[1] == 4:
        print("Already homogeneous")
        return points

def downsample_points(points, n_corr):
    p = np.random.permutation(len(points))
    points = points[p]
    return points[:n_corr,:]

def scale_points(ph, scale):
    temp = ph[:,:3] * scale
    ph[:,:3] = temp
    return ph

def compute_pixels(xh_w, T, C, sigma = 0, outlier_percentage = 0):
    sn = (np.eye(3,4) @ T @ xh_w.T).T
    snorm = sn/sn[:,2].reshape((sn.shape[0],1))
    pix = (C @ snorm.T).T
    pix = np.rint(pix)

    # Noise
    if not sigma < 1:
        noise = np.rint(np.random.normal(0, sigma, (pix.shape[0], 2)))
        pix[:,:2] = pix[:,:2] + noise

    # Outliers
    if not outlier_percentage < 1:
        outliers = np.rint(pix.shape[0]*(outlier_percentage/100)).astype(int)
        pix[:outliers,0] = np.random.randint(0,C[0,2]*2, outliers)
        pix[:outliers,1] = np.random.randint(0,C[1,2]*2, outliers)

    return pix

def shuffle_points(pix, points):
    assert len(pix) == len(points)
    p = np.random.permutation(len(pix))
    return pix[p], points[p]

# ---------- Helper functions for setting up a test ----------

# Helper functions for loading random data and such
# Transformation matrix
def compute_T(anglex, angley, anglez, x, y, z):
    R = expso3(np.array([anglex, angley, anglez]))
    T = np.array([x,y,z])
    temp1 = np.concatenate((R.reshape(3,3), T.reshape((3,1))), axis=1)
    temp2 = np.concatenate((temp1, np.array([[0,0,0,1]]).reshape((1,4))))
    return temp2

def new_compute_T(anglex, angley, anglez, x, y, z):
    Rx = computeRx(anglex)
    Ry = computeRy(angley)
    Rz = computeRz(anglez)
    R = Rx @ Ry @ Rz
    T = np.array([x,y,z])
    temp1 = np.concatenate((R.reshape(3,3), T.reshape((3,1))), axis=1)
    temp2 = np.concatenate((temp1, np.array([[0,0,0,1]]).reshape((1,4))))
    return temp2

def computeRx(anglex):
    return np.array([[1,0,0],[0,np.cos(anglex), -np.sin(anglex)], [0, np.sin(anglex), np.cos(anglex)]])
def computeRy(angley):
    return np.array([[np.cos(angley), 0, np.sin(angley)],[0,1,0], [-np.sin(angley),0,  np.cos(angley)]])
def computeRz(anglez):
    return np.array([[np.cos(anglez), -np.sin(anglez), 0], [np.sin(anglez), np.cos(anglez), 0],[0,0,1]])

# Camera matrix
def compute_C(fu, fv, u0, v0):
    return np.array([
        [  fu,   0, u0],
        [    0, fv, v0],
        [    0,  0,  1]
    ])



# Error functions for finding errors in rotation and translation
def angular_distance_mat(R_true, R_calc):
    R_inc = R_true @ R_calc.T
    return np.degrees(np.arccos(( (np.trace(R_inc) - 1 ) /2 )))

def translation_error(t_true, t_calc):
    return np.linalg.norm(t_true-t_calc)