"""
This is the EPnP+GNC implementation used in the master's thesis by Runar Jåtun.

I have used some from my project, but I have rewritten a lot to make the code
work more like the original matlab code from Lepetit. I have also tried to comment 
the code better in order to make it more understandable for the reader (and myself).


The code is supposed to run like this:

    from epnp import EPnP
    epnp = EPnP()
    epnp.load_data(xh_w, pix, Tr, Ca)
    epnp.compute_epnp(GN = True/False, GNC = True/False):
    Rt_best = epnp.Rt_best

There are many analytical functionalities implemented in this code, so there is 
much to learn about EPnP+GNC in this code.

This also means that the efficiency of the code is not very good. Using a better 
implementation and having less analytics would improve the efficiency I believe.


Point variable definition:
    xh_w    -> homogenous world points
    x_w     -> cartesian  world points
    xh_c    -> homogenous camera points
    x_c     -> cartesian  camera points

    ch_w    -> homogenous world control points
    c_w     -> cartesian  world control points
    ch_c    -> homogenous camera control points
    c_c     -> cartesian  camera control points
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import open3d as o3d
import time

from utility import *

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

# Defining the method using a class
class EPnP:
    # Initializing the class
    def __init__(self) -> None:
        # Initializing analytical tools for analytic use
        self.beta_optim = False
        self.gnc_optim = False
        self.gnc_parameters = False
        self.Rt_choose = "Trans"


    # ---------- Calculation of EPnP ------------------------- 
    # One function that can do the entire EPnP, with options to use GN and GNC
    # This function has tools for analyzing the results
    def compute_epnp(self, GN = True, GNC = True):
        # Timing computation
        timing_start = time.time_ns()

        # Initial calculation using basic EPnP
        self.alpha  = self.compute_alpha()
        self.M      = self.compute_M()
        self.K      = self.compute_K()
        self.L_6_10 = self.compute_L_6_10(self.ch_w)

        self.betas = self.compute_betas()
        self.compute_Xi()

        self.c_c1, self.x_c1, self.sc_1 = self.compute_norm_sign_scaling_factor(self.X1, self.x_w)
        self.c_c2, self.x_c2, self.sc_2 = self.compute_norm_sign_scaling_factor(self.X2, self.x_w)
        self.c_c3, self.x_c3, self.sc_3 = self.compute_norm_sign_scaling_factor(self.X3, self.x_w)

        self.Rt_1 = self.getRotT(self.x_w, self.x_c1)
        self.Rt_2 = self.getRotT(self.x_w, self.x_c2)
        self.Rt_3 = self.getRotT(self.x_w, self.x_c3)

        self.Rt_pre_opt, self.err_pre_opt, self.x_c_pre_opt, self.best_rot_idx = \
            self.best_trans(self.Rt_1, self.Rt_2, self.Rt_3, self.x_c1, self.x_c2, self.x_c3)

        # Saving best result so far
        self.err_best = self.err_pre_opt
        self.Rt_best = self.Rt_pre_opt
        self.x_c_best = self.x_c_pre_opt

        # Timing without optimization
        timing_no_opt = time.time_ns() 

        # Running GN optimization if selected
        if GN:
            self.x_c_GN, self.Rt_GN, self.err_GN = self.Gauss_Newton_Optim(self.K, self.betas, np.array([self.sc_1, self.sc_2, self.sc_3]))
            # If GN gives better result -> save as best
            if self.err_GN < self.err_pre_opt:
                self.err_best = self.err_GN
                self.Rt_best = self.Rt_GN
                self.x_c_best = self.x_c_GN
                self.beta_optim = True
        timing_GN = time.time_ns() # Timing of GN optimization 

        # Using opencv to do calculations
        # Changing values to fit opencv requirements
        cvpix = np.array(self.pix[:,:2])
        cvx = np.array(self.xh_w[:,:3])
        dist_coeffs = np.zeros((4,1))

        # Basic EPnP using opencv
        timing_opencv_epnp_start = time.time_ns()
        success, rotation_vector, translation_vector = cv.solvePnP(cvx, cvpix, self.C, dist_coeffs, flags=cv.SOLVEPNP_EPNP)
        rotation_matrix_ans = np.zeros((3,3))
        cv.Rodrigues(rotation_vector, rotation_matrix_ans)
        CV_pnp = np.hstack((rotation_matrix_ans, translation_vector))
        self.Rt_CV_EPnP = CV_pnp
        timing_opencv_epnp_end = time.time_ns()

        # Basic SQPnP using opencv
        timing_opencv_sqpnp_start = time.time_ns()
        success, rotation_vector, translation_vector = cv.solvePnP(cvx, cvpix, self.C, dist_coeffs, flags=cv.SOLVEPNP_SQPNP)
        rotation_matrix_ans = np.zeros((3,3))
        cv.Rodrigues(rotation_vector, rotation_matrix_ans)
        CV_pnp = np.hstack((rotation_matrix_ans, translation_vector))
        self.Rt_CV_SQPnP = CV_pnp
        timing_opencv_sqpnp_end = time.time_ns()

        # Solving using RANSAC and EPnP
        timing_opencv_ransac_start = time.time_ns()
        success, rotation_vector, translation_vector, _ = cv.solvePnPRansac(cvx, cvpix, self.C, dist_coeffs, flags=cv.SOLVEPNP_EPNP)
        rotation_matrix_ans = np.zeros((3,3))
        cv.Rodrigues(rotation_vector, rotation_matrix_ans)
        CV_pnp = np.hstack((rotation_matrix_ans, translation_vector))
        self.Rt_CV_Ransac = CV_pnp
        timing_opencv_ransac_end = time.time_ns()


        # Running GNC with TLS if selected
        timing_GNC_start = time.time_ns()
        if GNC:
            # Initialization of GNC parameters
            if self.gnc_parameters: 
                if self.Rt_choose == "Trans":
                    Rt_init = compute_T(0,0,0,0,0,4)[:3,:] # This is default
                elif self.Rt_choose == "OpenCV-EPnP":
                    Rt_init = self.Rt_CV_EPnP
                elif self.Rt_choose == "Eye":
                    Rt_init = np.eye(4, dtype=float)[:3,:]
                elif self.Rt_choose == "EPnP":
                    Rt_init = self.Rt_best
                elif self.Rt_choose == "Prev_best":
                    Rt_init = compute_T(np.pi/8,  np.pi/6,  np.pi/4,  -0.314, 0.294, 13.96)[:3,:]
            else:
                Rt_init = compute_T(0,0,0,0,0,4)[:3,:]


            sol = (Rt_init @ self.xh_w.T).T
            sol = sol/sol[:,2].reshape(self.n, 1)
            sol = (self.C @ sol.T).T
            self.r = np.zeros(self.n)
            for i in range(self.n):
                self.r[i] = np.linalg.norm(sol[i] - self.pix[i])**2
            r0_max = np.max(self.r)

            # Eps decides how big the threshold for cutout is, the same as \Hat{c} in the thesis
            # According to Yang this is set as max expected error, in this case of the pixel reprojection error
            if not self.gnc_parameters: self.eps = 1000 #default value
            
            # Same for all iterations, using values from Yang
            mu_update = 1.4
            max_iter = 1000

            # Initializing the weights
            self.w = np.ones(self.n)
            mu = np.square(self.eps**2 / (2*r0_max**2 - self.eps**2))

            last_iter = []          # Used to check if GNC has converged
            self.iterations = 0     # Analytical toll

            # Starting GNC iteration step
            for i in range(max_iter):
                self.iterations += 1
                last_iter.append(np.sum(self.w))

                # The calculation of the transformation matrix Tr using weights
                self.M = self.compute_M()
                self.M = np.diag(np.repeat(np.sqrt(self.w),2)) @ self.M

                self.K      = self.compute_K()
                self.L_6_10 = self.compute_L_6_10(self.K)

                self.betas = self.compute_betas()
                self.compute_Xi()

                self.c_c1_gnc, self.x_c1_gnc, self.sc_1_gnc = self.compute_norm_sign_scaling_factor(self.X1, self.x_w)
                self.c_c2_gnc, self.x_c2_gnc, self.sc_2_gnc = self.compute_norm_sign_scaling_factor(self.X2, self.x_w)
                self.c_c3_gnc, self.x_c3_gnc, self.sc_3_gnc = self.compute_norm_sign_scaling_factor(self.X3, self.x_w)

                self.Rt_1_gnc = self.getRotT(self.x_w, self.x_c1_gnc)
                self.Rt_2_gnc = self.getRotT(self.x_w, self.x_c2_gnc)
                self.Rt_3_gnc = self.getRotT(self.x_w, self.x_c3_gnc)

                self.Rt_GNC, self.err_GNC, self.x_c_GNC, self.best_rot_idx_GNC = \
                    self.best_trans(self.Rt_1_gnc, self.Rt_2_gnc, self.Rt_3_gnc, self.x_c1_gnc, self.x_c2_gnc, self.x_c3_gnc)

                # Loss function
                for j in range(self.n):
                    sol = (self.Rt_GNC @ self.xh_w.T).T
                    sol = sol/sol[:,2].reshape(self.n, 1)
                    sol = (self.C @ sol.T).T
                    self.r[j] = np.linalg.norm(sol[j] - self.pix[j])**2
                    self.w[j] = self.w_from_r(self.r[j], self.eps, mu)
                mu *= mu_update

                # Checking for convergence
                if i > 1:
                    if np.sum(self.w) == last_iter[i]:
                        break
            
            # Calculating reprojection error without outliers
            index = 1 - self.w
            placement = np.where(index == 1)
            self.xh_w = np.delete(self.xh_w, placement, axis=0)
            self.x_w = self.xh_w[:,:3]
            self.pix = np.delete(self.pix, placement, axis=0)
            self.n = self.xh_w.shape[0]
            err_outlier_removed = self.reprojection_error(self.Rt_GNC)

            # Running EPnP on only inliers 
            self.alpha  = self.compute_alpha()
            self.M      = self.compute_M()
            self.K      = self.compute_K()
            self.L_6_10 = self.compute_L_6_10(self.ch_w)

            self.betas = self.compute_betas()
            self.compute_Xi()

            self.c_c1_out_rem, self.x_c1_out_rem, self.sc_1_out_rem = self.compute_norm_sign_scaling_factor(self.X1, self.x_w)
            self.c_c2_out_rem, self.x_c2_out_rem, self.sc_2_out_rem = self.compute_norm_sign_scaling_factor(self.X2, self.x_w)
            self.c_c3_out_rem, self.x_c3_out_rem, self.sc_3_out_rem = self.compute_norm_sign_scaling_factor(self.X3, self.x_w)

            if not self.sc_1_out_rem == 0: self.Rt_1_out_rem = self.getRotT(self.x_w, self.x_c1_out_rem) 
            else: self.Rt_1_out_rem = np.eye(3,4)

            if not self.sc_2_out_rem == 0: self.Rt_2_out_rem = self.getRotT(self.x_w, self.x_c2_out_rem) 
            else: self.Rt_2_out_rem = np.eye(3,4)

            if not self.sc_3_out_rem == 0: self.Rt_3_out_rem = self.getRotT(self.x_w, self.x_c3_out_rem) 
            else: self.Rt_3_out_rem = np.eye(3,4)

            self.Rt_post_gnc, self.err_post_gnc, self.x_c_post_gnc, self.best_rot_idx_post_gnc = \
                self.best_trans(self.Rt_1_out_rem, self.Rt_2_out_rem, self.Rt_3_out_rem, self.x_c1_out_rem, self.x_c2_out_rem, self.x_c3_out_rem)

            # self.n = self.n_init
            # This is a bit funky, plotting is difficult when n is changed

            # Here it could be possible to use GN optimization on the answer from GNC
            # This would need a sufficient implementation of GN, which unfortunately is not the case in this project

            # Changing new best values to GNC values if they are more correct
            if self.err_GNC < self.err_best or err_outlier_removed < self.err_best:
                self.err_best = self.err_post_gnc
                self.Rt_best = self.Rt_post_gnc
                self.best_rot_idx = self.best_rot_idx_post_gnc
                self.x_c_best = self.x_c_post_gnc
                self.gnc_optim = True
            else:
                self.n = self.n_init # Lazy error handling
        
        # Timing at the end
        timing_GNC_end = time.time_ns()
        self.timing_no_opt =  timing_no_opt - timing_start
        self.timing_GN =  timing_GN - timing_start
        self.timing_GNC = timing_GNC_end - timing_GNC_start

        self.timing_opencv_epnp = timing_opencv_epnp_end-timing_opencv_epnp_start
        self.timing_opencv_sqpnp = timing_opencv_sqpnp_end-timing_opencv_sqpnp_start
        self.timing_opencv_ransac = timing_opencv_ransac_end-timing_opencv_ransac_start


        # This is also a function that depends on the size of n, so it might result in wierd plotting 
        self.compute_pixels()

        

    # ---------- Data loading methods ----------------------
    # Loading a given dataset with given Transformation matrix
    # Function used mostly early in the research, not used in the thesis
    def load_set_points(self, Tr, Ca, xh_w, noise = False): 

        # Transformation matrix, Camera matrix and number of correspondences(points)
        self.T = Tr
        self.C = Ca
        self.n = xh_w.shape[0]

        # Camera parameters
        self.fu = Ca[0,0]
        self.fv = Ca[1,1]
        self.u0 = Ca[0,2]
        self.v0 = Ca[1,2]
        
        # Reference points
        self.xh_w = xh_w
        self.x_w = self.xh_w[:,:3]
        self.xh_c = (self.T @ self.xh_w.T).T
        self.x_c = self.xh_c[:,:3]

        # Defining control points and calculating rho
        self.define_control_points()

        # Normalized reference coordinates
        self.sn = (np.eye(3,4) @ self.T @ self.xh_w.T).T
        self.snorm = self.sn/self.sn[:,2].reshape((self.n,1))

        # Reference points as pixels (i.e. taking a picture)
        self.pix = (self.C @ self.snorm.T).T
        # self.pix = np.rint((self.C @ self.snorm.T).T)
        self.pix_true = self.pix.copy()

        # Creating noise in pixels if specified - False by default
        # TODO: Find a better way of doing this
        if noise:
            a = 2
            smin = -10
            smax =  10
            for i, p in enumerate(self.pix):
                if i % a == 0:
                    p[0] += np.random.randint(smin, smax) 
                    p[1] += np.random.randint(smin, smax)

    # Loading all parameters already set
    # This function is used in the thesis, as pixels are calculated before loading
    def load_data(self, xh_w, pix, Tr, Ca):
        # Transformation matrix, Camera matrix and number of correspondences(points)
        self.T = Tr
        self.C = Ca
        self.n = xh_w.shape[0]
        self.n_init = self.n

        # Camera parameters
        self.fu = Ca[0,0]
        self.fv = Ca[1,1]
        self.u0 = Ca[0,2]
        self.v0 = Ca[1,2]

        # Reference points
        self.xh_w = xh_w
        self.x_w = self.xh_w[:,:3]
        self.xh_c = (self.T @ self.xh_w.T).T
        self.x_c = self.xh_c[:,:3]

        # Defining control points and calculating rho
        self.define_control_points()

        # Normalized reference points
        self.sn = (np.eye(3,4) @ self.T @ self.xh_w.T).T
        self.snorm = self.sn/self.sn[:,2].reshape((self.n,1))

        # Pixel values
        self.pix = pix
        self.pix_true = (self.C @ self.snorm.T).T # Pixels without noise




    # ---------- Defining funtions used -----------------------
    
    # Defning control points, one is centroid, three are principal axes
    def define_control_points(self):
        c0 = np.mean(self.x_w, axis=0)

        M = (self.x_w - np.mean(self.x_w))
        cov = np.cov(M.T)
        _, pcs = np.linalg.eig(cov)

        c1 = pcs[0,:]
        c2 = pcs[1,:]
        c3 = pcs[2,:]
        self.c_w = np.array([c0,c1,c2,c3])
        self.ch_w = np.hstack((self.c_w, np.ones((4,1))))
        self.rho = self.compute_rho()

    # Computing alpha - Following procedure from rapport 
    def compute_alpha(self):
        X = self.xh_w.T
        C = self.ch_w.T
        return (np.linalg.inv(C) @ X).T
    

    # Computing M matrix from Mx = 0
    def compute_M(self):
        M = np.empty((2*self.n, 12))
        for i in range(self.n):
            M[i*2,:]= [
                self.alpha[i, 0] * self.fu, 0, self.alpha[i, 0] * (self.u0 - self.pix[i, 0]),
                self.alpha[i, 1] * self.fu, 0, self.alpha[i, 1] * (self.u0 - self.pix[i, 0]),
                self.alpha[i, 2] * self.fu, 0, self.alpha[i, 2] * (self.u0 - self.pix[i, 0]),
                self.alpha[i, 3] * self.fu, 0, self.alpha[i, 3] * (self.u0 - self.pix[i, 0])
            ]
            M[i*2+1,:] = [
                0, self.alpha[i, 0] * self.fv, self.alpha[i, 0] * (self.v0 - self.pix[i, 1]),
                0, self.alpha[i, 1] * self.fv, self.alpha[i, 1] * (self.v0 - self.pix[i, 1]),
                0, self.alpha[i, 2] * self.fv, self.alpha[i, 2] * (self.v0 - self.pix[i, 1]),
                0, self.alpha[i, 3] * self.fv, self.alpha[i, 3] * (self.v0 - self.pix[i, 1])
            ]
        return M


    # Computing MtM -> K from 3.3 in Lepetit
    # Essentially returns the 4 lowest eigenvectors of MtM
    def compute_K(self):
        MtM = self.M.T @ self.M
        eig_val, eig_vec = np.linalg.eig(MtM)
        # These are not sorted, and we only want the eigenvectors with lowest eigenvalues
        # In Lepetit they say that only the four lowest are necessary
        sorted_val = np.argsort(eig_val)
        return eig_vec[:, sorted_val[:4]]


    # Compute rho from equation (13) in Lepetit
    # Lb = p    (13)
    def compute_rho(self):
        return np.array([
            np.linalg.norm(self.ch_w[0,:3]-self.ch_w[1,:3])**2,
            np.linalg.norm(self.ch_w[0,:3]-self.ch_w[2,:3])**2,
            np.linalg.norm(self.ch_w[0,:3]-self.ch_w[3,:3])**2,
            np.linalg.norm(self.ch_w[1,:3]-self.ch_w[2,:3])**2,
            np.linalg.norm(self.ch_w[1,:3]-self.ch_w[3,:3])**2,
            np.linalg.norm(self.ch_w[2,:3]-self.ch_w[3,:3])**2
        ])
    

    # Computing L matrix from Lb = p
    # Did not understand this, so this is on "loan" from EPnP Python. It is very 
    # similar to the original code from Lepetit
    def compute_L_6_10(self, Kernel_given, given = False):
        L_6_10 = np.zeros((6,10), dtype=np.complex)

        if given:
            kernel = Kernel_given
        else:
            kernel = np.array([self.K.T[3], self.K.T[2], self.K.T[1], self.K.T[0]]).T

        v = []
        for i in range(4):
            v.append(kernel[:, i])

        dv = []

        for r in range(4):
            dv.append([])
            for i in range(3):
                for j in range(i+1, 4):
                    dv[r].append(v[r][3*i:3*(i+1)]-v[r][3*j:3*(j+1)])

        index = [
            (0, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3)
            ]

        for i in range(6):
            j = 0
            for a, b in index:
                L_6_10[i, j] = dv[a][i] @ dv[b][i].T
                if a != b:
                    L_6_10[i, j] *= 2
                j += 1


        return L_6_10
    

    # Calculating lesser matrices
    def compute_L_6_6(self):
        return self.L_6_10[:, (2, 4, 7, 5, 8, 9)]
    def compute_L_6_3(self):
        return self.L_6_10[:, (5,8,9)]

    
    # Function (11) from Lepetit
    # Was stated in Lepetit that this could be used to further estimate the common beta
    def beta_computation_function(self, v):
        num = 0
        den = 0
        for i in range(4):
            for j in range(4):
                v_dist = np.linalg.norm(v[i*3:i*3+3] - v[j*3:j*3+3])
                c_dist = np.linalg.norm(self.c_w[i,:] - self.c_w[j,:])
                num += v_dist*c_dist
                den += v_dist**2
        beta1 = num/den
        return beta1

    def beta_computation_dimention(self, dim_ker, betas):
        if dim_ker == 1:
            v = self.K[:,0]
            return self.beta_computation_function(v) , v
        if dim_ker == 2:
            v = betas[1,0]*self.K[:,0] + betas[1,1]*self.K[:,1]
            return self.beta_computation_function(v) , v
        if dim_ker == 3:
            v = betas[2,0]*self.K[:,0] + betas[2,1]*self.K[:,1] + betas[2,2]*self.K[:,3]
            return self.beta_computation_function(v) , v


    # Computing betas
    def compute_betas(self):
        L_6_6 = self.compute_L_6_6()
        L_6_3 = self.compute_L_6_3()

        betas = np.zeros((3,4), dtype=np.complex)

        betas[0,0], self.v1 = self.beta_computation_dimention(1, betas)

        temp = np.matmul(np.linalg.pinv(L_6_3), self.rho)
        betas[1,0] = np.sqrt(abs(temp[0]))
        betas[1,1] = np.sqrt(abs(temp[2]))* np.sign(temp[1])*np.sign(temp[0])
        betas[1,2], self.v2 = self.beta_computation_dimention(2, betas)

        temp = np.matmul(np.linalg.pinv(L_6_6), self.rho)
        betas[2,0] = np.sqrt(abs(temp[0]))
        betas[2,1] = np.sqrt(abs(temp[3]))* np.sign(temp[1])*np.sign(temp[0])
        betas[2,2] = np.sqrt(abs(temp[5]))* np.sign(temp[2])*np.sign(temp[0])
        betas[2,3], self.v3 = self.beta_computation_dimention(3, betas)
        return betas
    

    # Computing the X in the solution x = sum(beta*v) for case 1 -> 3
    def compute_Xi(self):
        self.X1 = self.betas[0,0] * self.v1
        self.X2 = self.betas[1,2] * self.v2
        self.X3 = self.betas[2,3] * self.v3


    # Calculating scaling factor along with vectors, mostly from original code
    # X is the beta * v, basically
    def compute_norm_sign_scaling_factor(self, X, Xworld):
        Cc = np.zeros((4, 3), dtype=np.complex)
        for i in range(4):
            Cc[i,:] = X[i*3:i*3+3] 

        Xc = self.alpha @ Cc

        centr_w = np.mean(Xworld, axis=0)
        centroid_w = np.tile(centr_w.reshape((1, 3)), (self.n, 1))
        tmp1 = Xworld.reshape((self.n, 3)) - centroid_w
        dist_w = np.sqrt(np.sum(tmp1 ** 2, axis=1))
        
        centr_c = np.mean(np.array(Xc), axis=0)
        centroid_c = np.tile(centr_c.reshape((1, 3)), (self.n, 1))
        tmp2 = Xc.reshape((self.n, 3)) - centroid_c
        dist_c = np.sqrt(np.sum(tmp2 ** 2, axis=1))
        dist_c = dist_c.reshape((dist_c.shape[0],1))
        
        temp_mat = dist_c.T @ dist_c
        if np.real(np.linalg.det(temp_mat)) == 0:
            return np.zeros((4, 3)), np.zeros((self.n, 3)), 0

        sc = 1/(np.linalg.inv(temp_mat) @ dist_c.T @ dist_w)

        c_c = Cc/sc
        x_c = self.alpha @ c_c
        
        neg_z = x_c[:,-1] < 0
        if np.sum(neg_z) >= 1:
            sc = -sc
            x_c = x_c * -1
        return c_c, x_c, sc


    # Calculating the rotation matrix and the translation, and then combining
    # The method for this is basically Procrustes Method
    def getRotT(self, wpts, cpts):
        wcent = np.tile(np.mean(wpts, axis=0).reshape((1, 3)), (self.n, 1))
        ccent = np.tile(np.mean(cpts, axis=0).reshape((1, 3)), (self.n, 1))
        self.wpts = (wpts.reshape((self.n, 3)) - wcent)
        self.cpts = (cpts.reshape((self.n, 3)) - ccent)

        # If GNC has failed, cpts will be all zeros
        # This is an attempt of error handling
        if np.sum(np.real(cpts.any())) == 0:
            print("Error")
            return np.eye(3, 4)
        
        M = self.cpts.T @ self.wpts
        
        U, S, Vt = np.linalg.svd(M)
        # R = U @ Vt

        # Alternative method, from lectures at NTNU
        # Do not know which is more accurate
        intermediate = np.diag([1,1,np.linalg.det( Vt.T @ U.T)])
        R = (Vt.T @ intermediate @ U.T).T

        # Should have used the umeyama algorithm
        

        if np.linalg.det(R) < 0:
            R = - R
        T = ccent[0].T - R @  wcent[0].T

        Rt = np.concatenate((R.reshape((3, 3)), T.reshape((3, 1))), axis=1)
        
        return Rt
    
    # Reprojection error function
    def reprojection_error(self, Rt_calc):
        sol = (Rt_calc @ self.xh_w.T).T
        sol = sol/sol[:,2].reshape(self.n, 1)
        sol = (self.C @ sol.T).T
        res = np.linalg.norm(sol - self.pix)**2
        return res/self.n

    # Calculating the best transformation matrix from the 3 cases
    def best_trans(self, Rt1, Rt2, Rt3, xc1, xc2, xc3):
        err1 = self.reprojection_error(Rt1)
        err2 = self.reprojection_error(Rt2)
        err3 = self.reprojection_error(Rt3)
        if err1 < err2 and err1 < err3:
            return np.real(Rt1), err1, xc1, 1
        elif err2 < err1 and err2 < err3:
            return np.real(Rt2), err2, xc2, 2
        elif err3 < err1 and err3 < err2:
            return np.real(Rt3), err3, xc3, 3
        else:
            return np.real(Rt1), err1, xc1, 1 # Attempt at error handling


    # --------- Functions used in Gauss-Newton Optimizing ----------
    def Gauss_Newton_Optim(self, K, betas, sc):
        kernel = np.array([K.T[3], K.T[2], K.T[1], K.T[0]]).T
        
        # if self.best_rot_idx == 1:
        scaled_betas = betas * sc[0]
        self.x_c_GN_1, self.Rt_GN_1, self.err_GN_1 = self.optimize_betas_gauss_newton(kernel, scaled_betas[0,:])
        # elif self.best_rot_idx == 2:
        scaled_betas = betas * sc[1]
        self.x_c_GN_2, self.Rt_GN_2, self.err_GN_2 = self.optimize_betas_gauss_newton(kernel, scaled_betas[1,:])
        # elif self.best_rot_idx == 3:
        scaled_betas = betas * sc[2]
        self.x_c_GN_3, self.Rt_GN_3, self.err_GN_3 = self.optimize_betas_gauss_newton(kernel, scaled_betas[2,:])

        if self.err_GN_1 < self.err_GN_2 and self.err_GN_1 < self.err_GN_3:
            return self.x_c_GN_1, self.Rt_GN_1, self.err_GN_1
        elif self.err_GN_2 < self.err_GN_1 and self.err_GN_2 < self.err_GN_3:
            return self.x_c_GN_2, self.Rt_GN_2, self.err_GN_2
        elif self.err_GN_3 < self.err_GN_1 and self.err_GN_3 < self.err_GN_2:
            return self.x_c_GN_3, self.Rt_GN_3, self.err_GN_3

    def optimize_betas_gauss_newton(self, kernel, beta0):

        n = beta0.shape[0]
        beta_opt = self.gauss_newton(kernel, beta0)

        X = np.zeros((12,1), dtype=np.complex)
        X = beta_opt[0]*kernel[:,0] + beta_opt[1]*kernel[:,1] + beta_opt[2]*kernel[:,2] + beta_opt[3]*kernel[:,3]
        
        
        Cc = (X.reshape((4,3)))
        # In EPnP matlab there is a check of the sign of the determinant here
        s_Cw = self.sign_determinant(self.c_w)
        s_Cc = self.sign_determinant(Cc)
        Cc = Cc*(s_Cw/s_Cc)

        x_c_GN = self.alpha @ Cc

        Rt_GN = self.getRotT(self.x_w, x_c_GN)

        Rt_GN = np.real(Rt_GN)

        err_GN = self.reprojection_error(Rt_GN)

        return x_c_GN, Rt_GN, err_GN

    def sign_determinant(self, C):
        c0 = C[3,:].T
        c1 = C[0,:].T
        c2 = C[1,:].T
        c3 = C[2,:].T

        v1 = c1 - c0
        v2 = c2 - c0
        v3 = c3 - c0

        M = np.array([v1, v2, v3])
        detM = np.linalg.det(M)
        return np.sign(detM)

    def gauss_newton(self, kernel, beta0):
        L = self.compute_L_6_10(kernel, given=True)
        current_betas = beta0.reshape((4,1))

        n_iterations = 5
        for k in range(n_iterations):
            A, b = self.compute_A_b_GN(current_betas, L)
            AtA = A.T @ A
            if np.linalg.det(AtA) != 0:
                dbeta = np.linalg.inv(AtA) @ A.T @ b
            else:
                dbeta = np.zeros((4,1))
            current_betas += dbeta
            error = b.T @ b
        return current_betas

    def compute_A_b_GN(self, cb, L):
        A = np.zeros((6,4), dtype=np.complex)
        b = np.zeros((6,1), dtype=np.complex)

        B = np.array([
            [cb[0]*cb[0]],
            [cb[0]*cb[1]],
            [cb[1]*cb[1]],
            [cb[0]*cb[2]],
            [cb[1]*cb[2]],
            [cb[2]*cb[2]],
            [cb[0]*cb[3]],
            [cb[1]*cb[3]],
            [cb[2]*cb[3]],
            [cb[3]*cb[3]]
        ])

        for i in range(6):
            A[i,0] = 2*cb[0]*L[i, 0] + cb[1]*L[i, 1] + cb[2]*L[i, 3] + cb[3]*L[i, 6]
            A[i,1] = cb[0]*L[i, 1] + 2*cb[1]*L[i, 2] + cb[2]*L[i, 4] + cb[3]*L[i, 7]
            A[i,2] = cb[0]*L[i, 3] + cb[1]*L[i, 4] + 2*cb[2]*L[i, 5] + cb[3]*L[i, 8]
            A[i,3] = cb[0]*L[i, 6] + cb[1]*L[i, 7] + cb[2]*L[i, 8] + 2*cb[3]*L[i, 9]

            b[i] = self.rho[i] - L[i,:].reshape((1,10)) @ B.reshape((10,1))
        
        return A, b


    # ------------ Functions used in GNC --------------
    # Defining eps (\Bar{c}) if one does not want to use default value (1000)
    def define_gnc_parameters(self, eps):
        self.eps = eps
        self.gnc_parameters = True

    # Computing the weights
    def w_from_r(self, r, eps, mu):
        if abs(r) < eps * np.sqrt(mu/(mu+1)):
            w = 1
        elif abs(r) > eps * np.sqrt((mu+1)/mu):
            w = 0
        else:
            w = (eps * np.sqrt(mu*(mu+1)))/abs(r) - mu
        return w


    # ---------- Other functions --------------
    # Computing the pixels from the calculated transformation matrix
    def compute_pixels(self):
        snorm = self.x_c_best*(1/self.x_c_best[:,2].reshape((self.n,1)))
        self.pix_calc = np.real(np.rint((self.C @ snorm.T).T))
        self.point_calc = np.real(self.x_c_best)

        # if self.best_rot_idx == 1:
        #     snorm_1 = self.x_c1*(1/self.x_c1[:,2].reshape((self.n,1)))
        #     self.pix_calc = np.real(np.rint((self.C @ snorm_1.T).T))
        #     self.point_calc = np.real(self.x_c1)
        # elif self.best_rot_idx == 2:
        #     snorm_2 = self.x_c2*(1/self.x_c2[:,2].reshape((self.n,1)))
        #     self.pix_calc = np.real(np.rint((self.C @ snorm_2.T).T))
        #     self.point_calc = np.real(self.x_c2)
        # elif self.best_rot_idx == 3:
        #     snorm_3 = self.x_c3*(1/self.x_c3[:,2].reshape((self.n,1)))
        #     self.pix_calc = np.real(np.rint((self.C @ snorm_3.T).T))
        #     self.point_calc = np.real(self.x_c3)

    # Plotting pixels using Open3D
    # This function does not work when GNC is run, because the n is changed
    def plot_set_pixels(self, version, x_c):
        # snorm = x_c*(1/x_c[:,2].reshape((self.n,1))) # Real
        snorm = x_c[:self.n_init,:]*(1/x_c[:self.n_init,2].reshape((self.n_init,1)))
        pix_calc = np.real(np.rint((self.C @ snorm.T).T))
        point_calc = np.real(x_c)

        # geometry array
        geometries = []

        # Colors to be used
        color1 = np.array([0.0, 0.0, 1.0])
        color2 = np.array([255,165,0])/255

        # Points in camera
        pix_corners = np.array([
            [0, 0, 1],
            [self.u0*2, 0, 1],
            [0, self.v0*2, 1],
            [self.u0*2, self.v0*2, 1],
            [0, 0, 0]
        ]).T

        if version == "3D":
            # 3D picture camera corners
            Cinv = np.linalg.inv(self.C)
            snorm_corners = (Cinv @ pix_corners).T
            lines = np.array([[4,0], [4,1], [4,2], [4,3], [0,1], [1,3], [2,3], [2,0]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(snorm_corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            geometries.append(line_set)

            # Actual points
            pcd_true = o3d.geometry.PointCloud()
            pcd_true.points = o3d.utility.Vector3dVector(self.x_c)
            pcd_true.paint_uniform_color(color1)
            geometries.append(pcd_true)
            # downpcd_true = pcd_true.voxel_down_sample(voxel_size=0.05)
            # geometries.append(downpcd_true)

            # Calculated Pixeld
            pcd_epnp = o3d.geometry.PointCloud()
            pcd_epnp.points = o3d.utility.Vector3dVector(point_calc)
            pcd_epnp.paint_uniform_color(color2)
            geometries.append(pcd_epnp)
            # downpcd_calc = pcd_epnp.voxel_down_sample(voxel_size=0.06)
            # geometries.append(downpcd_calc)

            # Lines connecting corresponding points
            vector = np.vstack((self.x_c, point_calc))
            lines_p = np.array([[i, self.n_init+i] for i in range(self.n_init)])
            line_set_n = o3d.geometry.LineSet()
            line_set_n.points = o3d.utility.Vector3dVector(vector)
            line_set_n.lines = o3d.utility.Vector2iVector(lines_p)
            geometries.append(line_set_n)

            # Axis
            plot_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0])
            geometries.append(plot_axis)
        
        if version == "2D":
            # Picture frame
            pix_corners[:,4] = np.array([[self.u0,self.v0, -self.fu]])
            # lines = np.array([[4,0], [4,1], [4,2], [4,3], [0,1], [1,3], [2,3], [2,0]])
            lines = np.array([[0,1], [1,3], [2,3], [2,0]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(pix_corners.T)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            geometries.append(line_set)

            # Actual pixels
            pcd_true = o3d.geometry.PointCloud()
            pcd_true.points = o3d.utility.Vector3dVector(self.pix)
            # pcd_true.paint_uniform_color(color1)
            geometries.append(pcd_true)

            # Calculated Pixeld
            pcd_epnp = o3d.geometry.PointCloud()
            pcd_epnp.points = o3d.utility.Vector3dVector(pix_calc)
            pcd_epnp.paint_uniform_color(color2)
            geometries.append(pcd_epnp)

            # Lines connecting corresponding points
            vector = np.vstack((self.pix,self.pix_calc))
            lines_p = np.array([[i, self.n_init+i] for i in range(self.n_init)])
            line_set_n = o3d.geometry.LineSet()
            line_set_n.points = o3d.utility.Vector3dVector(vector)
            line_set_n.lines = o3d.utility.Vector2iVector(lines_p)
            geometries.append(line_set_n)

            # Axis
            plot_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=[0, 0, 0])
            geometries.append(plot_axis)

        # Drawing the pixels
        o3d.visualization.draw_geometries(geometries, width=1600, height=900)

    # This function does not work when GNC is run, because the n is changed
    def plot_results_o3d(self, version):
        # geometry array
        geometries = []

        # Colors to be used
        color1 = np.array([0.0, 0.0, 1.0])
        color2 = np.array([255,165,0])/255

        # Points in camera
        pix_corners = np.array([
            [0, 0, 1],
            [self.u0*2, 0, 1],
            [0, self.v0*2, 1],
            [self.u0*2, self.v0*2, 1],
            [0, 0, 0]
        ]).T

        if version == "3D":
            # 3D picture camera corners
            Cinv = np.linalg.inv(self.C)
            snorm_corners = (Cinv @ pix_corners).T
            lines = np.array([[4,0], [4,1], [4,2], [4,3], [0,1], [1,3], [2,3], [2,0]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(snorm_corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            geometries.append(line_set)

            # Actual points
            pcd_true = o3d.geometry.PointCloud()
            pcd_true.points = o3d.utility.Vector3dVector(self.x_c)
            pcd_true.paint_uniform_color(color1)
            # geometries.append(pcd_true)
            downpcd_true = pcd_true.voxel_down_sample(voxel_size=0.05)
            geometries.append(downpcd_true)

            # Calculated Pixeld
            pcd_epnp = o3d.geometry.PointCloud()
            pcd_epnp.points = o3d.utility.Vector3dVector(self.point_calc)
            pcd_epnp.paint_uniform_color(color2)
            # geometries.append(pcd_epnp)
            downpcd_calc = pcd_epnp.voxel_down_sample(voxel_size=0.06)
            geometries.append(downpcd_calc)

            vector = np.vstack((self.x_c,self.point_calc))
            lines_p = np.array([[i, self.n_init+i] for i in range(self.n_init)])
            line_set_n = o3d.geometry.LineSet()
            line_set_n.points = o3d.utility.Vector3dVector(vector)
            line_set_n.lines = o3d.utility.Vector2iVector(lines_p)
            geometries.append(line_set_n)

            # Axis
            plot_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0])
            geometries.append(plot_axis)


        if version == "2D":
            # Picture frame
            pix_corners[:,4] = np.array([[self.u0,self.v0, -self.fu]])
            lines = np.array([[4,0], [4,1], [4,2], [4,3], [0,1], [1,3], [2,3], [2,0]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(pix_corners.T)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            geometries.append(line_set)

            # Actual pixels
            pcd_true = o3d.geometry.PointCloud()
            pcd_true.points = o3d.utility.Vector3dVector(self.pix)
            pcd_true.paint_uniform_color(color1)
            geometries.append(pcd_true)

            # Calculated Pixeld
            pcd_epnp = o3d.geometry.PointCloud()
            pcd_epnp.points = o3d.utility.Vector3dVector(self.pix_calc)
            pcd_epnp.paint_uniform_color(color2)
            geometries.append(pcd_epnp)

            vector = np.vstack((self.pix,self.pix_calc))
            lines_p = np.array([[i, self.n_init+i] for i in range(self.n_init)])
            line_set_n = o3d.geometry.LineSet()
            line_set_n.points = o3d.utility.Vector3dVector(vector)
            line_set_n.lines = o3d.utility.Vector2iVector(lines_p)
            geometries.append(line_set_n)

            # Axis
            plot_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=[0, 0, 0])
            geometries.append(plot_axis)

        # Drawing the pixels
        o3d.visualization.draw_geometries(geometries, width=1600, height=900)

    # Printing function, has not been used in thesis, but was used when writing the code
    def print_results(self, GN = False, GNC = False):
        print("Results:")
        print("Actual Transfomration:")
        print(self.T[:3,:])
        print("\nBest calculated Transfomation:")
        print(self.Rt_best)
        print("\nBeta used:\t", self.best_rot_idx)
        print(f"Best error:\t {self.err_best:.4f}")
        print("n of points:\t", self.n)
        if self.beta_optim: print("GN Opt:\t\t Yes")
        else:               print("GN Opt:\t\t No")
        if self.gnc_optim: print("GNC Opt:\t Yes")
        else:               print("GNC Opt:\t No")

        # if GN:
        print()
        print(f"Error pre GN:\t {self.err_pre_opt:.4f}")
        print(f"Error post GN:\t {self.err_GN:.4f}")

        # if GNC:
        print(f"Error post GNC:\t {self.err_post_gnc:.4f}")
        print()
        print("Iterations:\t", self.iterations)
        print("GNC inliers:\t",np.sum(self.w))
        print("GNC outliers:\t",self.n_init - np.sum(self.w))
        print(f"Percentage:\t {(100 - np.sum(self.w)/self.n_init * 100):.1f}%")
            # print("Initial guess by epnp:\n", self.Rt_pre_opt)

        print(f"Rotation error quat:\t {(angular_distance_mat(self.T[:3,:3], self.Rt_best[:,:3])):.4f}")
        print(f"Translation error quat:\t {(translation_error(self.T[:3,3], self.Rt_best[:,3])):.4f}")