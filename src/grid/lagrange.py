#!/usr/bin/env python2
# Lagrange.py

# system includes
import sys
import os
import math
import numpy as np

# local includes
#from point import Point
#from fixation import Fixation
#from grid import Grid
#from aoi import AOI

class Lagrange:

  #        n: number of calibration points
  #        S: the actual (known) locations of the n calibration points
  #    X_bar: centroid of observed points (samples) about each clb point
  #        X: the nx2 quadratic X matrix that computes the fit
  #        B: the solution matrix (6,2)
  def __init__(self,w=1600,h=900,n=10):
    self.n = n
    self.S = np.matrix(np.zeros((n,2),dtype=float))
    self.X_bar = np.matrix(np.zeros((n,2),dtype=float))
    self.X = np.matrix(np.zeros((n,6),dtype=float))
    self.Y = np.matrix(np.zeros((n,2),dtype=float))
    self.B = np.matrix(np.zeros((6,2),dtype=float))

    # initially defined on a 1600x900 image, screen resolution may be different
    self.w = 1600.0
    self.h = 900.0
#   self.w = 2048.0
#   self.h = 1536.0

    self.S[0,0] = 0.5
    self.S[0,1] = 0.5

    self.S[1,0] = 0.24375
    self.S[1,1] = 0.7266

    self.S[2,0] = 0.7550
    self.S[2,1] = 0.7266

    self.S[3,0] = 0.7550
    self.S[3,1] = 0.2766

    self.S[4,0] = 0.24375
    self.S[4,1] = 0.2766

    self.S[5,0] = 0.5
    self.S[5,1] = 0.7266

    self.S[6,0] = 0.7550
    self.S[6,1] = 0.5

    self.S[7,0] = 0.5
    self.S[7,1] = 0.2766

    self.S[8,0] = 0.24375
    self.S[8,1] = 0.5

    self.S[9,0] = 0.5
    self.S[9,1] = 0.5

    if w != self.w or h != self.h:
      for i in range(self.n - 1):
        # map [0,1] -> [-1,1]
        self.S[i,0] = self.S[i,0] * 2.0 - 1.0
        self.S[i,1] = self.S[i,1] * 2.0 - 1.0

        # scale point by ratio of images
        self.S[i,0] = self.S[i,0] * float(self.w)/float(w)
        self.S[i,1] = self.S[i,1] * float(self.h)/float(h)

        # map [-1,1] -> [0,1]
        self.S[i,0] = (self.S[i,0] + 1.0) / 2.0
        self.S[i,1] = (self.S[i,1] + 1.0) / 2.0

  def solve(self,w,h):

    # X_bar contains the centroid of the closest points to each calib point
    # now that we have X_bar, assemble X matrix
    for i in range(self.n - 1):
      x = self.X_bar[i,0]
      y = self.X_bar[i,1]
      self.X[i,0] = 1.0
      self.X[i,1] = x
      self.X[i,2] = y
      self.X[i,3] = x * y
      self.X[i,4] = x * x
      self.X[i,5] = y * y

    # set Y to calibration points (minus the last one as it's the same as first)
    for i in range(self.n - 1):
      self.Y[i,0] = self.S[i,0] * float(w)
      self.Y[i,1] = self.S[i,1] * float(h)

    #      Y    =      X       B
    # (n-1 x 2) = (n-1 x 6) (6 x 2)
#   print "Y = \n", self.Y
#   print "X = \n", self.X
#   print "B = \n", self.B

    #  (Xt X)^{-1}    Xt          Y    =    B
    #     (6 x 6)  (6 x n-1) (n-1 x 2) = (6 x 2)
#   print "X.T * X = \n", self.X.T * self.X
#   print "(X.T * X).I = \n", (self.X.T * self.X).I
#   print "X = \n", self.X
#   print "(X.T * X).I * X.T = \n", (self.X.T * self.X).I * self.X.T
#   print "( (X.T * X).I * X.T) * Y = \n", \
#        ( (self.X.T * self.X).I * self.X.T ) * self.Y

    self.B = ( (self.X.T*self.X).I * self.X.T ) * self.Y
#   print "B_hat = \n", self.B


  def transform(self,x,y):

    V_in = np.matrix(np.zeros((1,6),dtype=float))
    V_out = np.matrix(np.zeros((1,2),dtype=float))

    V_in[0,0] = 1.0
    V_in[0,1] = x
    V_in[0,2] = y
    V_in[0,3] = x * y
    V_in[0,4] = x * x
    V_in[0,5] = y * y

    V_out = V_in * self.B

    return (V_out[0,0], V_out[0,1])
