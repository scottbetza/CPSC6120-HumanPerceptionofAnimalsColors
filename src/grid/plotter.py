#!/usr/bin/env python
# Scanpath.py
# This class encapsulates the analysis program

# system includes
import sys
import os
import math
import re
import os.path
#import xml.etree.ElementTree as ET
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import tostring
import numpy as np
import xmlpp
from scipy import spatial 
# use agg non-gui backend
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path
import matplotlib.patches as patches
from PIL import Image
import pylab

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = False
plt.rc('font',family='sans-serif')
#plt.rc('text.latex',preamble='\usepackage{sfmath}')
#plt.rc('text.latex',preamble='\usepackage{cmbright}')

# local includes
from point import Point
from fixation import Fixation
from aoi import AOI

def renderPoints1D(baseName,w,h,points,coord,title,scale=True):
  # plot
  fileName = baseName + '-' + coord + '.pdf'
  # clear plot
  plt.clf()
  # axes dimensions
  ax = plt.axes()
# ax = plt.axes(aspect=1)
  if(coord == 'x' and scale):
    ax.set_ylim((0,int(w)))
  elif(coord == 'y' and scale):
    ax.set_ylim((0,int(h)))
  # fill in data points
  px = []
  py = []
  for pt in points:
    t = pt.gettimestamp()
    if(coord == 'x'):
      x = pt.at(0) * float(w) if scale else pt.at(0)
    elif(coord == 'y'):
      x = pt.at(1) * float(h) if scale else pt.at(1)
    elif(coord == 'k'):
      x = pt.at(0)
    px.append(t)
    py.append(x)

  # lines
  opt = {'antialiased':True,\
         'alpha':.6,\
         'color':"#3F3F3F",\
         'lw':1,\
         'marker':"o",\
         'markersize':100,\
         'markeredgecolor': "#787878",\
         'markeredgewidth':10}
  plt.plot(px,py,antialiased=True,alpha=.6,color="#787878")#, marker='o', markersize=2, linestyle='--')
# line = plt.Line2D(px,py,**opt)
# ax.add_artist(line)
  if (coord == 'x'):
      labels = ['point{0}'.format(i) for i in range(len(px))]
      indexes = [i for i in xrange(len(px))]
  elif (coord == 'y'):
      labels = ['point{0}'.format(i) for i in range(len(py))]
      indexes = [i for i in xrange(len(py))]
  """
  for i, label, x, y in zip(indexes, labels, px, py):
      if (i >= 120 and i <= 125) or (i >= 240 and i <= 245):
          plt.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad=0'))
  """
  # title and labels
  if(coord == 'x'):
    plt.title(title + ": $x$ coordinates")
    plt.ylabel("$x$-coordinate (pixels)",family="sans-serif")
  elif(coord == 'y'):
    plt.title(title + ": $y$ coordinates")
    plt.ylabel("$y$-coordinate (pixels)",family="sans-serif")
  elif(coord == 'k'):
    plt.title(title + ": $k$ coefficient")
    plt.ylabel("$k$-coefficient (standardized)",family="sans-serif")
  plt.xlabel("Time (s)",family="sans-serif")
  plt.grid(True,'major',ls='solid',alpha=.1)
  # margins
  plt.tight_layout()

  plt.savefig(fileName,transparent=True)

def renderPoints2D(baseName,w,h,points,title,image=None,scale=False,xtiles=4,ytiles=3):
  # plot
  fileName = baseName + '.pdf'

  # clear plot
  plt.clf()

  # axes dimensions (scaled for gaze points, not scaled for velocity)
  if(scale):
    ax = plt.axes(aspect=1)
    ax.set_xlim((0,int(w)))
    ax.set_ylim((0,int(h)))

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    # use +1 to get the last tick to show up
    ax.set_xticks(np.arange(xmin,xmax+1,xmax/xtiles))
    ax.set_yticks(np.arange(ymin,ymax+1,ymax/ytiles))
    plt.tick_params(labelsize="9")
  else:
    ax = plt.axes()

  # set background image
  # from: http://matplotlib.org/users/image_tutorial.html
  if image is not None:
#   img = mpimg.imread(image)
#   plt.imshow(img)
    # using PIL, see: http://effbot.org/imagingbook/image.htm
#   img = Image.open(image)
    img = Image.open(image).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    (imw, imh) = img.size
    x0 = (int(w)-imw)/2
    y0 = (int(h)-imh)/2
    x1 = x0+imw
    y1 = y0+imh
#   print "extent=",x0,x1,y0,y1
#   print "Image mode: ", img.mode
    if img.mode == "L":
      plt.imshow(np.asarray(img),cmap=pylab.gray(),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))
    else:
      plt.imshow(np.asarray(img),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))

  # fill in data points
  px = []
  py = []
  for pt in points:
    x = pt.at(0) * float(w) if scale else pt.at(0)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - pt.at(1)) * float(h) if scale else pt.at(1)
    px.append(x)
    py.append(y)

  if(not scale):
    ax.set_xlim(min(px),max(px))
    ax.set_ylim(min(py),max(py))

  # lines
  opt = {'antialiased':True,\
         'alpha':.6,\
         'color':"#3F3F3F",\
         'lw':1,\
         'marker':"o",\
         'markersize':2,\
         'markeredgecolor':"#787878",\
         'markeredgewidth':1}
  line = plt.Line2D(px,py,**opt)
  ax.add_artist(line)

  # title and labels
  plt.title(title)
  plt.ylabel("$y$-coordinate (pixels)",family="sans-serif")
  plt.xlabel("$x$-coordinate (pixels)",family="sans-serif")
  plt.grid(True,'major',ls='solid',alpha=.1)
  # margins
# plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
  plt.tight_layout()

# fig = fileName[:-4] + ".pdf"
  plt.savefig(fileName,transparent=True)

def renderFixations(baseName,w,h,screen,viewdist,fixations,title,image=None,lagrange=None,xtiles=4,ytiles=3):
  # plot
  fileName = baseName + '.pdf'

  # clear plot
  plt.clf()

  # axes dimensions
  ax = plt.axes(aspect=1)
  ax.set_xlim((0,int(w)))
  ax.set_ylim((0,int(h)))

  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  # use +1 to get the last tick to show up
  ax.set_xticks(np.arange(xmin,xmax+1,xmax/xtiles))
  ax.set_yticks(np.arange(ymin,ymax+1,ymax/ytiles))
  plt.tick_params(labelsize="9")

  # set background image
  # from: http://matplotlib.org/users/image_tutorial.html
  if image is not None:
#   img = mpimg.imread(image)
#   plt.imshow(img)
    # using PIL, see: http://effbot.org/imagingbook/image.htm
#   img = Image.open(image)
    img = Image.open(image).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    (imw, imh) = img.size
    x0 = (int(w)-imw)/2
    y0 = (int(h)-imh)/2
    x1 = x0+imw
    y1 = y0+imh
#   print "Image mode: ", img.mode
    if img.mode == "L":
      plt.imshow(np.asarray(img),cmap=pylab.gray(),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))
    else:
      plt.imshow(np.asarray(img),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))

  cx = []
  cy = []
  # calibration dots
  for i in range(lagrange.n):
    x = lagrange.S[i,0] * float(w)
    # don't do the y-coord flip for cal dots rendering with (0,0) at bottom-left
    # no I think we do need the y-flip
    y = (1.0 - lagrange.S[i,1]) * float(h)
    # don't append the last calibration point: it is the same as the first
    if i < lagrange.n - 1:
      cx.append(x)
      cy.append(y)

  # init kd-tree with calibration points
  # (each fixation point will then query for nearest neighbor)
  kdtree = spatial.KDTree(zip(cx,cy))

  # for each fixation, find nearest neighbor and distance to it (in pixels)
  X_err_i = [0]*(lagrange.n - 1)
  X_err = [0.0]*(lagrange.n - 1)
  distances = []
  for fx in fixations:
    x = fx.at(0) * float(w)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - fx.at(1)) * float(h)

    # see: http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    # get two nearest neigbhors: the first will be itself, with distance 0
    # return (lists of) nearest neighbor distances and indeces in tree data

    # use fixation point only if it is inside image borders
    # (image may be smaller than screen w,h)
    if image is None or (x0 < x and x < x1 and y0 < y and y < y1):

      # get nearest neighbor
      nndist,nnidxs = kdtree.query(np.array([[x,y]]),1)
#     print "nndist: ", nndist
#     print "nnidxs: ", nnidxs
#     print "dist to nearest neighbor (in pixels): ", nndist[0]

      # get index to kdtree element (same index as lagrange calib points)
      i = nnidxs[0]
      # get distance to closest kdtree element (calibration point)
      dist = float(nndist[0])
  
      # add in to list of distances
      distances.append(dist)

      # compute running mean for each calibration point as we encounter it
      X_err[i] = float(X_err_i[i])/float(X_err_i[i]+1) * X_err[i] + \
                               1.0/float(X_err_i[i]+1) * dist
      # compute centroid (running mean) of error at each calibration point
      lagrange.X_bar[i,0] = float(X_err_i[i])/float(X_err_i[i]+1) * \
                                              lagrange.X_bar[i,0] + \
                               1.0/float(X_err_i[i]+1) * x
      lagrange.X_bar[i,1] = float(X_err_i[i])/float(X_err_i[i]+1) * \
                                              lagrange.X_bar[i,1] + \
                               1.0/float(X_err_i[i]+1) * y

      # increment how many times we've been close to this calib point
      X_err_i[i] += 1

# print "X_err: ", X_err

  if len(fixations) > 0:
    # compute observed mean distance between each feature (fixation) and
    # its nearest neihgbor (calibration point)
    avedist = np.mean(distances)
    stddist = np.std(distances)
    print "mean dist b/ween any fixation and clb: %f (pixels)" % (avedist)

    r = math.sqrt(float(w)*float(w) + float(h)*float(h))
    dpi = r/float(screen)

    D = float(viewdist)

    fov = 2*math.degrees(math.atan2(screen,2*D))
    fovx = 2*math.degrees(math.atan2(float(w)/dpi,2*D))
    fovy = 2*math.degrees(math.atan2(float(h)/dpi,2*D))

    avedist_deg = 2*math.degrees(math.atan2((avedist/dpi),(2*D)))
    stddist_deg = 2*math.degrees(math.atan2((stddist/dpi),(2*D)))

    print "mean dist b/ween any fixation and clb: %f (degrees)" % (avedist_deg)
  
    strout = "view distance: %5.2f (inches), screen: %3.0f (inches), %5.2f$^{\circ}$ (visual angle), dpi: %5.2f" % \
             (D,screen,fov,dpi)
    ax.text(10,int(h)-40,strout,fontsize=10)
    strout = "mean error (accuracy): %5.2f$^{\circ}$ (degrees visual angle), standard deviation (precision): %5.2f$^{\circ}$" % \
             (avedist_deg, stddist_deg)
    ax.text(10,10,strout,fontsize=10)

  # fill in data points
  px = []
  py = []
  i=0
  for fx in fixations:
    x = fx.at(0) * float(w)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - fx.at(1)) * float(h)
    if image is None or (x0 < x and x < x1 and y0 < y and y < y1):
      px.append(x)
      py.append(y)
      if i>0:
        sx = fixations[i-1].at(0) * float(w)
        # do the y-coord flip for rendering with (0,0) at bottom-left
        sy = (1.0 - fixations[i-1].at(1)) * float(h)
        if image is None or (x0 < sx and sx < x1 and y0 < sy and sy < y1):
          dx = x - sx
          dy = y - sy
#         arrow = plt.Arrow(sx,sy,dx,dy,\
#                           width=35,\
#                           alpha=.2,\
#                           fill=False,\
#                           fc="none",\
#                           ec="#101010")
#         ax.add_patch(arrow)
          opt = {'shape':"full",\
                 'fill':False,\
                 'fc':"none",\
                 'ec':"#101010",\
                 'width':1,\
                 'head_width':15,\
                 'head_length':45,\
                 'length_includes_head':True}
          if(abs(dx) > 0.0 and abs(dy) > 0.0):
            plt.arrow(sx,sy,dx,dy,alpha=.3,**opt)
    i += 1
  # old way, straight plot
# plt.plot(px,py,antialiased=True,alpha=.6,color="#756BB1")
  # or using Artist
  line = plt.Line2D(px,py,antialiased=True,alpha=.3,color="#494949",lw=1)
  ax.add_artist(line)

  # circles
  for fx in fixations:
    x = fx.at(0) * float(w)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - fx.at(1)) * float(h)
    if image is None or (x0 < x and x < x1 and y0 < y and y < y1):
      r = fx.getPercentDuration() * 100.0
      circ = plt.Circle((x,y),radius=r,fc='#BFBFBF',ec='#393939',alpha=.6)
      ax.add_patch(circ)
  # title and labels
  plt.title(title)
  plt.ylabel("$y$-coordinate (pixels)",family="sans-serif")
  plt.xlabel("$x$-coordinate (pixels)",family="sans-serif")
  plt.grid(True,'major',ls='solid',alpha=.1)
  # margins
# plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
  plt.tight_layout()

# fig = fileName[:-4] + ".pdf"
  plt.savefig(fileName,transparent=True)

def renderAmfocFixations(baseName,w,h,fixations,K,title,image=None,xtiles=4,ytiles=3):
  # plot
  fileName = baseName + '.pdf'

  # clear plot
  plt.clf()

  # axes dimensions
  ax = plt.axes(aspect=1)
  ax.set_xlim((0,int(w)))
  ax.set_ylim((0,int(h)))

  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  # use +1 to get the last tick to show up
  ax.set_xticks(np.arange(xmin,xmax+1,xmax/xtiles))
  ax.set_yticks(np.arange(ymin,ymax+1,ymax/ytiles))
  plt.tick_params(labelsize="9")

  # set background image
  # from: http://matplotlib.org/users/image_tutorial.html
  if image is not None:
#   img = mpimg.imread(image)
#   plt.imshow(img)
    # using PIL, see: http://effbot.org/imagingbook/image.htm
#   img = Image.open(image)
    img = Image.open(image).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    (imw, imh) = img.size
    x0 = (int(w)-imw)/2
    y0 = (int(h)-imh)/2
    x1 = x0+imw
    y1 = y0+imh
#   print "Image mode: ", img.mode
    if img.mode == "L":
      plt.imshow(np.asarray(img),cmap=pylab.gray(),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))
    else:
      plt.imshow(np.asarray(img),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))

  # fill in data points
  px = []
  py = []
  i=0
  for fx in fixations:
    x = fx.at(0) * float(w)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - fx.at(1)) * float(h)
    if image is None or (x0 < x and x < x1 and y0 < y and y < y1):
      px.append(x)
      py.append(y)
      if i>0:
        sx = fixations[i-1].at(0) * float(w)
        # do the y-coord flip for rendering with (0,0) at bottom-left
        sy = (1.0 - fixations[i-1].at(1)) * float(h)
        if image is None or (x0 < sx and sx < x1 and y0 < sy and sy < y1):
          dx = x - sx
          dy = y - sy
#         arrow = plt.Arrow(sx,sy,dx,dy,\
#                           width=35,\
#                           alpha=.2,\
#                           fill=False,\
#                           fc="none",\
#                           ec="#101010")
#         ax.add_patch(arrow)
          opt = {'shape':"full",\
                 'fill':False,\
                 'fc':"none",\
                 'ec':"#101010",\
                 'width':1,\
                 'head_width':15,\
                 'head_length':45,\
                 'length_includes_head':True}
          if(abs(dx) > 0.0 and abs(dy) > 0.0):
            plt.arrow(sx,sy,dx,dy,alpha=.3,**opt)
    i += 1
  # old way, straight plot
# plt.plot(px,py,antialiased=True,alpha=.6,color="#756BB1")
  # or using Artist
  line = plt.Line2D(px,py,antialiased=True,alpha=.3,color="#494949",lw=1)
  ax.add_artist(line)

  if len(K) > 0:
    Kmin = min(K)
    Kmax = max(K)
    Krange = Kmax - Kmin
    print "K min, max, range = ", Kmin, Kmax, Krange
    nK = [0.0]*len(K)
      # normalize K
    for i in range(len(K)):
      if Krange > 0.0:
        nK[i] = (K[i] - Kmin)/(Krange)
      else:
        nK[i] = (K[i] - Kmin)

  # circles
  i=0
  for fx in fixations:
    x = fx.at(0) * float(w)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - fx.at(1)) * float(h)
    if image is None or (x0 < x and x < x1 and y0 < y and y < y1):
      r = fx.getPercentDuration() * 100.0
#     if K[i] > 0.3:
#       # focal
#       circ = plt.Circle((x,y),radius=r,fc='#fc8d59',ec='#393939',alpha=.6)
#     elif K[i] < -0.3:
#       # ambient
#       circ = plt.Circle((x,y),radius=r,fc='#91bfdb',ec='#393939',alpha=.6)
#     else:
#       # neither ambient nor focal, use grey color
#       circ = plt.Circle((x,y),radius=r,fc='#BFBFBF',ec='#393939',alpha=.6)
#     circ = plt.Circle((x,y),radius=r,fc=plt.cm.Oranges(nK[i]),ec='#393939',alpha=.6)
      circ = plt.Circle((x,y),radius=r,fc=plt.cm.Blues(nK[i]),ec='#393939',alpha=.6)
      ax.add_patch(circ)
    i += 1
  # title and labels
  plt.title(title)
  plt.ylabel("$y$-coordinate (pixels)",family="sans-serif")
  plt.xlabel("$x$-coordinate (pixels)",family="sans-serif")
  plt.grid(True,'major',ls='solid',alpha=.1)
  # margins
# plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
  plt.tight_layout()

# fig = fileName[:-4] + ".pdf"
  plt.savefig(fileName,transparent=True)

def renderCalibFixations(baseName,w,h,screen,viewdist,fixations,title,image=None,lagrange=None,xtiles=4,ytiles=3):

  # diagonal
  d = math.sqrt(float(w)*float(w) + float(h)*float(h))

  # plot
  fileName = baseName + '.pdf'

  # clear plot
  plt.clf()

  # axes dimensions
  ax = plt.axes(aspect=1)
  ax.set_xlim((0,int(w)))
  ax.set_ylim((0,int(h)))

  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  # use +1 to get the last tick to show up
  ax.set_xticks(np.arange(xmin,xmax+1,xmax/xtiles))
  ax.set_yticks(np.arange(ymin,ymax+1,ymax/ytiles))
  plt.tick_params(labelsize="9")

  # set background image
  # from: http://matplotlib.org/users/image_tutorial.html
  if image is not None:
#   img = mpimg.imread(image)
#   plt.imshow(img)
    # using PIL, see: http://effbot.org/imagingbook/image.htm
#   img = Image.open(image)
    img = Image.open(image).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    (imw, imh) = img.size
    x0 = (int(w)-imw)/2
    y0 = (int(h)-imh)/2
    x1 = x0+imw
    y1 = y0+imh
#   print "Image mode: ", img.mode
    if img.mode == "L":
      plt.imshow(np.asarray(img),cmap=pylab.gray(),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))
    else:
      plt.imshow(np.asarray(img),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))

  cx = []
  cy = []
  # calibration dots
  for i in range(lagrange.n):
    x = lagrange.S[i,0] * float(w)
    # don't do the y-coord flip for cal dots rendering with (0,0) at bottom-left
    # no I think we do need the y-flip
    y = (1.0 - lagrange.S[i,1]) * float(h)
#   print "cb ",i,": (x,y): (", x, ",", y, ")"
    r = 1.0/256.0 * d
    # circle for calibrtation dot
#   circ = plt.Circle((x,y),radius=r,fc='#bababa',ec='#393939',alpha=.6)
#   ax.add_patch(circ)
    # square box for calibration dot
#   path = Path([(x-r,y-r),(x+r,y-r),(x+r,y+r),(x-r,y+r),(x-r,y-r)],\
#               [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY])
#   patch = patches.PathPatch(path,fc='#bababa',ec='#393939',alpha=.6)
#   ax.add_patch(patch)
    # an x for centroid
    path = Path([(x-r,y-r),(x+r,y+r)],[Path.MOVETO,Path.LINETO])
    patch = patches.PathPatch(path,fc='#4d4d4d',ec='#4d4d4d',alpha=.8)
    ax.add_patch(patch)
    path = Path([(x-r,y+r),(x+r,y-r)],[Path.MOVETO,Path.LINETO])
    patch = patches.PathPatch(path,fc='#4d4d4d',ec='#4d4d4d',alpha=.8)
    ax.add_patch(patch)
    # don't append the last calibration point: it is the same as the first
    if i < lagrange.n - 1:
      cx.append(x)
      cy.append(y)

  # init kd-tree with calibration points
  # (each fixation point will then query for nearest neighbor)
  kdtree = spatial.KDTree(zip(cx,cy))

  # for each fixation, find nearest neighbor and distance to it (in pixels)
  X_err_i = [0]*(lagrange.n - 1)
  X_err = [0.0]*(lagrange.n - 1)
  distances = []
  for fx in fixations:
    x = fx.at(0) * float(w)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - fx.at(1)) * float(h)

    # see: http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    # get two nearest neigbhors: the first will be itself, with distance 0
    # return (lists of) nearest neighbor distances and indeces in tree data
#   nndist,nnidxs = kdtree.query(np.array([[x,y]]),2)
#   print "query: (x,y): (", x, ",", y, ")"
#   print "nndist: ", nndist
#   print "nnidxs: ", nnidxs
#   print "dist to nearest neighbor (in pixels): ", nndist[0][0]
#   print "kdtree.data: ", kdtree.data
#   print "kdtree.data[nnidxs[0][0]]: ", kdtree.data[nnidxs[0][0]]
#   print "kdtree.data[nnidxs[0][0]][0]: ", kdtree.data[nnidxs[0][0]][0]
#   print "kdtree.data[nnidxs[0][0]][1]: ", kdtree.data[nnidxs[0][0]][1]
#
#   # get index to kdtree element (same index as lagrange calib points)
#   i = nnidxs[0][0]
#   # get distance to closest kdtree element (calibration point)
#   dist = float(nndist[0][0])

    # use fixation point only if it is inside image borders
    # (image may be smaller than screen w,h)
    if image is None or (x0 < x and x < x1 and y0 < y and y < y1):
      # get nearest neighbor
      nndist,nnidxs = kdtree.query(np.array([[x,y]]),1)
#     print "nndist: ", nndist
#     print "nnidxs: ", nnidxs
#     print "dist to nearest neighbor (in pixels): ", nndist[0]

      # get index to kdtree element (same index as lagrange calib points)
      i = nnidxs[0]
      # get distance to closest kdtree element (calibration point)
      dist = float(nndist[0])

      # add in to list of distances
      distances.append(dist)

      # compute running mean for each calibration point as we encounter it
      X_err[i] = float(X_err_i[i])/float(X_err_i[i]+1) * X_err[i] + \
                               1.0/float(X_err_i[i]+1) * dist
      # compute centroid (running mean) of error at each calibration point
      lagrange.X_bar[i,0] = float(X_err_i[i])/float(X_err_i[i]+1) * \
                                              lagrange.X_bar[i,0] + \
                               1.0/float(X_err_i[i]+1) * x
      lagrange.X_bar[i,1] = float(X_err_i[i])/float(X_err_i[i]+1) * \
                                              lagrange.X_bar[i,1] + \
                               1.0/float(X_err_i[i]+1) * y

      # increment how many times we've been close to this calib point
      X_err_i[i] += 1

# print "X_err: ", X_err

  # render error centroid
  for i in range(lagrange.n - 1):
    sx = lagrange.S[i,0] * float(w)
    sy = (1.0 - lagrange.S[i,1])* float(h)
    x = lagrange.X_bar[i,0]
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = lagrange.X_bar[i,1]
#   print "er ",i,": (x,y): (", x, ",", y, ")"
    r = 1.0/256.0 * d
    # circle for centroid, not good...
#   circ = plt.Circle((x,y),radius=r,fc='#fdae61',ec='#393939',alpha=.6)
#   ax.add_patch(circ)
    # square box for centroid
    path = Path([(x-r,y-r),(x+r,y-r),(x+r,y+r),(x-r,y+r),(x-r,y-r)],\
                [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY])
    patch = patches.PathPatch(path,fc='#fddbc7',ec='#393939',alpha=.6)
    ax.add_patch(patch)
    # an x for centroid
#   path = Path([(x-r,y-r),(x+r,y+r)],[Path.MOVETO,Path.LINETO])
#   patch = patches.PathPatch(path,fc='#fdae61',ec='#393939',alpha=.6)
#   ax.add_patch(patch)
#   path = Path([(x-r,y+r),(x+r,y-r)],[Path.MOVETO,Path.LINETO])
#   patch = patches.PathPatch(path,fc='#fdae61',ec='#393939',alpha=.6)
#   ax.add_patch(patch)
    # line from centroid calibration point to fixation centroid
    # from: http://matplotlib.org/users/path_tutorial.html
    path = Path([(sx,sy),(x,y)],[Path.MOVETO,Path.LINETO])
    patch = patches.PathPatch(path,fc='#878787',ec='#878787',lw=1,alpha=.8)
    ax.add_patch(patch)

  if len(fixations) > 0:
    # compute observed mean distance between each feature (fixation) and
    # its nearest neihgbor (calibration point)
    avedist = np.mean(distances)
    stddist = np.std(distances)
    print "mean dist b/ween any fixation and clb: %f (pixels)" % (avedist)

    r = math.sqrt(float(w)*float(w) + float(h)*float(h))
    dpi = r/float(screen)

    D = float(viewdist)

    fov = 2*math.degrees(math.atan2(screen,2*D))
    fovx = 2*math.degrees(math.atan2(float(w)/dpi,2*D))
    fovy = 2*math.degrees(math.atan2(float(h)/dpi,2*D))

    avedist_deg = 2*math.degrees(math.atan2((avedist/dpi),(2*D)))
    stddist_deg = 2*math.degrees(math.atan2((stddist/dpi),(2*D)))

    print "mean dist b/ween any fixation and clb: %f (degrees)" % (avedist_deg)
  
    strout = "view distance: %5.2f (inches), screen: %3.0f (inches), %5.2f$^{\circ}$ (visual angle), dpi: %5.2f" % \
             (D,screen,fov,dpi)
    ax.text(10,int(h)-40,strout,fontsize=10)
    strout = "mean error (accuracy): %5.2f$^{\circ}$ (degrees visual angle), standard deviation (precision): %5.2f$^{\circ}$" % \
             (avedist_deg, stddist_deg)
    ax.text(10,10,strout,fontsize=10)

  lagrange.solve(w,h)

  # fill in data points
  px = []
  py = []
  i=0
  for fx in fixations:
    x = fx.at(0) * float(w)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - fx.at(1)) * float(h)
    if image is None or (x0 < x and x < x1 and y0 < y and y < y1):
      px.append(x)
      py.append(y)
      if i>0:
        sx = fixations[i-1].at(0) * float(w)
        # do the y-coord flip for rendering with (0,0) at bottom-left
        sy = (1.0 - fixations[i-1].at(1)) * float(h)
        if image is None or (x0 < sx and sx < x1 and y0 < sy and sy < y1):
          dx = x - sx
          dy = y - sy
#         arrow = plt.Arrow(sx,sy,dx,dy,\
#                           width=35,\
#                           alpha=.2,\
#                           fill=False,\
#                           fc="none",\
#                           ec="#101010")
#         ax.add_patch(arrow)
          opt = {'shape':"full",\
                 'fill':False,\
                 'fc':"none",\
                 'ec':"#101010",\
                 'width':1,\
                 'head_width':15,\
                 'head_length':45,\
                 'length_includes_head':True}
          if(abs(dx) > 0.0 and abs(dy) > 0.0):
            plt.arrow(sx,sy,dx,dy,alpha=.2,**opt)
      i += 1
    # old way, straight plot
# plt.plot(px,py,antialiased=True,alpha=.6,color="#756BB1")
  # or using Artist
  line = plt.Line2D(px,py,antialiased=True,alpha=.3,color="#494949",lw=1)
  ax.add_artist(line)

  # circles
  for fx in fixations:
    x = fx.at(0) * float(w)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - fx.at(1)) * float(h)
    if image is None or (x0 < x and x < x1 and y0 < y and y < y1):
      (fitx, fity) = lagrange.transform(x,y)
      r = fx.getPercentDuration() * 100.0
      circ = plt.Circle((x,y),radius=r,fc='#fddbc7',ec='#393939',alpha=.2)
      ax.add_patch(circ)
      circ = plt.Circle((fitx,fity),radius=r,fc='#d6604d',ec='#393939',alpha=.4)
      ax.add_patch(circ)

  # title and labels
  plt.title(title)
  plt.ylabel("$y$-coordinate (pixels)",family="sans-serif")
  plt.xlabel("$x$-coordinate (pixels)",family="sans-serif")
  plt.grid(True,'major',ls='solid',alpha=.1)
  # margins
# plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
  plt.tight_layout()

# fig = fileName[:-4] + ".pdf"
  plt.savefig(fileName,transparent=True)

def renderAOIs(baseName,w,h,aoidict,key,title,image=None,xtiles=4,ytiles=3):
  # plot
  fileName = baseName + '.pdf'

  # clear plot
  plt.clf()

  # axes dimensions
  ax = plt.axes(aspect=1)
  ax.set_xlim((0,int(w)))
  ax.set_ylim((0,int(h)))

  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  # use +1 to get the last tick to show up
  ax.set_xticks(np.arange(xmin,xmax+1,xmax/xtiles))
  ax.set_yticks(np.arange(ymin,ymax+1,ymax/ytiles))
  plt.tick_params(labelsize="9")

  # set background image
  # from: http://matplotlib.org/users/image_tutorial.html
  if image is not None:
#   img = mpimg.imread(image)
#   plt.imshow(img)
    # using PIL, see: http://effbot.org/imagingbook/image.htm
#   img = Image.open(image)
    img = Image.open(image).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    (imw, imh) = img.size
    x0 = (int(w)-imw)/2
    y0 = (int(h)-imh)/2
    x1 = x0+imw
    y1 = y0+imh
#   print "Image mode: ", img.mode
    if img.mode == "L":
      plt.imshow(np.asarray(img),cmap=pylab.gray(),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))
    else:
      plt.imshow(np.asarray(img),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))

  # AOIs
  for aoi in aoidict[key]:
    rect = plt.Rectangle(xy=aoi.getXY(h),\
                         width=aoi.getWidth(),\
                         height=aoi.getHeight(),\
                         fc='#BFBFBF',ec='#393939',alpha=.4)
    ax.add_patch(rect)

  # title and labels
  plt.title(title)
  plt.ylabel("$y$-coordinate (pixels)",family="sans-serif")
  plt.xlabel("$x$-coordinate (pixels)",family="sans-serif")
  plt.grid(True,'major',ls='solid',alpha=.1)
  # margins
# plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
  plt.tight_layout()

# fig = fileName[:-4] + ".pdf"
  plt.savefig(fileName,transparent=True)

def renderAOIFixations(baseName,w,h,fixations,aoidict,key,title,image=None,xtiles=4,ytiles=3):
  # plot
  fileName = baseName + '.pdf'

  # clear plot
  plt.clf()

  # axes dimensions
  ax = plt.axes(aspect=1)
  ax.set_xlim((0,int(w)))
  ax.set_ylim((0,int(h)))

  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  # use +1 to get the last tick to show up
  ax.set_xticks(np.arange(xmin,xmax+1,xmax/xtiles))
  ax.set_yticks(np.arange(ymin,ymax+1,ymax/ytiles))
  plt.tick_params(labelsize="9")

  # set background image
  # from: http://matplotlib.org/users/image_tutorial.html
  if image is not None:
#   img = mpimg.imread(image)
#   plt.imshow(img)
    # using PIL, see: http://effbot.org/imagingbook/image.htm
#   img = Image.open(image)
    img = Image.open(image).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    (imw, imh) = img.size
    x0 = (int(w)-imw)/2
    y0 = (int(h)-imh)/2
    x1 = x0+imw
    y1 = y0+imh
#   print "Image mode: ", img.mode
    if img.mode == "L":
      plt.imshow(np.asarray(img),cmap=pylab.gray(),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))
    else:
      plt.imshow(np.asarray(img),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))

  # fill in data points
  px = []
  py = []
  i=0
  for fx in fixations:
    x = fx.at(0) * float(w)
    y = fx.at(1) * float(h)
#   this code omits fixations that are outside of AOIs...trouble with it
#   is that it messes up the fixation scanpath sequence
#   inAOI = False
#   for aoi in aoidict[key]:
#     if aoi.inside(x,y + aoi.getHeight()):
#       inAOI = True
#       break
    inAOI = True
    if inAOI:
      # do the y-coord flip for rendering with (0,0) at bottom-left
      y = (1.0 - fx.at(1)) * float(h)
      px.append(x)
      py.append(y)
      if i>0:
        sx = fixations[i-1].at(0) * float(w)
        # do the y-coord flip for rendering with (0,0) at bottom-left
        sy = (1.0 - fixations[i-1].at(1)) * float(h)
        dx = x - sx
        dy = y - sy
#       arrow = plt.Arrow(sx,sy,dx,dy,\
#                         width=35,\
#                         alpha=.2,\
#                         fill=False,\
#                         fc="none",\
#                         ec="#101010")
#       ax.add_patch(arrow)
        opt = {'shape':"full",\
               'fill':False,\
               'fc':"none",\
               'ec':"#101010",\
               'width':1,\
               'head_width':15,\
               'head_length':45,\
               'length_includes_head':True}
        if(abs(dx) > 0.0 and abs(dy) > 0.0):
          plt.arrow(sx,sy,dx,dy,alpha=.3,**opt)
      i += 1
  # old way, straight plot
# plt.plot(px,py,antialiased=True,alpha=.6,color="#756BB1")
  # or using Artist
  line = plt.Line2D(px,py,antialiased=True,alpha=.3,color="#494949",lw=1)
  ax.add_artist(line)

  # circles
  for fx in fixations:
    x = fx.at(0) * float(w)
    y = fx.at(1) * float(h)
#   this code omits fixations that are outside of AOIs...trouble with it
#   is that it messes up the fixation scanpath sequence
#   inAOI = False
#   for aoi in aoidict[key]:
#     if aoi.inside(x,y + aoi.getHeight()):
#       inAOI = True
#       break
    inAOI = True
    if inAOI:
      # do the y-coord flip for rendering with (0,0) at bottom-left
      y = (1.0 - fx.at(1)) * float(h)
      r = fx.getPercentDuration() * 100.0
#     circ = plt.Circle((x,y),radius=r,fc='#BFBFBF',ec='#393939',alpha=.6)
      circ = plt.Circle((x,y),radius=r,fc='#fdae61',ec='#393939',alpha=.6)
      ax.add_patch(circ)

  # AOIs
  for aoi in aoidict[key]:
    inside = False
    for fx in fixations:
      x = fx.at(0) * float(w)
      # don't do the y-flip here (same ref. frame as AOIs)
      y = fx.at(1) * float(h)
      # add in AOI height---AOIs apear to be off-by-one line height
      if aoi.inside(x,y + aoi.getHeight()):
        inside = True
        break
    if inside:
      rect = plt.Rectangle(xy=aoi.getXY(h),\
                           width=aoi.getWidth(),\
                           height=aoi.getHeight(),\
                           fc='#d7191c',ec='#393939',alpha=.4)
    else:
      rect = plt.Rectangle(xy=aoi.getXY(h),\
                           width=aoi.getWidth(),\
                           height=aoi.getHeight(),\
                           fc='#abdda4',ec='#393939',alpha=.4)
    ax.add_patch(rect)

  # title and labels
  plt.title(title)
  plt.ylabel("$y$-coordinate (pixels)",family="sans-serif")
  plt.xlabel("$x$-coordinate (pixels)",family="sans-serif")
  plt.grid(True,'major',ls='solid',alpha=.1)
  # margins
# plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
  plt.tight_layout()

# fig = fileName[:-4] + ".pdf"
  plt.savefig(fileName,transparent=True)

def renderFixatedAOIs(baseName,w,h,fixations,aoidict,key,title,image=None,xtiles=4,ytiles=3):
  # plot
  fileName = baseName + '.pdf'

  # clear plot
  plt.clf()

  # axes dimensions
  ax = plt.axes(aspect=1)
  ax.set_xlim((0,int(w)))
  ax.set_ylim((0,int(h)))

  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  # use +1 to get the last tick to show up
  ax.set_xticks(np.arange(xmin,xmax+1,xmax/xtiles))
  ax.set_yticks(np.arange(ymin,ymax+1,ymax/ytiles))
  plt.tick_params(labelsize="9")

  # set background image
  # from: http://matplotlib.org/users/image_tutorial.html
  if image is not None:
#   img = mpimg.imread(image)
#   plt.imshow(img)
    # using PIL, see: http://effbot.org/imagingbook/image.htm
#   img = Image.open(image)
    img = Image.open(image).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    (imw, imh) = img.size
    x0 = (int(w)-imw)/2
    y0 = (int(h)-imh)/2
    x1 = x0+imw
    y1 = y0+imh
#   print "Image mode: ", img.mode
    if img.mode == "L":
      plt.imshow(np.asarray(img),cmap=pylab.gray(),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))
    else:
      plt.imshow(np.asarray(img),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))

  # fill in data points
  px = []
  py = []
  i=0
  prev_fixation=0
  for fx in fixations:
    x = fx.at(0) * float(w)
    y = fx.at(1) * float(h)
    inAOI = False
    for aoi in aoidict[key]:
      if aoi.inside(x,y + aoi.getHeight()):
        inAOI = True
        break
    if inAOI:
#     rect = plt.Rectangle(xy=aoi.getXY(h),\
#                          width=aoi.getWidth(),\
#                          height=aoi.getHeight(),\
#                          fc='#d7191c',ec='#393939',alpha=.4)
      rect = plt.Rectangle(xy=aoi.getXY(h),\
                           width=aoi.getWidth(),\
                           height=aoi.getHeight(),\
                           fc='#BFBFBF',ec='#393939',alpha=.2)
      ax.add_patch(rect)
      # do the y-coord flip for rendering with (0,0) at bottom-left
      y = (1.0 - fx.at(1)) * float(h)
      r = fx.getPercentDuration() * 100.0
#     circ = plt.Circle((x,y),radius=r,fc='#BFBFBF',ec='#393939',alpha=.6)
      circ = plt.Circle((x,y),radius=r,fc='#fdae61',ec='#393939',alpha=.6)
      ax.add_patch(circ)
      px.append(x)
      py.append(y)
      if prev_fixation>0:
        sx = fixations[prev_fixation].at(0) * float(w)
        # do the y-coord flip for rendering with (0,0) at bottom-left
        sy = (1.0 - fixations[prev_fixation].at(1)) * float(h)
        dx = x - sx
        dy = y - sy
        opt = {'shape':"full",\
               'fill':False,\
               'fc':"none",\
               'ec':"#101010",\
               'width':1,\
               'head_width':15,\
               'head_length':45,\
               'length_includes_head':True}
        if(abs(dx) > 0.0 and abs(dy) > 0.0):
          plt.arrow(sx,sy,dx,dy,alpha=.3,**opt)
      prev_fixation=i
    i += 1
  # or using Artist
  line = plt.Line2D(px,py,antialiased=True,alpha=.3,color="#494949",lw=1)
  ax.add_artist(line)

  # title and labels
  plt.title(title)
  plt.ylabel("$y$-coordinate (pixels)",family="sans-serif")
  plt.xlabel("$x$-coordinate (pixels)",family="sans-serif")
  plt.grid(True,'major',ls='solid',alpha=.1)
  # margins
# plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
  plt.tight_layout()

# fig = fileName[:-4] + ".pdf"
  plt.savefig(fileName,transparent=True)

def renderHeatmap(baseName,w,h,fixations,title,image=None,xtiles=4,ytiles=3):
  # see also: http://www.pmavridis.com/misc/heatmaps/

  # diagonal
  d = math.sqrt(float(w)*float(w) + float(h)*float(h))
  sigma = 0.0

  # plot
  fileName = baseName + '.pdf'
  fileName_png = baseName + '.png'

  # clear plot
  plt.clf()

  # axes dimensions
  ax = plt.axes(aspect=1)
  ax.set_xlim((0,int(w)))
  ax.set_ylim((0,int(h)))

  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  # use +1 to get the last tick to show up
  ax.set_xticks(np.arange(xmin,xmax+1,xmax/xtiles))
  ax.set_yticks(np.arange(ymin,ymax+1,ymax/ytiles))
  plt.tick_params(labelsize="9")

  # set background image
  # from: http://matplotlib.org/users/image_tutorial.html
  if image is not None:
#   img = mpimg.imread(image)
#   plt.imshow(img)
    # using PIL, see: http://effbot.org/imagingbook/image.htm
#   img = Image.open(image)
    img = Image.open(image).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    (imw, imh) = img.size
    x0 = (int(w)-imw)/2
    y0 = (int(h)-imh)/2
    x1 = x0+imw
    y1 = y0+imh
#   print "Image mode: ", img.mode
    if img.mode == "L":
      plt.imshow(np.asarray(img),alpha=0.5,cmap=pylab.gray(),
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))
    else:
      plt.imshow(np.asarray(img),alpha=0.5,
                 origin='None',aspect='auto',extent=(x0,x1,y0,y1))

  # heatmap: a 32-bit floating point image, initially set to black
  lum = Image.new("F", (int(w),int(h)), 0.0)
  pix = lum.load()
  print "processing ", len(fixations), " fixations..."
  for fx in fixations:
    x = fx.at(0) * float(w)
    # do the y-coord flip for rendering with (0,0) at bottom-left
    y = (1.0 - fx.at(1)) * float(h)
    # hack: if there's only 1 fixation @ 0% duration: 1/6th of image
    sigma = fx.getPercentDuration() * d/6.0 if fx.getPercentDuration() > 0 else d/6.0
#   print "percent duration = ", fx.getPercentDuration(), "d = ", d
#   print "sigma = ", sigma
    # lum.putdata() might be faster!!
#   for i in range(int(h)):
#     for j in range(int(w)):
    for i in xrange(int(y-2.0*sigma),int(y+2.0*sigma)):
      for j in xrange(int(x-2.0*sigma),int(x+2.0*sigma)):
        if( 0 <= i and i < int(h) and 0 <= j and j < int(w) ):
          sx = j - x
          sy = i - y
          heat = math.exp((sx*sx + sy*sy)/(-2.0*sigma*sigma))
          pix[j,i] = pix[j,i] + heat
  print "done."

  # get max value
  minlum, maxlum = lum.getextrema()
  print "minlum, maxlum = ", (minlum, maxlum)

  # normalize
# for i in range(int(h)):
#   for j in range(int(w)):
#     pix[j,i] = pix[j,i] / maxlum
  if(abs(maxlum) < 0.00001):
    maxlum = 1.0
  lum = lum.point(lambda f: f * (1.0/maxlum) + 0)
  print "done normalizing"

  # convert to grayscale
# out = lum.point(lambda f: f * 255.0 + 0,"L")
  out = lum.point(lambda f: f * 255.0 + 0)
  print "done converting"

  # plot
  # imshow can handle lum image, default colormap is "jet", e.g., heatmap
# plt.imshow(np.asarray(lum),cmap="gray")
  plt.imshow(np.asarray(lum),cmap="jet",alpha=0.5)
# plt.imshow(np.asarray(lum),cmap="rainbow",alpha=0.5)

  # title and labels
  plt.title(title)
  plt.ylabel("$y$-coordinate (pixels)",family="sans-serif")
  plt.xlabel("$x$-coordinate (pixels)",family="sans-serif")
  plt.grid(True,'major',ls='solid',alpha=.1)
  # margins
# plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
  plt.tight_layout()

# fig = fileName[:-4] + ".pdf"
  plt.savefig(fileName,transparent=True)
  plt.savefig(fileName_png,transparent=True)


def raw_stats(w, h, screen, viewdist, scanpath, filename, T, outdir, smooth, sf):
  stddevs_x = []
  stddevs_y = []
  stddevs = []
  normal_stddevs_x = []
  normal_stddevs_y = []
  normal_stddevs = []

  fixation_x = []
  fixation_y = []
  fixations = 0
  et = 0
  st = 0
  tt = 0
  FIXATION = 1
  SACCADE = 0
# state = SACCADE
  state = FIXATION


  r = math.sqrt(float(w)*float(w) + float(h)*float(h))
  dpi = r/float(screen)

  D = float(viewdist)
  fov = 2*math.degrees(math.atan2(screen,2*D))
  fovx = 2*math.degrees(math.atan2(float(w)/dpi,2*D))
  fovy = 2*math.degrees(math.atan2(float(h)/dpi,2*D))
  
  et_i = 0
  st_i = 0
  for i, point in enumerate(scanpath.gazepoints):
      velx = scanpath.velocity[i].at(0)
      vely = scanpath.velocity[i].at(1)
      amp = math.fabs(math.sqrt(velx * velx + vely * vely))
      # Saccade
      if (amp > T / 2.0):
          if (state == FIXATION):
            et = scanpath.smthpoints[i].gettimestamp()
            et_i = i
            tt = et - st
            if tt > 0.0:  
              if (fixation_x != []):
                  xstddev = np.std(fixation_x)
                  ystddev = np.std(fixation_y)
                  stddev = math.sqrt(xstddev * xstddev + ystddev * ystddev)
                  deg_stddev = 2 * math.degrees(math.atan((stddev/dpi) / (2 * D)))
                  deg_stddev_x = 2 * math.degrees(math.atan((xstddev/dpi) / (2 * D)))
                  deg_stddev_y = 2 * math.degrees(math.atan((ystddev/dpi) / (2 * D)))

                  stddevs_x.append(deg_stddev_x)
                  stddevs_y.append(deg_stddev_y)
                  stddevs.append(deg_stddev)

                  
                  normal_x = [x / float(w) for x in fixation_x]
                  normal_y = [y / float(h) for y in fixation_y]
                  normalx_stddev = np.std(normal_x)
                  normaly_stddev = np.std(normal_y)
                  normal_stddev = math.sqrt(normalx_stddev * normalx_stddev + 
                                           normaly_stddev * normaly_stddev)

                  normal_stddevs_x.append(normalx_stddev)
                  normal_stddevs_y.append(normaly_stddev)
                  normal_stddevs.append(normal_stddev)

                  fixation_x = []
                  fixation_y = []
                  fixations += 1
          state = SACCADE
          #print scanpath.gazepoints[i].coord[0], " ", scanpath.gazepoints[i].coord[1]
      # Fixation
      else:
          if (state != FIXATION):
              st = scanpath.smthpoints[i].gettimestamp()
              if et != 0:
                  sacc_time =((scanpath.smthpoints[i - 1].gettimestamp() - et) * 1000)
           #       print "sacc_time: ", sacc_time
                  x = scanpath.gazepoints[i - 1].at(0) - scanpath.gazepoints[et_i].at(0)
            #      print "Points: ", scanpath.gazepoints[i - 1].at(0), " ", scanpath.gazepoints[et_i].at(0)
                  y =  scanpath.gazepoints[i - 1].at(1) - scanpath.gazepoints[et_i].at(1)
                  x *= 1600
                  y *= 900
                  amp = math.sqrt(pow(x, 2) + pow(y, 2)) / dpi
             #     print "amp: ", amp 
                  sacc_amp = 2 * math.degrees(math.atan(amp / (2 * D)))
              #    Remove erroneous saccades.
                  if (sacc_time > 0 and sacc_time < 133.73049 and sacc_amp > 2.0):
                      subj = None
                      subj = os.path.basename(filename)
                      subj = re.findall(r'\d+', subj)
                      if len(subj) == 0:
                          subj = 0
                      else: 
                          subj = subj[-1]
                      subj = "S" + str(subj)
                      cond = None
                      if smooth:
                          cond = "B-SG"
                      else:
                          cond = "SG"
                      sf.write("%s,%s,%f,%f\n" % (cond, subj, sacc_amp, sacc_time))
                  else:
                      print sacc_time
                  pass
          x = point.at(0) * float(w)
         # y = point.at(1) * float(h)
          y = (1.0 - point.at(1)) * float(h)
          fixation_x.append(x)
          fixation_y.append(y)
          state = FIXATION


  total_stddev_mean_x = np.mean(stddevs_x)
  total_stddev_mean_y = np.mean(stddevs_y)
  total_stddev_mean = np.mean(stddevs)

  normal_stddev_mean_x = np.mean(normal_stddevs_x)
  normal_stddev_mean_y = np.mean(normal_stddevs_y)
  normal_stddev_mean = np.mean(normal_stddevs)

  with open(os.path.join(outdir, "%s-stats.%s" % (filename, "txt")), "w") as f:
      f.write("degree stddev x, y, total, normalized stddev x, y, total\n")
      for i in xrange(0, fixations):
          f.write(" %f, %f, %f, %f, %f, %f\n" % (stddevs_x[i], stddevs_y[i], stddevs[i], normal_stddevs_x[i], normal_stddevs_y[i], normal_stddevs[i]))
      f.write("total:  %f, %f, %f, %f, %f, %f" % (total_stddev_mean_x, total_stddev_mean_y, total_stddev_mean, normal_stddev_mean_x, normal_stddev_mean_y, normal_stddev_mean)) 

