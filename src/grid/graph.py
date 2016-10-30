#!/usr/bin/env python2

# was using /sw/bin/python2.7 -tt
# filter.py
# this file is the driver for the python analysis script

# system includes
import sys
import os
import getopt
import glob
import numpy as np
import math

# local includes
from scanpath import Scanpath
from lagrange import Lagrange

import plotter
import point
from point import Point

from scipy import spatial 

def usage():
  print "Usage: python filter.py " \
        " --width=? --height=? --screen=? --dist=?" \
        " --xtiles=? --ytiles=?\n" \
        " --indir=? --imgdir=? --outdir=? --pltdir=?" \
        " --file=? --hertz=? --sfdegree=?\n" \
        " --sfcutoff=? --vt=?\n" \
        " --smooth=?\n"\
        "   width, height: screen dimensions (in pixels)\n" \
        "   screen: screen diagonal dimension (in inches)\n" \
        "   dist: viewing distance (in inches)\n" \
        "   xtiles: number of AOI tiles (in x direction)\n" \
        "   ytiles: number of AOI tiles (in y direction)\n" \
        "   indir: a directory containing input files to be processed\n" \
        "   imgdir: a directory containing image files\n" \
        "   outdir: a directory containing output files\n" \
        "   pltdir: a directory containing plot files\n" \
        "   file: a single file to process\n" \
        "   image: a single image to use as background\n" \
        "   hertz: the sampling rate of the data\n" \
        "   sfdegree: butterworth filter smoothing degree \n" \
        "   sfcutoff: butterworth filter smoothing cutoff \n" \
        "   vt: min velocity for change to be a saccade\n"\
        "   smooth: True enables butterworth smoothing\n"

def main(argv):
# if not len(argv):
#   usage()

  try:
    opts, args = getopt.getopt(argv, '', \
                 ['width=','height=',\
                  'hertz=',\
                  'screen=','dist=',\
                  'xtiles=','ytiles=',\
                  'indir=','imgdir=','outdir=','pltdir=',\
                  'file=','image=', 'hertz=', 'sfdegree=', 'sfcutoff=',\
                  'vt=', 'smooth='])
  except getopt.GetoptError, err:
    usage()
    exit()

  # Enable/disable butterworth smoothing.
  smooth = False
  # screen height in pixels
  width = 1680
  height = 1050
  # screen diagonal (in inches)
  screen = 22
  # viewing distance (in inches)
  dist = 23.62
  # sampling rate
  herz = 60.0
  # smoothing (Butterworth) filter parameters: degree, cutoff
  sfdegree = 2
# sfcutoff = 1.15 # more smooth
# sfcutoff = 1.65
# sfcutoff = 1.85
#  sfcutoff = 2.15 # last used
# sfcutoff = 2.35
# sfcutoff = 3.15
# sfcutoff = 4.15
  sfcutoff = 6.15 # less smooth
  # differentiation (SG) filter parameters: width, degree, order
  # 5, 3 for no smoothing, 3, 2 for smoothing
  if (smooth):
    dfwidth = 3
    dfdegree = 2
  else:
    dfwidth = 5
    dfdegree = 3
  #dfwidth = 7
  #dfwidth = 3
  #dfdegree = 4
  #dfdegree = 2
  dfo = 1
  # velocity threshold
  #T = 16.0
#T = 5.0  # more fixations
# T = 18.0
# T = 20.0
  T = 30.0
# T = 40.0 # last used
  #T = 25.0
# T = 30.0
  #T = 35.0 # fewer fixations
  #T = 40
  #T = 80.0
  #T = 100
  file = None
  # initially don't use an image (just plain white bgnd for plots)
  image = None

  # set up AOI grid
  xtiles = 4
  ytiles = 3

  outdir = "./data"
  pltdir = "./plots"

  # Checked beforehand so that custom parameters could still be used...
  # Check if smooth is an option. We will set default parameters based on 
  # value. If others are provided via the command line, we will use them.
  try:
      arg = opts[[t[0] for t in opts].index('--smooth')][1]
      if arg.lower() == 'true':
          smooth = True
      elif arg.lower() == 'false':
          smooth = False
      else:
          print "Warning, invalid smooth value. Assuming default."
      if (smooth):
        dfwidth = 3
        dfdegree = 2
      else:
        dfwidth = 5
        dfdegree = 3
  except Exception as e:
    print e
    sys.exit()
    pass

  for opt,arg in opts:
    opt = opt.lower()
    if(opt != '--file' and opt != '--image'):
      arg = arg.lower()

    if opt == '--indir':
      indir = arg
    elif opt == '--imgdir':
      imgdir = arg
    elif opt == '--outdir':
      outdir = arg
    elif opt == '--pltdir':
      pltdir = arg
    elif opt == '--width':
      width = arg
    elif opt == '--height':
      height = arg
    elif opt == '--screen':
      screen = float(arg)
    elif opt == '--dist':
      dist = float(arg)
    elif opt == '--xtiles':
      xtiles = int(arg)
    elif opt == '--ytiles':
      ytiles = int(arg)
    elif opt == '--file':
      file = arg
    elif opt == '--image':
      image = arg
    elif opt == '--hertz':
      herz = float(arg)
    elif opt == '--sfcutoff':
      sfcutoff = float(arg)
    elif opt == '--sfdegree':
      sfdegree = float(arg)
    elif opt == '--vt':
      T = float(arg)
    else:
      sys.argv[1:]

  # get .raw input files to process
  if os.path.isdir(indir):
    files = glob.glob('%s/*.raw' % (indir))

  # if user specified --file="..." then we use that as the only one to process
  if(file != None and os.path.isfile(file)):
    files = [file]

  # check to see if use specified image, if so, then don't compose filename
  # every time through loop; same image will be used (calibration image
  # presumably)...kind of a hack, but WTF...
  haveimage = False
  if(image != None and os.path.isdir(imgdir)):
    image = os.path.join(imgdir,image)
    haveimage = True

  lagrange = Lagrange(int(width),int(height))

  for file in files:
    scanpath = Scanpath()
    scanpath.parseFile(file,width,height,herz)

    base = os.path.basename(file)

    print "Processing: ", file, "[", base, "]"

    # extract stimulus name
#   imagebase = 'white-1600x900'
#   imagebase = 'gabor-parallel-notarget-1600x900'
#
#   serial_parallel = base.split('_')[2]
#   target_present = base.split('_')[3]
#   target_position = base.split('_')[4]
    imagebase = '_'.join(base.split('_')[1:5])

    # create filename of corresponding image
    if(haveimage == True):
      print "Image: ", image
    else:
      image = '{0}.png'.format(os.path.join(imgdir,imagebase))
      print "Image: ", image, "[", imagebase, "]"

    # split filename from extension
    filename, ext = os.path.splitext(base)

    scanpath.smooth("%s/%s-smth%s" % (outdir,filename,".dat"),\
                    width,height,herz,sfdegree,sfcutoff, smooth)
    scanpath.differentiate("%s/%s-diff%s" % (outdir,filename,".dat"),\
                            width,height,screen,dist,herz,dfwidth,dfdegree,dfo)
    scanpath.threshold("%s/%s-fxtn%s" % (outdir,filename,".dat"),\
                            width,height,T)
    scanpath.amfoc("%s/%s-amfo%s" % (outdir,filename,".dat"),\
                            width,height)
#   scanpath.gridify("%s/%s-aois%s" % (outdir,filename,".csv"),\
#                           subj,cond,width,height,xtiles,ytiles)

    scanpath.dumpDAT("%s/%s%s" % (outdir,filename,".dat"),width,height)
#   scanpath.dumpXML("%s/%s%s" % (outdir,filename,".xml"),width,height)

    plotter.renderPoints1D("%s/%s-%s" % (pltdir,filename,"gzpt"),\
                           width,height,\
                           scanpath.gazepoints,'x',\
                           "Gaze point data")

#   plotter.renderPoints1D("%s/%s-%s" % (pltdir,filename,"gzpt"),\
#                          width,height,\
#                          scanpath.gazepoints,'y',\
#                          "Gaze point data")

    plotter.renderPoints2D("%s/%s-%s" % (pltdir,filename,"gzpt"),\
                           width,height,\
                           scanpath.gazepoints,\
                           "Gaze point data",\
                           image,True,\
                           xtiles,ytiles)

#    plotter.renderPoints1D("%s/%s-%s" % (pltdir,filename,"smth"),\
#                          width,height,\
#                          scanpath.smthpoints,'x',\
#                          "Smoothed gaze point data")

#   plotter.renderPoints1D("%s/%s-%s" % (pltdir,filename,"smth"),\
#                          width,height,\
#                          scanpath.smthpoints,'y',\
#                          "Smoothed gaze point data")

#   plotter.renderPoints2D("%s/%s-%s" % (pltdir,filename,"smth"),\
#                          width,height,\
#                          scanpath.smthpoints,\
#                          "Smoothed gaze point data",
#                          image,True,\
#                          xtiles,ytiles)

    plotter.renderPoints1D("%s/%s-%s" % (pltdir,filename,"dfft"),\
                          width,height,\
                          scanpath.velocity,'x',\
                          "Differentiated gaze point data",False)

    plotter.renderPoints1D("%s/%s-%s" % (pltdir, filename, "accel"), \
                           width, height, \
                           scanpath.acceleration, 'x',\
                           "Twice Differentiated gaze point data", False)
    
#   plotter.renderPoints1D("%s/%s-%s" % (pltdir,filename,"dfft"),\
#                          width,height,\
#                          scanpath.velocity,'y',\
#                          "Differentiated gaze point data",False)

#   plotter.renderPoints2D("%s/%s-%s" % (pltdir,filename,"dfft"),\
#                          width,height,\
#                          scanpath.velocity,\
#                          "Differentiated gaze point data",\
#                          None,False,\
#                          xtiles,ytiles)

    plotter.renderFixations("%s/%s-%s" % (pltdir,filename,"fxtn"),\
                            width,height,\
                            screen,dist,\
                            scanpath.fixations,\
                            "Fixations",\
                            image,\
                            lagrange,\
                            xtiles,ytiles)

#   plotter.renderAmfocFixations("%s/%s-%s" % (pltdir,filename,"affx"),\
#                           width,height,\
#                           scanpath.fixations,\
#                           scanpath.K,\
#                           "Ambient/Focal Fixations",\
#                           image,\
#                           xtiles,ytiles)

    plotter.renderCalibFixations("%s/%s-%s" % (pltdir,filename,"clbf"),\
                            width,height,\
                            screen,dist,\
                            scanpath.fixations,\
                            "Mean Error and Fit of Fixated Calibration Locations",\
                            image,\
                            lagrange,\
                            xtiles,ytiles)

#   plotter.renderHeatmap("%s/%s-%s" % (pltdir,filename,"heat"),\
#                           width,height,\
#                           scanpath.fixations,\
#                           "Heatmap",\
#                           image,\
#                           xtiles,ytiles)

    print " "
    del scanpath

if __name__ == "__main__":
  main(sys.argv[1:])
