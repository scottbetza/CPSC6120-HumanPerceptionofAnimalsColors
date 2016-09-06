#!/usr/bin/env python
# Scanpath.py
# This class encapsulates the analysis program

# system includes
import sys
import os
import math
#import xml.etree.ElementTree as ET
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import tostring
import numpy as np
from scipy.interpolate import UnivariateSpline
import xmlpp

# local includes
from point import Point
from fixation import Fixation
from grid import Grid
from aoi import AOI

import bw
import sg

class Run:
  def __init__(self,subj,runn,date):
    self.subjectName = subj
    self.runNumber = runn
    self.dateTime = date

class Scanpath:
  def __init__(self):
    self.normal = [1280.0, 1024.0]
    self.refresh_rate = 50
    self.fileName = None
    self.expName = None
    self.run = Run("1","1","January")
    self.fileExtension = None
    self.filter = None
    self.gazepoints = []
    self.smthpoints = []
    self.velocity = []
    self.acceleration = []
    self.fixations = []
    self.K = []

  def parseFile(self, fileName, w, h, herz):
    self.fileName = fileName

    # get text after last '.' symbol and force it to lowercase
    self.fileExtension = fileName.split('.')[-1].lower()

    if self.fileExtension == 'xml':
      # currently, Mirametrix stimulus dimensions are NOT
      # recorded in the input files, so we have to hardcode
      # it here.
      self.normal = [float(w), float(h)]
      self.refresh_rate = float(herz)
      self.parseXML()
    elif self.fileExtension == 'raw':
      self.normal = [float(w), float(h)]
      self.refresh_rate = float(herz)
      self.parseRAW()
    elif self.fileExtension == 'fxd':
      self.normal = [float(w), float(h)]
      self.refresh_rate = float(herz)
      self.parseFXD()
    elif self.fileExtension == 'txt':
      self.normal = [float(w), float(h)]
      self.refresh_rate = float(herz)
      self.parseClearViewEFD()
    else:
      print 'Unsupported file extension! (%s)' % (self.fileExtension)
			
  # parse fxd (x y d) fixation file
  def parseFXD(self):
    try:
      f = open(self.fileName,'r')
    except IOError:
      print "Can't open file: " + self.fileName
      return

    # read lines, throwing away first one (header)
  # linelist = f.readlines()
    # linelist = f.read().split('\r')
    linelist = f.read().splitlines()
    # no header in raw files
  # linelist = linelist[1:]

    for line in linelist:
      entry = line.split(' ')

      # coords should already be normalized
      x = float(entry[0])/self.normal[0]
      y = float(entry[1])/self.normal[1]
      d = float(entry[2])

      x, y = self.clamp(x, y)

      self.fixations.append(Fixation(x,y,0,d))

    # compute mean fixation duration
    durations = [0.0]*len(self.fixations)
    for i in range(len(self.fixations)):
      durations[i] = self.fixations[i].getDuration()
    avedur = np.mean(durations)
    mindur = min(durations)
    maxdur = max(durations)

        #print "Read: %d fixations, %f sec mean duration, [%f,%f]" % \
         #         (len(self.fixations), avedur, mindur, maxdur)

    # normalize
    for i in range(len(self.fixations)):
      self.fixations[i].normalizeDuration(mindur,maxdur)

  # parse a raw (x y t) source file
  def parseRAW(self):
    try:
      f = open(self.fileName,'r')
    except IOError:
      print "Can't open file: " + self.fileName
      return

    # read lines, throwing away first one (header)
  # linelist = f.readlines()
    # linelist = f.read().split('\r')
    linelist = f.read().splitlines()
    # no header in raw files
  # linelist = linelist[1:]

    for line in linelist:
      entry = line.split(' ')

      # coords should already be normalized
      x = float(entry[0])
      y = float(entry[1])
      t = float(entry[2])
      err = 'None'

      x, y = self.clamp(x, y)

      self.gazepoints.append(Point(x,y,t,err))
      
  # print "Read: ", len(self.gazepoints), "points"

  # parse a mirametrix source file
  # (first part is old code used to parse Alicann's redone XML file)
  def parseXML(self):
      #print "Parsing: ",self.fileName
    tree = ET.parse(self.fileName)
      #print "tree = %s" % tree
 
    # should be something like Experiment, {}
    root = tree.getroot()
      #print "root.tag = %s, root.attrib = %s" % (root.tag, root.attrib)
 
    for rec in root:
      gazeattr = rec.attrib

      bogus = False
 
      t = float(gazeattr['TIME'])
      xl = float(gazeattr['LPOGX'].split()[0])
      yl = float(gazeattr['LPOGY'].split()[0])
      xr = float(gazeattr['RPOGX'].split()[0])
      yr = float(gazeattr['RPOGY'].split()[0])
      errl = gazeattr['LPOGV']
      errr = gazeattr['RPOGV']

      if(errl == "0" and errr == "0"):
        bogus = True

      if(not bogus):
        x = (xl+xr)/2.0
        y = (yl+yr)/2.0

        x, y = self.clamp(x, y)

        self.gazepoints.append(Point(x,y,t,"None"))
     
    # print "Read: ", len(self.gazepoints), "points"
 
  # dump out XML
  def dumpXML(self,fileName,w,h):
      #print "Dumping ", fileName

    # buld tree structure
    exp = ET.Element('Experiment',filename=self.expName)
#   print exp.tag, exp.attrib

    run = ET.SubElement(exp,'Run')
    run.attrib['subjectName'] = self.run.subjectName
    run.attrib['runNumber'] = self.run.runNumber
    run.attrib['dateTime'] = self.run.dateTime

#   print run.tag, run.attrib

#   for pt in self.gazepoints:
#     data = ET.SubElement(run,'GazeData')
#     data.attrib['Id'] = "1"
#     data.attrib['Time'] = "%f" % (pt.gettimestamp())
#     data.attrib['Error'] = "%s" % (pt.getStatus())
#     data.attrib['X'] = "%f" % (pt.at(0))
#     data.attrib['Y'] = "%f" % (pt.at(1))

    for fx in self.fixations:
      data = ET.SubElement(run,'FixationData')
      data.attrib['Id'] = "1"
      data.attrib['Time'] = "%f" % (fx.gettimestamp())
      data.attrib['FX'] = "%f" % (fx.at(0) * float(w))
      data.attrib['FY'] = "%f" % (fx.at(1) * float(h))
      data.attrib['Duration'] = "%f" % (fx.getDuration())
      data.attrib['PercentDuration'] = "%f" % (fx.getPercentDuration())

    # wrap experiment in ElementTree instance, output
    # this is machine-readable output, not very human-readable
#   tree = ET.ElementTree(exp)
##  tree.write(fileName)

    # this is more human-readable but quite slow
    outfile = open(fileName,'w')
#   print xmlpp.pprint(tostring(exp))
    xmlpp.pprint(tostring(exp),output=outfile)
    outfile.close()

  def dumpDAT(self,fileName,w,h):

    outfile = open(fileName,'w')
    for pt in self.gazepoints:
      x = pt.at(0) * float(w)
      y = pt.at(1) * float(h)
      str = "%f %f %f\n" % (pt.gettimestamp(), x, y)
      outfile.write(str)
    outfile.close()

  
  def smooth(self,fileName,w,h,herz,sfdegree,sfcutoff, smooth):

    if smooth:
        # use Butterworth filter for smoothing
        self.smthpoints = bw.applyBWFilter(self.gazepoints,sfdegree,herz,sfcutoff)
    else:
        self.smthpoints = self.gazepoints
    outfile = open(fileName,'w')
    for pt in self.smthpoints:
      x = pt.at(0) * float(w)
      y = pt.at(1) * float(h)
      str = "%f %f %f\n" % (pt.gettimestamp(), x, y)
      outfile.write(str)
    outfile.close()

  def differentiate(self,fileName,w,h,screen,dist,herz,dfwidth,dfdegree,dfo):

    # differentiate smoothed points with Savitzky-Golay filter
    diffpoints = sg.applySGFilter(self.smthpoints,dfwidth,dfdegree,dfo)
    accelpoints = sg.applySGFilter(diffpoints, dfwidth, dfdegree, dfo)

    # sampling period in s
    period = float(1.0/float(herz))
    dt = period * float(dfwidth)

    r = math.sqrt(float(w)*float(w) + float(h)*float(h))
    dpi = r/float(screen)

    D = float(dist)
#   fov = 2*math.degrees(math.atan(math.radians(screen/(2*D))))
    fov = 2*math.degrees(math.atan2(screen,2*D))
    fovx = 2*math.degrees(math.atan2(float(w)/dpi,2*D))
    fovy = 2*math.degrees(math.atan2(float(h)/dpi,2*D))

        #print "screen subtends %f (%f x %f) degrees" % \
        #                   (float(fov),float(fovx),float(fovy))
      #print "screen aspect = %f" % float(float(w)/float(h))
      #print "sampling period = %f (ms)" % float(period * 1000.0)
      #print "filter window = %f (ms)" % float(dt * 1000.0)
      #print "dpi = %f" % dpi

    for pt in diffpoints:

      # distance covered in pixels (across diff filter window size)
      dx = pt.at(0) * float(w)
      dy = pt.at(1) * float(h)

      # assume D is in inches
#     degx = 2*math.degrees(math.atan(math.radians((dx/dpi)/(2*D))))
#     degy = 2*math.degrees(math.atan(math.radians((dy/dpi)/(2*D))))
      degx = 2*math.degrees(math.atan2((dx/dpi),(2*D)))
      degy = 2*math.degrees(math.atan2((dy/dpi),(2*D)))

      # degx, degy is degrees per filter window, div by dt to get per second
      velx = degx / dt
      vely = degy / dt

      self.velocity.append(Point(velx,vely,pt.gettimestamp(),pt.getStatus()))

          #print "pts: %d, smth: %d, diff: %d, vel: %d" % \
           #      (len(self.gazepoints),len(self.smthpoints), \
            #      len(diffpoints),len(self.velocity))
    for pt in accelpoints:
      dx = pt.at(0) * float(w)
      dy = pt.at(1) * float(h)

      degx = 2*math.degrees(math.atan2((dx/dpi),(2*D)))
      degy = 2*math.degrees(math.atan2((dy/dpi),(2*D)))

      # degx, degy is degrees per filter window, div by dt to get per second
      accelx = degx / dt
      accely = degy / dt

      self.acceleration.append(Point(accelx,accely,pt.gettimestamp(),pt.getStatus()))


    outfile = open(fileName,'w')
    for pt in self.velocity:
      # don't scale by w,h here, already did so above
      x = pt.at(0)
      y = pt.at(1)
      str = "%f %f %f\n" % (pt.gettimestamp(), x, y)
      outfile.write(str)
    outfile.close()

  def threshold(self,fileName,w,h,T):

    # state table
    # state         |   input    |  output
    # ------------------------------------
    # (1) fixation  | a < T (1)  |    1
    # (0) saccade   | a < T (1)  |    1
    # (1) fixation  | a > T (0)  |    0
    # (0) saccade   | a > T (0)  |    0
    #

    # fixation state enums
    fixation = 1
    saccade = 0

    # assuming starting state = saccade
    state = fixation

    st = 0.
    et = 0.
    tt = 0.
    ux = 0.
    uy = 0.
    k = 0
    if(len(self.velocity) > 0):
      for i in range(len(self.smthpoints)):

        # (smoothed) data we are classifying
        x = self.smthpoints[i].at(0)
        y = self.smthpoints[i].at(1)

        # corresponding velocity
        vx = self.velocity[i].at(0)
        vy = self.velocity[i].at(1)

        # saccade amplitude
        amp = math.sqrt(vx*vx + vy*vy)

        if math.fabs(amp) > float(T)/2:

          # amplitude > T
          if state == fixation:
            # state transition from fixation to saccade (fixation ends)

            # get end time of current point, compute duration
            et = self.smthpoints[i].gettimestamp()
            tt = et - st
            # don't add fixation with -ve duration
            # (due to spurious timestamp in .raw file)
            if st > 0.0 and tt > 0.0:
              # need to clamp fixation centroid: it could have gone negative
              ux, uy = self.clamp(ux, uy)
              self.fixations.append(Fixation(ux,uy,st,tt))
          else:
            # state transition from saccade to saccade (saccade continues)
            pass

          state = saccade

        else:

          # amplitude < T
          if state == fixation:
            # state transition from fixation to fixation (fixation continues)
            pass
          else:
            # state transition from saccade to fixation (fixation starts)

            # set start time
            st = self.smthpoints[i].gettimestamp()

            # reset running mean and counter
            ux = 0.
            uy = 0.
            k = 0

          # compute running mean
          # from Brown, Robert Grover, "Introduction to Random Signal
          #   Analysis and Kalman Filtering", John Wiley & Sons, New York, NY
          #   1983 [p.182] TK5102.5 .B696
          ux = float(k)/float(k+1) * ux + 1.0/float(k+1) * x
          uy = float(k)/float(k+1) * uy + 1.0/float(k+1) * y
          k += 1

          state = fixation

#     print "i = %d, st = %f, et = %f, tt = %f" % (i,st,et,tt)

    # compute mean fixation duration
    if(len(self.fixations) > 1):
      durations = [0.0]*len(self.fixations)
      for i in range(len(self.fixations)):
        durations[i] = self.fixations[i].getDuration()
      avedur = np.mean(durations)
      mindur = min(durations)
      maxdur = max(durations)
    elif(len(self.fixations) == 1):
      avedur = self.fixations[0].getDuration()
      mindur = avedur
      maxdur = avedur
    else:
      avedur = 0.0
      mindur = avedur
      maxdur = avedur

          #print "Found: %d fixations, %f sec mean duration, [%f,%f]" % \
           #       (len(self.fixations), avedur, mindur, maxdur)

    # normalize
    for i in range(len(self.fixations)):
      self.fixations[i].normalizeDuration(mindur,maxdur)

    outfile = open(fileName,'w')
#    for pt in self.fixations:
#      x = pt.at(0) * float(w)
#      y = pt.at(1) * float(h)
#      str = "%f %f %f %f\n" % (pt.gettimestamp(), x, y, pt.getDuration())
##     str = "%f %f %f %f\n" % (pt.gettimestamp(), x, y, pt.getPercentDuration())
#      outfile.write(str)
    for i in range(len(self.fixations)):
      x = self.fixations[i].at(0) * float(w)
      y = self.fixations[i].at(1) * float(h)
      t = self.fixations[i].gettimestamp()
      dur = self.fixations[i].getDuration()
      st = t   # fixation timestamp is its starttime (st)
      tt = dur # fixation duration is its endtime - startime (et - st)
      et = st + tt
      if i < len(self.fixations)-1:
        dx = x - (self.fixations[i+1].at(0) * float(w))
        dy = y - (self.fixations[i+1].at(1) * float(h))
        sacc_dur = self.fixations[i+1].gettimestamp() - et
        amp = math.sqrt(dx*dx + dy*dy)
      else:
        amp = 0.0
# what we used to do
#     str = "%f %f %f %f %f\n" % (t, x, y, dur, amp)
# now adding in saccade duration
      str = "%f %f %f %f %f %f\n" % (t, x, y, dur, amp, sacc_dur)
      outfile.write(str)

    outfile.close()

  def amfoc(self,fileName,w,h):

    self.K = [0.0]*(len(self.fixations))

    # compute mean fixation duration
    if(len(self.fixations) > 1):
      durations = [0.0]*len(self.fixations)
      for i in range(len(self.fixations)):
        durations[i] = self.fixations[i].getDuration()
      avedur = np.mean(durations)
      stddur = np.std(durations)
      mindur = min(durations)
      maxdur = max(durations)

      # compute mean amplitudes
      amps = [0.0]*(len(self.fixations)-1)
      for i in range(len(self.fixations)-1):
        dx = self.fixations[i].at(0) - self.fixations[i+1].at(0)
        dy = self.fixations[i].at(1) - self.fixations[i+1].at(1)
        dist = math.sqrt(dx*dx + dy*dy)
        amps[i] = dist
      aveamp = np.mean(amps)
      stdamp = np.std(amps)

      # compute ambient/focal coefficients
      for i in range(len(self.fixations)-1):
        if(stdamp < 0.00001):
          self.K[i] = 0.0
        else:
          self.K[i] = (self.fixations[i].getDuration() - avedur)/stddur - \
                      (amps[i] - aveamp)/stdamp

      # set K for the last entry (should be same as penultimate K)
      self.K[len(self.fixations)-1] = self.K[len(self.fixations)-2]

    elif(len(self.fixations) == 1):
      avedur = self.fixations[0].getDuration()
      stddur = 1.0
      mindur = avedur
      maxdur = avedur
    else:
      avedur = 0.0
      stddur = 0.0
      mindur = avedur
      maxdur = avedur

    outfile = open(fileName,'w')
    for i in range(len(self.fixations)):
      str = "%f %f\n" % (self.fixations[i].gettimestamp(), self.K[i])
      outfile.write(str)
    outfile.close()

  def gridify(self,fileName,subj,cond,w,h,xtiles,ytiles):

    # AOIs: xtiles x ytiles overlaid on the image
    grid = Grid(xtiles,ytiles)

#   grid.fill(self.fixations,w,h)

    outfile = open(fileName,'w')
    str = "subj,cond,AOI,duration,order\n"
    outfile.write(str)

#   this prints out the number of fixations per AOI
#   for i in range(xtiles):
#     for j in range(ytiles):
#       str = "%c%c %d\n" % (chr(int(i+97)), chr(int(j+97)), grid.at(i,j))
#       outfile.write(str)
    ct = 1
    for fx in self.fixations:
      # ix,iy give the integer coords of the AOI
      # use float(w)-1 and float(h)-1 to avoid points on the last AOI boundary
      ix = (fx.at(0) * (float(w)-1))//((int(w)//xtiles))
#     iy = (fx.at(1) * (float(h)-1))//((int(h)//ytiles))
      # do the y-coord flip for rendering with (0,0) at bottom-left
      iy = ((1.0 - fx.at(1)) * (float(h)-1))//((int(h)//ytiles))
      # get character labels (1-26 set)
#     lx = "%c" % (chr(int(ix+97)))
#     ly = "%c" % (chr(int(iy+97)))
      # get character labels (1-52 set; note that range is [0,1] not [1,n])
      lx = "%c" % (chr(int(ix+97))) if ix <= 25 else (chr(int((ix-26)+65)))
      ly = "%c" % (chr(int(iy+97))) if iy <= 25 else (chr(int((iy-26)+65)))
      # output AOI label, fixation duration, count
      str = "%s,%s,%c%c,%f,%d\n" % (subj, cond, lx, ly, fx.getDuration(), ct)
      outfile.write(str)
      ct += 1
    outfile.close()

  def entropy(self,fileName,subj,cond,w,h):

    outfile = open(fileName,'w')
    str = "subj,cond,entropy\n"
    outfile.write(str)

    # diagonal
    d = math.sqrt(float(w)*float(w) + float(h)*float(h))
    sigma = 0.0

    # heatmap: a 32-bit floating point image, initially set to black
    lum = Image.new("F", (int(w),int(h)), 0.0)
    pix = lum.load()

    for fx in self.fixations:
      x = fx.at(0) * float(w)
      # do the y-coord flip for rendering with (0,0) at bottom-left
      y = (1.0 - fx.at(1)) * float(h)
      # hack: if there's only 1 fixation @ 0% duration: 1/6th of image
      sigma = fx.getPercentDuration() * d if fx.getPercentDuration() > 0 else d/6.0
 #    for i in range(int(h)):
 #      for j in range(int(w)):
      for i in xrange(int(y-2.0*sigma),int(y+2.0*sigma)):
        for j in xrange(int(x-2.0*sigma),int(x+2.0*sigma)):
          if( 0 <= i and i < int(h) and 0 <= j and j < int(w) ):
            sx = j - x
            sy = i - y
            heat = math.exp((sx*sx +sy*sy)/(-2.0*sigma*sigma))
            pix[j,i] = pix[j,i] + heat

    # get max value
    minlum, maxlum = lum.getextrema()

    # normalize
    lum = lum.point(lambda f: f * (1.0/maxlum) + 0)
    pix = lum.load()

    # compute lum entropy
    gray_freq = [0.0]*256;
    for i in range(int(h)):
      for j in range(int(w)):
        idx = int(pix[j,i] * 255.0)
 #      print "pix[%d,%d] = " % (j,i)
 #      print "%f, idx = %d" % (pix[j,i], idx)
        gray_freq[idx] = gray_freq[idx] + 1

    prob = 0.0
    entropy = 0.0
    for i in range(256):
      if(gray_freq[i] == 0):
        pass
      else:
        prob = float(gray_freq[i]/(float(w)*float(h)))
        if(prob > 0.0):
          entropy = entropy - (prob * np.log2(prob))

    # normalize entropy since max entropy is 2^8
    entropy = entropy / np.log2(256.0)

    # print entropy
    str = "%s,%s,%f\n" % (subj, cond, entropy)
    outfile.write(str)
    outfile.close()

  def dumpFixatedAOIs(self,fileName,w,h,aoidict,key):

    outfile = open(fileName,'w')
    str = "aoi_order,timestamp,x,y,duration,prev_sacc_amplitude,aoi_label,aoi_line,word,word_length_signs\n"
    outfile.write(str)
    i=0
    prev_fixation=0
    for fx in self.fixations:
      x = fx.at(0) * float(w)
      y = fx.at(1) * float(h)
      inAOI = False
      for aoi in aoidict[key]:
        if aoi.inside(x,y + aoi.getHeight()):
          inAOI = True
          break
      if inAOI:
        prev_sacc_amp = 0.0
        if prev_fixation > 0:
          sx = self.fixations[prev_fixation].at(0) * float(w)
          sy = self.fixations[prev_fixation].at(1) * float(h)
          dx = x - sx
          dy = y - sy
          prev_sacc_amp = math.sqrt(float(dx)*float(dx) + float(dy)*float(dy))
        # with word_type
#       str = "%s,%f,%f,%f,%f,%f,%s,%s,%s,%s,%s\n" % \
#             (aoi.getAOIOrder(),\
#              fx.gettimestamp(),x,y,fx.getDuration(),prev_sacc_amp,\
#              aoi.getAOILabel(),\
#              aoi.getAOILine(),\
#              aoi.getWord(),\
#              aoi.getWordLengthSigns(),\
#              aoi.getWordType()\
#             )
        # no word_type
        str = "%s,%f,%f,%f,%f,%f,%s,%s,%s,%s\n" % \
              (aoi.getAOIOrder(),\
               fx.gettimestamp(),x,y,fx.getDuration(),prev_sacc_amp,\
               aoi.getAOILabel(),\
               aoi.getAOILine(),\
               aoi.getWord(),\
               aoi.getWordLengthSigns()\
              )
        outfile.write(str)
        prev_fixation=i
      i += 1
    outfile.close()

  def clamp(self, x, y):
    if x < 0:
      x = 0.0
    if y < 0:
      y = 0.0
    if x > 1:
      x = 1.0
    if y > 1:
      y = 1.0
    return x,y
