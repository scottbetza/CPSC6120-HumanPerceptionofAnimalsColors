#!/usr/bin/env python2

# system includes
import sys
import os
import numpy as np

# local includes
from aoi import AOI

def main(argv):

  indir = "../aoi/"

  aoi = AOI(179,764.0,211,108)

  aoi.dump()

  print "in" if aoi.inside(180.0,780) else "out"
  print "in" if aoi.inside(0.0,0.0) else "out"

  # process AOI file
  aoidir = "../aoi/"
  aoifile = aoidir + "aoi_definition.csv"
  print "aoifile = ", aoifile

  aoidict = {}

  if(os.path.isfile(aoifile)):
    f = open(aoifile,'r')
    linelist = f.read().splitlines()
    header = linelist[0].split(',')
    linelist = linelist[1:]

    for idx, label in enumerate(header):
      if label.strip() == "task_id":
        TASK_ID = idx
      if label.strip() == "aoi_label":
        AOI_LABEL = idx
      if label.strip() == "x_bl":
        X_BL = idx
      if label.strip() == "y_bl":
        Y_BL = idx
      if label.strip() == "x_br":
        X_BR = idx
      if label.strip() == "y_br":
        Y_BR = idx
      if label.strip() == "x_tr":
        X_TR = idx
      if label.strip() == "y_tr":
        Y_TR = idx
      if label.strip() == "x_tl":
        X_TL = idx
      if label.strip() == "y_tl":
        Y_TL = idx

    for line in linelist:
      entry = line.split(',')
      stimulus = entry[TASK_ID].replace('\'','')
      type = entry[AOI_LABEL]
#     print stimulus,",",type

      x_bl = float(entry[X_BL])
      y_bl = float(entry[Y_BL])

      x_br = float(entry[X_BR])
      y_br = float(entry[Y_BR])

      x_tr = float(entry[X_TR])
      y_tr = float(entry[Y_TR])

      x_tl = float(entry[X_TL])
      y_tl = float(entry[Y_TL])

      aoi = AOI(x_bl,y_bl,x_br - x_bl,y_tl - y_bl)
#     aoi.dump()
      if aoidict.has_key(stimulus):
        aoidict[stimulus].append(aoi)
      else:
        aoidict[stimulus] = [aoi]

      del aoi

# print aoidict
  for key in aoidict:
    print key
    print "number of AOIs: ",len(aoidict[key])
    for aoi in aoidict[key]:
      aoi.dump()

if __name__ == "__main__":
  main(sys.argv[1:])
