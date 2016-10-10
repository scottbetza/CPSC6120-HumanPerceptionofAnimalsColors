#!/usr/bin/env python

import sys, os, math

def catCSVFile(infile,df,ct):
  try:
    f = open(infile,'r')
  except IOError:
    print "Can't open file: " + infile
    return

  base = os.path.basename(infile)

  print "Processing: ", infile, "[", base, "]"

  # extract stimulus name and subj id
  stimulus = base.split('_')[1].split('-')[0]
  subj = base.split('_')[0]
  if stimulus == 'image1':
    stimulus = "image1"
  elif stimulus == 'p2':
    stimulus = "painting1016x1536"
  print "stim = ", stimulus
  print "subj = ", subj

  # read lines, throwing away first one (header)
# linelist = f.readlines()
# linelist = f.read().split('\r')
  linelist = f.read().splitlines()
  header = linelist[0].split(',')
  linelist = linelist[1:]

  # timestamp,x,y,duration,prev_sacc_amplitude,aoi_label
  for idx, label in enumerate(header):
    if label.strip() == "timestamp":
      TIMESTAMP = idx
    if label.strip() == "x":
      X = idx
    if label.strip() == "y":
      Y = idx
    if label.strip() == "duration":
      DURATION = idx
    if label.strip() == "prev_sacc_amplitude":
      PREV_SACC_AMPLITUDE = idx
    if label.strip() == "aoi_label":
      AOI_LABEL = idx

  for line in linelist:
    entry = line.split(',')

    # get line elements
    timestamp = entry[TIMESTAMP]
    x  = entry[X]
    y  = entry[Y]
    duration  = entry[DURATION]
    prev_sacc_amplitude  = entry[PREV_SACC_AMPLITUDE]
    aoi_label  = entry[AOI_LABEL]

    str = "%s,%s,%s,%s,%s,%s,%s,%s" % ( \
                                    subj, \
                                    stimulus,\
                                    timestamp,\
                                    x,y,\
                                    duration,\
                                    prev_sacc_amplitude,\
                                    aoi_label)
    print >> df,str
    ct += 1

  return ct

###############################################################################

# clear out output file
df = open("fxtn-aois.csv",'w')
print >> df,"subj,stimulus,timestamp,x,y,duration,prev_sacc_amplitude,aoi_label"

dir = './data/'

# find all files in dir with .csv extension
lst = filter(lambda a: a.endswith('-fxtn-aoi.csv'),os.listdir(dir))

lineno = 1

for item in lst:

  file = dir + item
  print 'Processing ', file

  # cat csv files into one
  lineno = catCSVFile(file,df,lineno)

df.close()
