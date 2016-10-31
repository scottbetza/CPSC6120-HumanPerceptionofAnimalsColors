#!/usr/bin/env python2

import sys,os,getopt,glob
import locale
import numpy as np

def usage():
  print "Usage: python csv2raw.py " \
        " --indir=? -outdir=?\n" \
        " --file=?\n" \
        "   indir: a directory containing input files to be processed\n" \
        "   outdir: a directory containing output files\n" \
        "   file: a single file to process\n"

# convert csv file to raw
def csv2raw(infile,outdir):
  try:
    f = open(infile,'r')
  except IOError:
    print "Can't open file: " + infile
    return

  # parse the subject name from the file name
# base = infile[:infile.rfind('.')]
  base = os.path.basename(infile)
  print "Processing: ", infile, "[", base, "]"

  # strip out extension
  filename, ext = os.path.splitext(base)
  print 'Processing: ' + filename

  outfile = None

  # read lines, throwing away first one (header)
  linelist = f.read().splitlines()
  header = linelist[0].split(',')
  linelist = linelist[1:]

  for idx, label in enumerate(header):
    if "TIME(" in label.strip():
      TIME = idx
    if label.strip() == "BPOGX":
      BPOGX = idx
    if label.strip() == "BPOGY":
      BPOGY = idx
    if label.strip() == "MEDIA_NAME":
      MEDIA_NAME = idx

  # reset coords
  subj = "User x"
  x = ''
  y = ''
  t = ''
  stimulus = "Blank"

  # process each line, splitting on ','
  for line in linelist:
    elements = line.split(',')
    entry = np.asarray(elements)

    if entry[MEDIA_NAME] != stimulus:
      stimulus = entry[MEDIA_NAME]

      # parse the subject name from the file name
      subj = filename[:filename.find('_')]

      # new stimulus, open new file
      print 'Processing: ' + subj + ' (' + stimulus + ')'

      if outfile is not None:
        outfile.close()

      outfile = open(outdir + subj + '-' + stimulus + '.raw','w+')

      # reset coords
      x = ''
      y = ''
      t = ''

    # dump out gaze coordinates
    if(outfile is not None and
       entry[BPOGX] != x and
       entry[BPOGY] != y and
       entry[TIME] != t):

       # data already normalized, no need to divide by screen size
       x = str(float(entry[BPOGX]))
       y = str(float(entry[BPOGY]))
       t = entry[TIME]

       # don't process empty or negative coords
#      if locale.atof(x) < 0 or locale.atof(y) < 0:
#        print "locale data: ",locale.atof(x),locale.atof(y),"*******"
#      else:
#        print "locale data: ",locale.atof(x),locale.atof(y)
       if(x != '' and y != '' and \
          locale.atof(x) < 1 and locale.atof(y) < 1 and \
          locale.atof(x) >= 0 and locale.atof(y) >= 0):
         strout = "%f %f %f" % (locale.atof(x),locale.atof(y),float(t))
         outfile.write(strout + '\n')

def main(argv):
# if not len(argv):
#   usage()

  try:
    opts, args = getopt.getopt(argv, '', \
                 ['indir=','outdir=','file=',\
                  'hertz=','sfdegree=','sfcutoff='])
  except getopt.GetoptError, err:
    usage()
    exit()

  file = None

  for opt,arg in opts:
    opt = opt.lower()
    if(opt != '--file'):
      arg = arg.lower()

    if opt == '--indir':
      indir = arg
    elif opt == '--outdir':
      outdir = arg
    else:
      sys.argv[1:]

  # get .csv input files to process
  if os.path.isdir(indir):
    files = glob.glob('%s/User*_all_gaze.csv' % (indir))

  # if user specified --file="..." then we use that as the only one to process
  if(file != None and os.path.isfile(file)):
    files = [file]

  for file in files:
    csv2raw(file,outdir)

#######################
#locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' )

if __name__ == "__main__":
  main(sys.argv[1:])
