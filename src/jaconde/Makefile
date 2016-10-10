UNAME           = $(shell uname)

ifeq ($(UNAME),Linux)
#PYTHON = /opt/modules/python/2.7/bin/python2.7
PYTHON = /usr/bin/python2
endif

ifeq ($(UNAME),Darwin)
#PYTHON = /usr/local/bin/python
#PYTHON = /usr//bin/python
#PYTHON = /sw//bin/python2.7
PYTHON = python
R = /usr/local/bin/R
endif

# use Butterworth?
SMOOTH = False

#WIDTH = 1600
#HEIGHT = 900
WIDTH = 2048
HEIGHT = 1536
HERTZ = 150
DIST = 25.59 # 65 cm
#DIST = 23.62 # 60 cm
#DIST = 19.68 # 50 cm
#DIST = 15.74 # 40 cm
SCREEN = 13

#XTILES = 4
#YTILES = 3
XTILES = 16
YTILES = 12

INDIR = ../../exp/jaconde/
#IMGDIR = ../../stimulus/blank/
IMGDIR = ../../stimulus/jaconde/
#IMGDIR = ../../stimulus/blank/

PLTDIR = ./plots/
OUTDIR = ./data/
RAWDIR = ./data/raw/

#all: dirs raw process graph
#all: dirs raw process collate
all: dirs raw process collate stats
#all: dirs raw graph

dirs:
	mkdir -p data
	mkdir -p data/raw
	mkdir -p plots

raw:
	$(PYTHON) ./csv2raw.py --indir=$(INDIR) --outdir=$(RAWDIR)

process:
	$(PYTHON) ./filter.py --smooth=$(SMOOTH) --indir=$(RAWDIR) --imgdir=$(IMGDIR) --dist=$(DIST) --screen=$(SCREEN) --width=$(WIDTH) --height=$(HEIGHT) --hertz=$(HERTZ) --xtiles=$(XTILES) --ytiles=$(YTILES)

graph:
	$(PYTHON) ./graph.py --smooth=$(SMOOTH) --indir=$(RAWDIR) --imgdir=$(IMGDIR) --dist=$(DIST) --screen=$(SCREEN) --width=$(WIDTH) --height=$(HEIGHT) --hertz=$(HERTZ) --xtiles=$(XTILES) --ytiles=$(YTILES) --outdir=$(OUTDIR) --pltdir=$(PLTDIR)
#	$(PYTHON) ./graph.py --smooth=$(SMOOTH) --indir=$(RAWDIR) --imgdir=$(IMGDIR) --dist=$(DIST) --screen=$(SCREEN) --width=$(WIDTH) --height=$(HEIGHT) --hertz=$(HERTZ) --xtiles=$(XTILES) --ytiles=$(YTILES) --outdir=$(OUTDIR) --pltdir=$(PLTDIR) --image="white-1600x900.png"

collate:
	$(PYTHON) ./collate-fxtn-aois.py

stats:
	$(R) --vanilla < fxtn-aois.R > fxtn-aois.out

clean: 
	rm -rf *.pyc 
	rm -rf plots/
	rm -rf data/
	rm -rf *.pdf
	rm -rf *.out
