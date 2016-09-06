set PYTHON=python

set INDIR=../../exp/jaconde/
set IMGDIR=../../stimulus/jaconde/
set IMGDIR=../../stimulus/jaconde/

set PLTDIR=./plots/
set OUTDIR=./data/
set RAWDIR=./data/raw/

REM raw
%PYTHON% csv2raw.py --indir=%INDIR% --outdir=%RAWDIR%
