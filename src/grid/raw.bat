set PYTHON=python

set INDIR=../../exp/grid/
set IMGDIR=../../stimulus/grid/

set PLTDIR=./plots/
set OUTDIR=./data/
set RAWDIR=./data/raw/

REM raw
%PYTHON% csv2raw.py --indir=%INDIR% --outdir=%RAWDIR%
