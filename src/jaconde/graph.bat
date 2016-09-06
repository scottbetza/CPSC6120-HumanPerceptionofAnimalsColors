set PYTHON=python
::set WIDTH=1600
::set HEIGHT=900
set WIDTH=2048
set HEIGHT=1536
set HERTZ=150
set DIST=25.59
set SCREEN=13
set XTILES=16
set YTILES=12

REM use Butterworth?
set SMOOTH=False

set INDIR=../../exp/jaconde/
set IMGDIR=../../stimulus/jaconde/

set PLTDIR=./plots/
set OUTDIR=./data/
set RAWDIR=./data/raw/

REM process
%PYTHON% graph.py --smooth=%SMOOTH% --indir=%RAWDIR% --imgdir=%IMGDIR% --dist=%DIST% --screen=%SCREEN% --width=%WIDTH% --height=%HEIGHT% --hertz=%HERTZ% --xtiles=%XTILES% --ytiles=%YTILES% --outdir=%OUTDIR% --pltdir=%PLTDIR%
