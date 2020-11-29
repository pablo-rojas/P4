#!/bin/bash

## \file
## \TODO This file implements a very trivial feature extraction; use it as a template for other front ends.
## 
## Please, read SPTK documentation and some papers in order to implement more advanced front ends.

# Base name for temporary files
base=/tmp/$(basename $0).$$ 

# Ensure cleanup of temporary files on exit
trap cleanup EXIT
cleanup() {
   \rm -f $base.*
}

if [[ $# != 3 || $# != 4 || $# != 5]]; then
   echo "$0 lpc_order input.wav output.lp"
   echo "--Alternatively--"
   echo "$0 lpc_order cepstrum_order input.wav output.lp"
   echo "--Alternatively--"
   echo "$0 lpc_order cepstrum_order mfcc_order input.wav output.lp"
   exit 1
fi

if [[$# == 3]]; then
   lpc_order=$1
   inputfile=$2
   outputfile=$3
elif [[$# == 4]]; then
   lpc_order=$1
   cepstrum_order=$2
   inputfile=$3
   outputfile=$4
elif [[$# == 5]]; then
   lpc_order=$1
   cepstrum_order=$2
   mfcc_order=$3
   inputfile=$4
   outputfile=$5
fi

UBUNTU_SPTK=1
if [[ $UBUNTU_SPTK == 1 ]]; then
   # In case you install SPTK using debian package (apt-get)
   X2X="sptk x2x"
   FRAME="sptk frame"
   WINDOW="sptk window"
   LPC="sptk lpc"
   LPCC="sptk lpc2c"
   MFCC="sptk mfcc"
else
   # or install SPTK building it from its source
   X2X="x2x"
   FRAME="frame"
   WINDOW="window"
   LPC="lpc"
   LPCC="lpc2c"
   MFCC="mfcc"
fi

# Main command for feature extration
sox $inputfile -t raw -e signed -b 16 - | $X2X +sf | $FRAME -l 240 -p 80 | $WINDOW -l 240 -L 240 |
	$LPC -l 240 -m $lpc_order | $LPCC -m $lpc_order -M $cepstrum_order |
	$MFCC -a 0.97 -c 22 -e 1 -s 16 -l 240 -L 256 -m $mfcc_order -n 20 -w 1 > $base.lp

# Our array files need a header with the number of cols and rows:
ncol=$((lpc_order+1)) # lpc p =>  (gain a1 a2 ... ap) 
nrow=`$X2X +fa < $base.lp | wc -l | perl -ne 'print $_/'$ncol', "\n";'`

# Build fmatrix file by placing nrow and ncol in front, and the data after them
echo $nrow $ncol | $X2X +aI > $outputfile
cat $base.lp >> $outputfile

exit

# -- MFCC -- #
#–a A preemphasise coefficient [0.97]
#–c C liftering coefficient [22]
#–e E flooring value for calculating log(x) in filterbank analysis [1.0]
   #if x < E then return x = E
#–s S sampling frequency (kHz) [16.0]
#–l L1 frame length of input [256]
#–L L2 frame length for fft. default value 2n satisfies L1 < 2n [2n]
#–m M order of mfcc [12]
#–n N order of channel for mel-filter bank [20]
#–w W type of window [0]
   #0 Hamming
   #1 Do not use a window function
#–d use dft (without using fft) for dct [FALSE]
#–E output energy [FALSE]
#–0 output 0’th static coefficient [FALSE]