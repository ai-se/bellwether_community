#!/bin/csh
#BSUB -W 6000
#BSUB -n 8
#BSUB -R span[ptile=8]
#BSUB -o ./out/out.%J
#BSUB -e ./out/out.%J

module load python
python get_attributes.py
