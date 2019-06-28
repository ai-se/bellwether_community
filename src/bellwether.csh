#!/bin/csh
#BSUB -W 6000
#BSUB -n 4
#BSUB -o ./out/out.%J
#BSUB -e ./out/out.%J

module load python
python bellwether_v2.py
