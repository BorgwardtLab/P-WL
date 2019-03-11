#!/usr/env/bash

for a in data/*; do 
   python grid_search.py -u -c -n 10 $a/*.gml -l $a/Labels.txt
   python grid_search.py -s -n 10 $a/*.gml -l $a/Labels.txt
done

