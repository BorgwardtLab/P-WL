#!/usr/bin/env bash

for a in ../data/MUTAG; do 
   python3 grid_search.py -c -n 10 $a/*.gml -l $a/Labels.txt
   python3 grid_search.py -s -n 10 $a/*.gml -l $a/Labels.txt
done

