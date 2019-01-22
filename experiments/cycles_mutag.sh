# Starts an experiment about the number of iterations on the MUTAG data
# set. The performance of the *original* features is plotted against the
# performance of the *topological* features, including cycles.

GRAPHS=../data/MUTAG/*.gml
LABELS=../data/MUTAG/Labels.txt
SCRIPT=../main.py

echo "h acc_topological_c sdev_topological_c acc_original sdev_original"

for h in `seq 0 10`; do
  ACCURACY_TOP=`$SCRIPT -p 1 -c -n $h $GRAPHS -l $LABELS 2>&1 | grep Accuracy  | cut -f 2,4 -d " "`
  ACCURACY_ORG=`$SCRIPT      -u -n $h $GRAPHS -l $LABELS 2>&1 | grep Accuracy  | cut -f 2,4 -d " "`
  echo $h $ACCURACY_TOP $ACCURACY_ORG
done

