# Starts an experiment about the number of iterations on the PTC-MR data
# set. The performance of the *original* features is plotted against the
# performance of the *topological* features, including cycles.

GRAPHS=../data/PTC_MR/*.gml
LABELS=../data/PTC_MR/Labels.txt
SCRIPT=../src/main.py

echo "h acc_topological_c sdev_topological_c acc_original sdev_original acc_topological_smooth sdev_topological_smooth"

for h in `seq 0 10`; do
  ACCURACY_TOP=`$SCRIPT       -p 1 -c -n $h $GRAPHS -l $LABELS 2>&1 | grep Accuracy  | cut -f 2,4 -d " "`
  ACCURACY_ORG=`$SCRIPT            -u -n $h $GRAPHS -l $LABELS 2>&1 | grep Accuracy  | cut -f 2,4 -d " "`
  ACCURACY_SMOOTH=`$SCRIPT -S -p 1 -c -n $h $GRAPHS -l $LABELS 2>&1 | grep Accuracy  | cut -f 2,4 -d " "`
  echo $h $ACCURACY_TOP $ACCURACY_ORG $ACCURACY_SMOOTH
done

