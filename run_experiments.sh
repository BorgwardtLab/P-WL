for DATASET in ENZYMES; do
	screen -dmS ${DATASET}
	screen -S ${DATASET} -X stuff 'python main.py data/'${DATASET}'/*.gml -d '${DATASET}' -n 10 -d '${DATASET}' -l data/'${DATASET}'/Labels.txt -g\n'
done

