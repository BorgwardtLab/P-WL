for DATASET in MUTAG; do
	python3 main.py ../data/${DATASET}/*.gml -d ${DATASET} -n 10 -l ../data/${DATASET}/Labels.txt -g
done

