for DS in ../data/PTC*; do
    echo Handling $DS
    python3 grid_search_v_kernel.py $DS/*.gml -l $DS/Labels.txt
    python3 grid_search_e_kernel.py $DS/*.gml -l $DS/Labels.txt
done
