./run_gsks.sh > data
python $HMLP_DIR/python/utils/spreadsheet.py --prim gsks

./run_gsknn.sh > data
python $HMLP_DIR/python/utils/spreadsheet.py --prim gsknn

./run_conv2d.sh > data
python $HMLP_DIR/python/utils/spreadsheet.py --prim conv2d
