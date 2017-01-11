./run_gsks.sh > data
python $HMLP_DIR/python/utils/spreadsheet.py

./run_gsknn.sh > data
python $HMLP_DIR/python/utils/spreadsheet.py

./run_conv2d.sh > data
python $HMLP_DIR/python/utils/spreadsheet.py

./run_strassen.sh > data
python $HMLP_DIR/python/utils/spreadsheet.py 
