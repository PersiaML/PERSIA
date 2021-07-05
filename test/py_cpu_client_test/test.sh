cp $OUTPUT_DIR/libpersia_embedding_py_cpu_client_sharded_server.so $OUTPUT_DIR/persia_embedding_py_cpu_client_sharded_server.so 

export PYTHONPATH=$PYTHONPATH:$OUTPUT_DIR/
basepath=`dirname $0`
python3.7 $basepath/test.py