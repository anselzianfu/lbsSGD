#!/bin/bash
python3 src/plot_logs.py --across --logdir ./logs --fields test_acc --export ./plots/ --agg max --experiments baseline,hessian_vec --batch_sizes 128,256,512,1024,2048,4096 asdf
