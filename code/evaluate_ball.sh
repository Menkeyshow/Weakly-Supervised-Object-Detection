#!/bin/bash

#source ../BA/bin/activate
python evaluate.py --path '../data/test/' --weight 'ball_5_none.h5' --label 'ball_round'
python evaluate.py --path '../data/test/' --weight 'ball_5_naive.h5' --label 'ball_round'
python evaluate.py --path '../data/test/' --weight 'ball_5_box.h5' --label 'ball_round'

python evaluate.py --path '../data/test/' --weight 'ball_15_none.h5' --label 'ball_round'
python evaluate.py --path '../data/test/' --weight 'ball_15_naive.h5' --label 'ball_round'
python evaluate.py --path '../data/test/' --weight 'ball_15_box.h5' --label 'ball_round'

python evaluate.py --path '../data/test/' --weight 'ball_25_none.h5' --label 'ball_round'
python evaluate.py --path '../data/test/' --weight 'ball_25_naive.h5' --label 'ball_round'
python evaluate.py --path '../data/test/' --weight 'ball_25_box.h5' --label 'ball_round'
