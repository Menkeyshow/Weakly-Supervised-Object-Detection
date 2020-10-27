#!/bin/bash

#source ../BA/bin/activate
python train.py --path "../data/train/" --epoch 5 --label "ball" --approach "none" --use_weight "none" --save_weight "ball_none_5.h5"
python train.py --path "../data/train/" --epoch 10 --label "ball" --approach "none" --use_weight "ball_none_5.h5" --save_weight "ball_none_15.h5"
python train.py --path "../data/train/" --epoch 10 --label "ball" --approach "none" --use_weight "ball_none_15.h5" --save_weight "ball_none_25.h5"

python train.py --path "../data/train/" --epoch 5 --label "ball" --approach "naive" --use_weight "none" --save_weight "ball_naive_5.h5"
python train.py --path "../data/train/" --epoch 10 --label "ball" --approach "naive" --use_weight "ball_naive_5.h5" --save_weight "ball_naive_15.h5"
python train.py --path "../data/train/" --epoch 10 --label "ball" --approach "naive" --use_weight "ball_naive_15.h5" --save_weight "ball_naive_25.h5"

python train.py --path "../data/train/" --epoch 5 --label "ball" --approach "box" --use_weight "none" --save_weight "ball_box_5.h5"
python train.py --path "../data/train/" --epoch 10 --label "ball" --approach "box" --use_weight "ball_box_5.h5" --save_weight "ball_box_15.h5"
python train.py --path "../data/train/" --epoch 10 --label "ball" --approach "box" --use_weight "ball_box_15.h5" --save_weight "ball_box_25.h5"
