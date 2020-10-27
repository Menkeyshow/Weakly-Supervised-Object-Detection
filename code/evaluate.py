#coding=utf8
import sys, os, argparse, logging, glob, time, importlib

#Math Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import util.util as util

#Keras Imports
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import pydot

from base.base import base_model 
from util.generator.DataLoader import DataLoader

def save_imagebatch_with_pediction_overlay(imagebatch, predictions, weight):
    imagenumber = 1

    # create directory that dont exist in data
    if not(os.path.exists('../data/evaluation_images/')):
        os.mkdir('../data/evaluation_images')
    weight_name = weight.split('.')[0]
    if not(os.path.exists('../data/evaluation_images/%s' % weight_name)):
        os.mkdir('../data/evaluation_images/%s' % weight_name)
    
    # create image with overlay
    for i in range(BATCHSIZE):
        heatmap = predictions[i]
        image = imagebatch[i]
        heatmap[heatmap < 0.5] = 0
        heatmap[heatmap >= 0.5] = 1
        plt.imshow(np.squeeze(heatmap))
        plt.imshow(image.astype('uint8'), alpha=0.6)
        plt.axis('off')
        yellow = mpatches.Patch(color='yellow', label='prediction')
        plt.legend(handles=[yellow])
        plt.savefig('../data/evaluation_images/%s/results_%s.png'%(weight_name, imagenumber))
        plt.close()
        plt.imshow(image.astype('uint8'))
        plt.axis('off')
        plt.savefig('../data/evaluation_images/%s/results_%s_original.png' % (weight_name, imagenumber))
        imagenumber = imagenumber + 1

    logger.info('Saved predicted labels')
    
if __name__ == "__main__":
    # set up configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='data path', default='../data/test/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--label', type=str, default='ball_round', help='type of label')
    parser.add_argument('--weight', type=str, help='weight in ../data/weights/')
    
    args = parser.parse_args()
    PATH = args.path
    LABEL = args.label
    BATCHSIZE = args.batch_size
    WEIGHT = args.weight 
    LOGPATH = '../data/logs/results_%s_%s.log' % (WEIGHT, LABEL)

    # set up data
    data = DataLoader(PATH, LABEL)

    # load neural network and to test weight
    model_object = base_model()
    model = model_object.getModel()
    model.load_weights('../data/weights/%s' % WEIGHT)

    # set up logger
    if not(os.path.exists('../data/logs/')):
        os.mkdir('../data/logs')
    logging.basicConfig(filename=LOGPATH, format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # set up metric variables
    Accuracy = []
    Precision = []
    Recall = []
    Intersection_over_union = []

    # begin evaluation
    eval_time_begin = time.time()
    logger.info('Begin evaluating at %s' % time.asctime(time.localtime(time.time())))
    logger.info('Weights: %s' % WEIGHT)
    logger.info('Using data from path: %s' % PATH)

    for batch_number in range(data.get_number_batches()):
        # get data for batch
        data.calculate_next_batch()
        batch, labels = data.get_current_batch_image_label()
        predictions = model.predict(batch)
        
        # save results form first batch as images
        if batch_number == 0:
            save_imagebatch_with_pediction_overlay(batch, predictions, WEIGHT)

        # calculate the metrics
        for label, prediction in zip(labels, predictions):
            assert label.shape == prediction.shape
            label = util.normalize_predicted_Labels(label)
            prediction = util.normalize_predicted_Labels(prediction)
            accuracy, precision, recall, intersection_over_union = util.calculate_metrics(label, prediction)

            Accuracy.append(accuracy)
            Precision.append(precision)
            Recall.append(recall)
            Intersection_over_union.append(intersection_over_union)

    # calculate percentiles
    Accuracy, Precision, Recall, Intersection_over_union = util.accumulate_metrics(Accuracy, Precision, Recall, Intersection_over_union) # on accumulated lists
    
    # calculate eval duration
    eval_time_end = time.time()
    eval_time = (eval_time_end - eval_time_begin) / 60

    # logging metrics
    logger.info('Accuracy:\tmean:%.7f, 90th:%.7f, 99th:%.7f' , Accuracy[0], Accuracy[1], Accuracy[2])
    logger.info('Precision:\tmean:%.7f, 90th:%.7f, 99th:%.7f', Precision[0], Precision[1], Precision[2])
    logger.info('Recall:\t\tmean:%.7f, 90th:%.7f, 99th:%.7f', Recall[0], Recall[1], Recall[2])
    logger.info('IoU:\t\tmean:%.7f, 90th:%.7f, 99th:%.7f', Intersection_over_union[0], Intersection_over_union[1], Intersection_over_union[2])
    logger.info('End evaluation after %s minutes' % eval_time)
    #print(Accuracy, Precision, Recall, Intersection_over_union)

    print('done evaluation')
