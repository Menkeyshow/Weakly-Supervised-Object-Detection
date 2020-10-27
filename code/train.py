import glob, os, logging, time, argparse
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from skimage.draw import rectangle, polygon
import shutil # for removing temp

from base.base import base_model 
from util.generator.DataLoader import DataLoader

if __name__ == "__main__":

    # Parse the configuration from bash
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='data path', default='../data/train/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--label', type=str, default='ball', help='type of label')
    parser.add_argument('--approach', type=str, default='grabcut', help='which approach to use')
    parser.add_argument('--data_aug', type=bool, default=False)
    parser.add_argument('--use_weight', type=str, default='none')
    parser.add_argument('--save_weight', type=str, help='save name', default='ball.h5')

    args = parser.parse_args()
    PATH = args.path
    LABEL = args.label
    BATCHSIZE = args.batch_size
    EPOCHS = args.epoch
    APPROACH = args.approach

    # Use weights path
    if args.use_weight == 'none':
        USE_WEIGHT = 'none'
    else:
        USE_WEIGHT = '../data/weights/%s' % args.use_weight
    
    # Save weights path
    if not(os.path.exists('../data/weights/')):
        os.mkdir('../data/weights')
    SAVE_WEIGHT = '../data/weights/%s' % args.save_weight

    # set up DataLoader
    # the label decides how detailed
    data = DataLoader(PATH, LABEL, BATCHSIZE)
    if os.path.exists('../data/temp'):
        print('removed old temp!')
        shutil.rmtree('../data/temp')
    os.mkdir('../data/temp')

    # load neural network
    model_object = base_model()
    model = model_object.getModel()
    if not(USE_WEIGHT == 'none'):
        model.load_weights(USE_WEIGHT)

    # set up logger
    if not(os.path.exists('../data/logs/')):
        os.mkdir('../data/logs')
    LOGPATH = '../data/logs/train.log'
    logging.basicConfig(filename=LOGPATH, format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # set up training
    print('Begin Training')
    training_time_begin = time.time()
    logger.info('      ')
    logger.info('Begin Training at %s' % time.asctime(time.localtime(time.time())))
    logger.info('Epochs: %s' % EPOCHS)
    logger.info('Label: %s' % LABEL)
    logger.info('Approach: %s' % APPROACH)
    logger.info('Using data from: %s' % PATH)

    if not (USE_WEIGHT == 'none'):
        logger.info('Use existing weights: %s' % USE_WEIGHT)
    if USE_WEIGHT == 'none':
        logger.info('Weights: none ==> new weights')

    if not(USE_WEIGHT == 'none'):
        data.shuffle()
        data.reset_image_counter()
        print("Predicting batches for first training epoch on trained weights")
        for batch_number in range(data.get_number_batches()):
            data.calculate_next_batch()
            batch, _ = data.get_current_batch_image_label()
            predictions = model.predict(batch)
            data.save_labels_for_batch_simple_does_it(predictions)
        logger.info('Done predicting batches')
        print('Done predicting batches')

    # Training
    for epoch in range(EPOCHS):
        data.shuffle()
        data.reset_image_counter()
        logger.info('Start of epoch #%s' % epoch)
        
        for batch_number in range(data.get_number_batches()):
            print('Trainingepoch: ', epoch + 1, ' of ', EPOCHS)
            print('Batch: ', batch_number + 1, ' of ', data.get_number_batches())

            # use only chosen label
            if APPROACH == 'none':
                data.calculate_next_batch()

            # use chosen label one time and then recursive training without denoising
            if APPROACH == 'naive':
                data.calculate_next_batch_simple_does_it_naive()

            #use chosen one time and use denoised recursive training
            if APPROACH == 'box':
                data.calculate_next_batch_simple_does_it_box()
            
            #use chosen one time and reset outside bbox and iou >50%
            if APPROACH == 'box_iou':
                data.calculate_next_batch_simple_does_it_box_iou()
            
            #use grabcut and no recursive training
            if APPROACH == 'grabcut':
                data.calculate_next_batch_simple_does_it_grabcut()

            batch, labels = data.get_current_batch_image_label()
            model.train_on_batch(batch, labels)
            print('trained on these images via, %s' %(APPROACH))

        # predict for recursive training
        if ((APPROACH == 'naive') or (APPROACH == 'box') or (APPROACH == 'box_iou')):
            data.shuffle()
            data.reset_image_counter()
            logger.info('Predicting after training on epoch #%s' % int(epoch + 1))
            print("Predicting batches now")
            for batch_number in range(data.get_number_batches()):
                data.calculate_next_batch()
                batch, _ = data.get_current_batch_image_label()
                predictions = model.predict(batch)
                print('saving predictions')
                data.save_labels_for_batch_simple_does_it(predictions)
            logger.info('Done predicting batches')
            print('Done predicting batches')

    # clean up
    model.save_weights(SAVE_WEIGHT)
    training_time_end = time.time()
    training_duration = (training_time_end - training_time_begin) / 60 /60
    logger.info('Training finished at %s' % time.asctime(time.localtime(time.time())))
    logger.info('Training Runtime: %s hours' % training_duration)
    logger.info('Weights saved in %s' % SAVE_WEIGHT)
    print('done training :) and saved as %s' %(SAVE_WEIGHT))
    shutil.rmtree('../data/temp')
