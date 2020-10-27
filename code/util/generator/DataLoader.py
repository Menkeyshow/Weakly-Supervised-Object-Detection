import glob, os, logging
import numpy as np
import cv2, math
from skimage.draw import rectangle, polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.set_printoptions(threshold=np.inf)
class DataLoader:
    def __init__(self, path, label = 'ball', batchsize = 16, temppath = '../data/temp'):
        self.path = path
        self.temppath = temppath
        self.label = label
        self.batchsize = batchsize
        self.all_image_paths = self.get_imagepaths()
        self.number_images = self.all_image_paths.shape[0]
        self.number_batches = math.ceil(self.number_images / self.batchsize)
        self.current_image_count = 0
        self.current_batch_images = []
        self.current_batch_labels = []
        self.current_batch_paths = []

    def Intersection_Over_Union_for_labels(self, truth, prediction):
        """
        Returns truth, if IoU of truth and prediction <50%, prediciton otherwise
        """
        area_of_overlap = 0.0
        area_of_union = 0.0
        for x in range(0, truth.shape[0]):
            for y in range(0, truth.shape[1]):
                if truth[x][y] == [1.0] and prediction[x][y] == [1.0]:
                    area_of_overlap = area_of_overlap + 1.0

                if truth[x][y] == [1.0] or prediction[x][y] == [1.0]:
                    area_of_union = area_of_union + 1.0
        try:
            intersection_over_union = area_of_overlap/area_of_union
        except ZeroDivisionError:
            intersection_over_union = 0.0
        if intersection_over_union >= 0.5:
            return prediction
        return truth

    def get_number_batches(self):
        """
        Returns the number of batches per epoch. (#images / batchsize)
        """
        return self.number_batches
    
    def shuffle(self):
        """
        Randomly arange the image paths array.
        """
        np.random.shuffle(self.all_image_paths)
        print('shuffled data paths')

    def reset_image_counter(self):
        """
        Reset the image counter of the image paths.
        This must be done per epoch.
        """
        self.current_image_count = 0
        print('counter was reset!')
    
    def get_current_batch_image_label(self):
        """
        Returns images, labels of the current batch.
        """
        return self.current_batch_images, self.current_batch_labels

    def get_current_batch_paths(self):
        """
        Returns the image paths of the current batch.
        """
        return self.current_batch_paths

    def calculate_next_batch_paths(self):
        """
        Calculates the image paths of the next batch.
        Only used in calculate_next_batch(self)
        """
        batch_paths = []

        for _ in range(self.batchsize):
            try:
                batch_paths.append(self.all_image_paths[self.current_image_count])
                self.current_image_count = self.current_image_count + 1
            except:
                #if the last batch has less than batchsize label:
                print('no images left or wrong directiory? Array has %s entries' % len(batch_paths))
                break
        self.current_batch_paths = batch_paths
        print('loaded ', self.current_image_count, 'images of ', self.number_images)

    def get_imagepaths(self):
        """
        Returns an array with all relative image paths of all sets.
        :param path: path to imagesets (eg ../data/train/')
        """
        
        _all_image_paths = np.array([])

        # paths of all sets
        _set_path_list = np.array(glob.glob(self.path + '*')) # paths of all sets
        _set_path_list = [p.replace("\\","/") for p in _set_path_list]

        # paths of all images
        for set_path in _set_path_list:
            _image_paths = np.array(glob.glob(set_path + '/*.jpg')) # image paths of one set jpg
            _image_paths = [p.replace("\\","/") for p in _image_paths]
            _all_image_paths = np.concatenate((_all_image_paths, _image_paths))

            _image_paths = np.array(glob.glob(set_path + '/*.png')) # image paths of one set PNG
            _image_paths = [p.replace("\\","/") for p in _image_paths]
            _all_image_paths = np.concatenate((_all_image_paths, _image_paths))

            _image_paths = np.array(glob.glob(set_path + '/*.jpeg')) # image paths of one set jpeg
            _image_paths = [p.replace("\\","/") for p in _image_paths]
            _all_image_paths = np.concatenate((_all_image_paths, _image_paths))

        print(_all_image_paths.shape, 'images were found')
        return _all_image_paths

    def get_image_info(self, image_path):
        """
        Returns set-name, filename and the textfilepath of a given image path
        :param image_path: relative! path to an image
        """
        # eg. ['..', 'data', 'train', 'bitbots-set00-01', 'jan16_seq__000.080.png']
        _image_info = image_path.split('/')
        _set_name= _image_info[-2]
        _image_name = _image_info[-1]

        #print(glob.glob(self.path + _set_name + '/*.txt'))
        if self.label == 'goalpost_poly':
            _text_file_path = glob.glob(self.path + _set_name + '/*goalpostpoly*.txt')[0]
        elif self.label == 'goalpost':
           _text_file_path = glob.glob(self.path + _set_name + '/*goalpost_*.txt')[0]
        else:
            _text_file_path = glob.glob(self.path + _set_name + '/*.txt')[0]
        _text_file_path = _text_file_path.replace("\\","/")

        return _set_name, _image_name, _text_file_path

    def get_label(self, image_path):
        """
        Returns a heatmap with the label of an image.
        ball = bbox ball label
        ball_round = round ball label
        :param image_path: relative path to an image
        """
        _set_name, _image_name, _text_file_path = self.get_image_info(image_path)
        _text_file = open(_text_file_path)
        _skip_label_info_lines_counter = 0
        _truth = None
        for _line in _text_file:
            _skip_label_info_lines_counter = _skip_label_info_lines_counter + 1
            if _skip_label_info_lines_counter > 6: # skip first lines
                _line = _line.strip() # remove whitespace and linebreaks
                #[format: "label::annotation_type|filename|img_width|img_height|x1|y1|x2|y2|center_x|center_y|width|height"]
                _split_line = _line.split('|')
                if not(_split_line == ['']): #jump over blank lines in export
                    if (_split_line[1] == _image_name) and not(_split_line[2] == 'not_in_image'):
                        if (self.label =='goalpost_poly') and (_split_line[0] == 'label::goalpost_poly'):
                            img_height = int(_split_line[3])
                            img_width = int(_split_line[2])
                            if _truth is None:
                                _truth = np.zeros((int(img_height), int(img_width),1),  dtype=np.float32)

                            x1 =int(_split_line[4])
                            x2 =int(_split_line[6])
                            x3 =int(_split_line[8])
                            x4 =int(_split_line[10])
                            y1 =int(_split_line[5])
                            y2 =int(_split_line[7])
                            y3 =int(_split_line[9])
                            y4 =int(_split_line[11])


                            c = np.array([y1,y2,y3,y4])
                            r = np.array([x1,x2,x3,x4])

                            cc, rr = polygon(r, c, (img_width, img_height))
                            _truth[rr, cc] = 1.0

                        else:
                            labeltype, filename, img_width, img_height, x1, y1, x2, y2, center_x, center_y, box_width, box_height = _split_line
                        if _truth is None:
                            _truth = np.zeros((int(img_height), int(img_width), 1),  dtype=np.float32)
                        if (self.label == 'ball_round') and (_split_line[0] == 'label::ball'):
                            center_x = float(_split_line[9])
                            center_y = float(_split_line[8])
                            bbwidth = float(_split_line[10])
                            bbheight = float(_split_line[11])

                            radius = min(bbheight, bbwidth)/2
                            width = _truth.shape[0]
                            height = _truth.shape[1]
                            x,y = np.ogrid[-center_x:width-center_x, -center_y:height-center_y]
                            mask = x*x + y*y < radius*radius
                            _truth[mask] = 1.0

                        if (self.label == 'ball') and (_split_line[0] == 'label::ball'):    
                            _x1y1 = (int(_split_line[4]),int(_split_line[5]))
                            _x2y2 = (int(_split_line[6]), int(_split_line[7]))
                            cv2.rectangle(_truth, _x1y1, _x2y2, (255,255,255), -1)
                            _truth = _truth / 255 

                        if (self.label == 'goalpost') and (_split_line[0] == 'label::goalpost'):
                            _x1y1 = (int(_split_line[4]),int(_split_line[5]))
                            _x2y2 = (int(_split_line[6]), int(_split_line[7]))
                            cv2.rectangle(_truth, _x1y1, _x2y2, (255,255,255), -1)
                            _truth = _truth / 255
                        if (self.label == 'robot') and (_split_line[0] == 'label::robot'):
                            _x1y1 = (int(_split_line[4]),int(_split_line[5]))
                            _x2y2 = (int(_split_line[6]), int(_split_line[7]))
                            cv2.rectangle(_truth, _x1y1, _x2y2, (255,255,255), -1)
                            _truth = _truth / 255 
        if _truth is None:
            _truth = np.zeros((800, 600, 1),  dtype=np.float32)
        _text_file.close
        _truth_resized = cv2.resize(_truth, (200, 150)) #shape (150,200)
        mask = _truth_resized > 0.0
        _truth_resized[mask] = 1.0
        return _truth_resized

    def get_image(self, image_path):
        """
        Returns an RGB image.
        param image_path: relative path to an image
        """
        # load image and convert to RGB
        print(image_path)
        _image = cv2.imread(image_path)[:, :, ::-1]
        #_image = plt.imread(image_path)
        _image_resized = cv2.resize(_image, (200, 150)) #(shape 150, 200 ,3)
        return _image_resized

    def save_image_and_labels(self, number):
        """
        Saves @number images and heatmaps to the cwd.
        param path: path to data
        param number: number of images to save
        """
        _images_saved = 0
        for _image_path in self.all_image_paths:
            _image = self.get_image(_image_path)
            _label = self.get_label(_image_path)
            #_label = np.reshape(_label, (150, 200, 1)) #not allowed or it doesnt save
            # print(_label)
            if _images_saved == number:
                break
            if _images_saved < number:
                plt.imsave(str(_images_saved)+'.png', _image)
                plt.imsave(str(_images_saved)+ 'label.png', _label)
                print('saved image number %s' %_images_saved)
                _images_saved = _images_saved + 1
        
    def save_specific_imaage_and_label(self, image_path):
        label = self.get_label(image_path)
        image = self.get_image(image_path)
        plt.imsave('test.png', image)
        plt.imsave('testlabel.png', label)

    def calculate_next_batch(self):
        """
        Calculates the next batch.
        Get values with get_next_batch().
        """
        self.calculate_next_batch_paths()
        number_of_current_paths = np.shape(self.current_batch_paths)[0]
        batch_images = np.empty((number_of_current_paths, 150, 200, 3))
        batch_labels = np.empty((number_of_current_paths, 150, 200, 1))

        for batch_image_number in range(number_of_current_paths):
            try:
                path = self.current_batch_paths[batch_image_number]
                image = self.get_image(path)
                label = self.get_label(path) # shape(150,200)
                label = label.reshape((150 ,200, 1)) # shape (150,200,1)

                batch_images[batch_image_number] = image
                batch_labels[batch_image_number] = label

            except:
                print('i have loaded all images!, reset counter!')

        self.current_batch_images = batch_images
        self.current_batch_labels = batch_labels

    def calculate_next_batch_simple_does_it_naive(self):
        """
        Use the labels saved from earlier predictions as groundtruth.
        """
        self.calculate_next_batch_paths()
        number_of_current_paths = np.shape(self.current_batch_paths)[0]
        batch_images = np.empty((number_of_current_paths, 150, 200, 3))
        batch_labels = np.empty((number_of_current_paths, 150, 200, 1))

        for batch_image_number in range(number_of_current_paths):
            
            #load original labels and image
            path = self.current_batch_paths[batch_image_number]
            image = self.get_image(path)
            bbox_label = self.get_label(path)
            bbox_label = bbox_label.reshape((150, 200, 1))

            # load images from temp
            try:
                set_path, image_name, _ = self.get_image_info(path)
                label = np.load('%s/%s/%s.npy' %(self.temppath, set_path, image_name))
                batch_labels[batch_image_number] = label
                print('loaded %s .label from temp' % batch_image_number)

            # load images from original data  
            except:
                print('loaded %s .label from original data' % batch_image_number)
                batch_labels[batch_image_number] = bbox_label
            batch_images[batch_image_number] = image

        self.current_batch_images = batch_images
        self.current_batch_labels = batch_labels

    def calculate_next_batch_simple_does_it_box(self):
        """
        Use the labels saved from earlier predictions as groundtruth.
        But do a check for outside the bbox labels.
        """
        self.calculate_next_batch_paths()
        number_of_current_paths = np.shape(self.current_batch_paths)[0]
        batch_images = np.empty((number_of_current_paths, 150, 200, 3))
        batch_labels = np.empty((number_of_current_paths, 150, 200, 1))

        for batch_image_number in range(number_of_current_paths):
            
            #load original labels and image
            path = self.current_batch_paths[batch_image_number]
            image = self.get_image(path)
            bbox_label = self.get_label(path)
            bbox_label = bbox_label.reshape((150, 200, 1))

            # load images from temp
            try:
                set_path, image_name, _ = self.get_image_info(path)
                label = np.load('%s/%s/%s.npy' %(self.temppath, set_path, image_name))

                # Tresholding
                mask = label >= 0.7
                label[mask] = 1.0
                mask = label < 0.7
                label[mask] = 0.0

                # reset everything outside of bbox to 0
                mask = bbox_label > 0.5
                label[~mask] = 0.0

                batch_labels[batch_image_number] = label
                print('loaded %s .label from temp and manipulated via box' % batch_image_number)

            # load images from original data  
            except:
                print('loaded %s .label from original data' % batch_image_number)
                batch_labels[batch_image_number] = bbox_label
            batch_images[batch_image_number] = image

        self.current_batch_images = batch_images
        self.current_batch_labels = batch_labels

    def calculate_next_batch_simple_does_it_box_iou(self):
        """
        Use the labels saved from earlier predictions as groundtruth.
        But do a check for outside the bbox labels and check with iou
        """
        self.calculate_next_batch_paths()
        number_of_current_paths = np.shape(self.current_batch_paths)[0]
        batch_images = np.empty((number_of_current_paths, 150, 200, 3))
        batch_labels = np.empty((number_of_current_paths, 150, 200, 1))

        for batch_image_number in range(number_of_current_paths):
            
            #load original labels and image
            path = self.current_batch_paths[batch_image_number]
            image = self.get_image(path)
            bbox_label = self.get_label(path)
            bbox_label = bbox_label.reshape((150, 200, 1))

            # load images from temp
            try:
                set_path, image_name, _ = self.get_image_info(path)
                label = np.load('%s/%s/%s.npy' %(self.temppath, set_path, image_name))

                # Tresholding
                mask = label >= 0.7
                label[mask] = 1.0
                mask = label < 0.7
                label[mask] = 0.0

                # reset everything outside of bbox to 0
                mask = bbox_label > 0.5
                label[~mask] = 0.0

                label = self.Intersection_Over_Union_for_labels(bbox_label, label)
                batch_labels[batch_image_number] = label
                print('loaded %s .label from temp and manipulated via box + iou' % batch_image_number)

            # load images from original data  
            except:
                print('loaded %s .label from original data' % batch_image_number)
                batch_labels[batch_image_number] = bbox_label
            batch_images[batch_image_number] = image

        self.current_batch_images = batch_images
        self.current_batch_labels = batch_labels
    
    def calculate_next_batch_simple_does_it_grabcut(self):
        """
        Calculate current images and grabcutted labes of those images and save
        them in self.current_batch_images/labels
        """
        self.calculate_next_batch_paths()
        number_of_current_paths = np.shape(self.current_batch_paths)[0]
        batch_images = np.empty((number_of_current_paths, 150, 200, 3))
        batch_labels = np.empty((number_of_current_paths, 150, 200, 1))

        for batch_image_number in range(number_of_current_paths):
            path = self.current_batch_paths[batch_image_number]
            image = cv2.imread(path)[:, :, ::-1]
            set_name, image_name, _text_file_path = self.get_image_info(path)
            _truth = np.zeros_like(image, dtype=np.float32)
            try:
                #load labels from temp directory if it exists
                #this time those will be grabcutted images!
                label = np.load('%s/%s/%s.npy' %(self.temppath, set_name, image_name))
                print('loaded grabcutted label from temp')
            except:
                if not(os.path.exists('%s/%s' % (self.temppath, set_name))):
                    os.mkdir(('%s/%s' % (self.temppath, set_name)))
                _text_file = open(_text_file_path)
                _skip_label_info_lines_counter = 0
                for _line in _text_file:
                    _skip_label_info_lines_counter = _skip_label_info_lines_counter + 1
                    if _skip_label_info_lines_counter > 6: # skip first lines
                        
                        _line = _line.strip()
                        _split_line = _line.split('|')
                        if not(_split_line == ['']): #jump over blank lines in export
                            if (_split_line[1] == image_name) and not(_split_line[2] == 'not_in_image'):
                                if ((self.label == 'ball') and (_split_line[0] == 'label::ball') 
                                or ((self.label == 'goalpost') and (_split_line[0] == 'label::goalpost')
                                or ((self.label == 'robot') and (_split_line[0] == 'label::robot')))):    
                                    _x1y1 = (int(_split_line[4]),int(_split_line[5]))
                                    w = int(_split_line[10])
                                    h = int(_split_line[11])

                                    #Grabcut
                                    mask = np.zeros(image.shape[:2],np.uint8)
                                    bgdModel = np.zeros((1,65),np.float64)
                                    fgdModel = np.zeros((1,65),np.float64)
                                    rect = (_x1y1[0],_x1y1[1],w,h) # x y w h
                                    cv2.grabCut(image,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
                                    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                                    grabcutted_image = image*mask2[:,:,np.newaxis]
                                    mask = (grabcutted_image[:,:,0] > 0.0) & (grabcutted_image[:,:,1] > 0.0) & (grabcutted_image[:,:,2] > 0.0)

                                    _truth[mask] = 1.0
                                    _text_file.close

            _truth_resized = cv2.resize(_truth, (200, 150)) #shape (150,200)
            mask = _truth_resized > 0.0
            _truth_resized[mask] = 1.0
            label = _truth_resized #(150,200)
            label = label.reshape((150, 200, 1)) # or else error! da netz so arbeitet
            np.save('%s/%s/%s.npy' %(self.temppath, set_name, image_name), label)
            print('created new grabcuts and saved them')                        

            batch_images[batch_image_number] = self.get_image(path)
            batch_labels[batch_image_number] = label
        self.current_batch_images = batch_images
        self.current_batch_labels = batch_labels

    def save_labels_for_batch_simple_does_it(self, predictions):
        """
        Saves the predictions of the batch to disk.
        """
        for path, prediction in zip(self.current_batch_paths, predictions):
            set_name, image_name, _ = self.get_image_info(path)

            #create directories that dont exist in temp
            if not(os.path.exists('%s/%s' % (self.temppath, set_name))):
                os.mkdir(('%s/%s' % (self.temppath, set_name)))

            #save array as numpy file
            np.save('%s/%s/%s.npy' %(self.temppath, set_name, image_name), prediction)
        




if __name__ == "__main__":
    PATH = '../data/goalpost/train/'
    LABEL = 'goalpost_poly'
    BATCHSIZE = 16
    np.set_printoptions(threshold=np.inf)
    print(os.getcwd())
    data = DataLoader(PATH, LABEL, BATCHSIZE)
    #data.save_specific_imaage_and_label('../data/goalpost_test/train/bitbots-2018-iran-01/frame0027.jpg')
    logging.basicConfig(filename='testlog.log', format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('AUF DER SUCHE NACH FEHLERN')

    data.reset_image_counter
    for batch_num in range(data.get_number_batches()):
        print(batch_num, " Batch of ", data.get_number_batches())
        data.calculate_next_batch()
        data.get_current_batch_paths()
        images, label = data.get_current_batch_image_label()

    data.save_image_and_labels(20)
    print('all done')

    