import numpy as np
import logging, os

normalize_threshold = 0.5

#########################

def create_heatmap_from_bbox(heatmap, x1, x2, y1, y2):
	"""
	returns a 2D-vector, in which all values are set to 1,
	specified by the given coordinates
	"""
	for x in range(x1, x2+1):
		for y in range(y1, y2+1):
			heatmap[x][y] = 1
	return heatmap

def normalize_predicted_Labels(array):
	"""
	returns a vector, in which every value is set to
	1 or 0 according to the threshold
	"""
	normalized_array = np.copy(array)
	normalized_array[normalized_array >= normalize_threshold] = 1.0
	normalized_array[normalized_array < normalize_threshold] = 0.0
	return normalized_array
    

def Intersection_Over_Union(truth, prediction, img_dim=(250,150)):
	"""
	calculates the Jaccard-Index/IoU between
	two normalized vectors
	"""
	area_of_overlap = 0.0
	area_of_union = 0.0
	for x in range(0, truth.shape[0]):
		for y in range(0, truth.shape[1]):
			if truth[x][y] == [1.0] and prediction[x][y] == [1.0]:
				area_of_overlap = area_of_overlap + 1

			if truth[x][y] == [1.0] or prediction[x][y] == [1.0]:
				area_of_union = area_of_union + 1
				
	intersection_over_union = area_of_overlap/area_of_union
	return intersection_over_union


def Precision(truth, prediction, img_dim=(250,150)):
	"""
	calculates the precision value between two vectors
	-measures how accurate is your predictions. 
	-i.e. the percentage of your positive predictions are correct.
	"""
	true_positives = 0.0
	false_positives= 0.0
	for x in range(0, truth.shape[0]):
		for y in range(0,truth.shape[1]):
			if truth[x][y] == 1 and prediction[x][y] == 1 :
					true_positives =  true_positives + 1
			if truth[x][y] == 0 and prediction[x][y] == 1:
				false_positives = false_positives + 1
	precision = true_positives/(true_positives + false_positives)
	return precision

def Recall(truth, prediction, img_dim=(250,150)):
	"""
	calculates the recall value between two vectors
	-measures how good all positives were found
	"""
	true_positives = 0.0
	false_negatives= 0.0
	for x in range(0,truth.shape[0]):
		for y in range(0,truth.shape[1]):
			if truth[x][y] == 1 and prediction[x][y] == 1:
				true_positives =  true_positives + 1
			if truth[x][y] == 1 and prediction[x][y] == 0:
				false_negatives = false_negatives + 1
	recall = true_positives/(true_positives + false_negatives)
	return recall

def Accuracy(truth, prediction, img_dim=(250,150)):
	"""
	calculates the accuracy value between two vectors
	-measures the difference between the result and the truth
	"""
	true_positives = 0.0
	true_negatives = 0.0
	false_positives= 0.0
	false_negatives= 0.0
	for x in range(0,truth.shape[0]):
		for y in range(0,truth.shape[1]):
			if truth[x][y] == 1 and prediction[x][y] == 1:
				true_positives =  true_positives + 1
			if truth[x][y] == 0 and prediction[x][y] == 0:
				true_negatives = true_negatives + 1
			if truth[x][y] == 0 and prediction[x][y] == 1:
				false_positives = false_positives + 1
			if truth[x][y] == 1 and prediction[x][y] == 0:
				false_negatives = false_negatives + 1
	accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
	return accuracy


def calculate_metrics(truth, prediction, img_dim=(150,200)):
	#https://towardsdatascience.com/accuracy-recall-precision-f-score-specificity-which-to-optimize-on-867d3f11124
	"""
	calculates all the metrics(accuracy, precision, recall, IoU) per batch using the 
	formulas in the paper/from above but only calculating the positives and
	negatives once to save in complexity
	"""
	true_positives = 0.0
	true_negatives = 0.0
	false_positives= 0.0
	false_negatives= 0.0
	area_of_union  = 0.0
	for x in range(0,truth.shape[0]):
		for y in range(0,truth.shape[1]):
			#if prediction[x][y] != [0.0] and truth[x][y] == [1.0]:
			#	print(truth[x][y], prediction[x][y])
			if truth[x][y] == [1.0] and prediction[x][y] == [1.0]:
				true_positives = true_positives + 1
			if truth[x][y] == [0.0] and prediction[x][y] == [0.0]:
				true_negatives = true_negatives + 1
			if truth[x][y] == [0.0] and prediction[x][y] == [1.0]:
				false_positives = false_positives + 1
			if truth[x][y] == [1.0] and prediction[x][y] == [0.0]:
				false_negatives = false_negatives + 1
			if truth[x][y] == [1.0] or prediction[x][y] == [1.0]:
				area_of_union = area_of_union + 1
	try:
		accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
	except ZeroDivisionError:
		accuracy = 0.0
	try:
		precision = true_positives/(true_positives + false_positives)
	except ZeroDivisionError:
		precision = 0.0
	try:        
		recall = true_positives/(true_positives + false_negatives)
	except ZeroDivisionError:
		recall = 0.0
	try:        
		intersection_over_union = true_positives/area_of_union
	except ZeroDivisionError:
		intersection_over_union = 0.0
	print(accuracy, precision, recall, intersection_over_union, "LOOK AT THOSE VALUES")
	return (accuracy, precision, recall, intersection_over_union)

def accumulate_metrics(accuracy, precision, recall, IoU):
	"""
	calculates for accumulated metrics 
	the mean and percentiles in a list
	"""
	Accuracy = [np.mean(accuracy), np.percentile(accuracy, 90), np.percentile(accuracy, 99)]
	Precision = [np.mean(precision), np.percentile(precision, 90), np.percentile(precision, 99)]
	Recall = [np.mean(recall), np.percentile(recall, 90), np.percentile(recall, 99)]
	IoU = [np.mean(IoU), np.percentile(IoU, 90), np.percentile(IoU, 99)]
	return Accuracy,Precision,Recall,IoU

if __name__ == "__main__":
	"""
	running this file as main starts a test
	to ensure correct working functions
	"""

	PATH = '../data/test'
	LABEL = 'ball'
	BATCHSIZE = 16
	np.set_printoptions(threshold=np.inf)
	print(os.getcwd())
	data = DataLoader(PATH, LABEL, BATCHSIZE)

	data.calculate_next_batch()
	data.get_current_batch_paths()
	images, label = data.get_current_batch_image_label()
	for l in label:
		a,b,c,d = calculate_metrics(l,l)
		print(a,b,c,d)
	#normalizeLabels
	a = np.random.rand(2,2)
	a2 = normalize_predicted_Labels(a)
	print('Random Array to normalize:\n', a, '\nNormalized Array:\n', a2)

	truth = [[1,1,1,0,0,0],
			[1,1,1,0,0,0],
			[1,1,1,0,0,0],
			[0,0,0,0,0,0],
			[0,0,0,0,0,0],
			[0,0,0,0,0,0]]

	prediction =[[0,1,1,1,0,0],
				[0,1,1,1,0,0],
				[0,1,1,1,0,0],
				[0,0,0,0,0,0],
				[0,0,0,0,0,0],
				[0,0,0,0,0,0]]

	nptruth = np.array(truth)
	npprediction = np.array(prediction)

	metrics = calculate_metrics(nptruth,npprediction,(6,6))
	
	print('Accuracy:', metrics[0], '\n',
			'Precision:', metrics[1], '\n',
			'Recall:', metrics[2], '\n',
			'IoU:', metrics[3], '\n')

	a,b,c,d = accumulate_metrics([1,2,3],[1,2,3],[1,2,3],[1,2,3])
	print('Accumulate Metrics:,\n', a,'\n',b,'\n',c,'\n',d,'\n')
	
	a = np.zeros((5,5))
	print('Array vor heatmapping: \n', a)
	a = create_heatmap_from_bbox(a, 0, 2, 0, 3)
	print('Array nach heatmapping mit x:0-2 und y:0-3 \n', a)
