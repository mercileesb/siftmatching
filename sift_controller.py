import cv2
import pickle
import os
import glob
from multiprocessing import Pool
import time
from utils import *


class SIFT():
	def __init__(self):
		self.sift = cv2.xfeatures2d.SIFT_create()
		self.threshold = 0.75
		self.indexedfolder = './sift'
		self.thumbfolder = './thumb'
	
	def dump_eachfile(self, img_name):
		img_path = os.path.join(self.thumbfolder, img_name)
		input_img = cv2.imread(img_path, 0)
		kp, des = self.sift.detectAndCompute(input_img, None)
		img_id = img_name.split('.')[0]
		binfile = img_id + '.pkl'
		path = os.path.join(self.indexedfolder, binfile) 
		with open(path, 'wb') as dumpfile:
			pickle.dump(des, dumpfile)
	
	def dump_onefile(self):
		dumpfile = open("siftdump.pkl","wb")
		for img_path in glob.glob("./thumb/*.jpg"):
			img_name = img_path.split("/")[2]
			input_img = cv2.imread(img_path, 0)
			kp, des = self.sift.detectAndCompute(input_img, None)
			img_id = img_name.split('.')[0]
			contents = {"id" : img_id, "des" : des}
			pickle.dump(contents, dumpfile)
		dumpfile.close()

	def read(self, featurepath):
		with open(featurepath, "rb") as dump:
			des = pickle.load(dump)
		return des
	
	def extract(self, img):
		_, des = self.sift.detectAndCompute(img, None)
		return des
	
	def search(self, query_path):
		query_img = cv2.imread(query_path, 0)
		query_des = self.extract(query_img)
		match_list = []
		indexed_list = os.listdir(self.indexedfolder)
		for idx, feature_file in enumerate(indexed_list):
			feature_path = os.path.join(self.indexedfolder, feature_file)
			features = self.read(feature_path)
			if (features.all()) == None:
				continue
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(query_des, features, k=2)
			similar_list = []
			for m,n in matches:
				if m.distance < self.threshold * n.distance:
					similar_list.append([m])
			match_list.append([feature_file, len(similar_list)])
			del features, similar_list
		result = get_top_k_result(match_list=match_list, k=5)
		return result
	
	def measure(self, query_des, indexed_list):
		bf = cv2.BFMatcher()
		id = indexed_list[0]
		indexed_des = indexed_list[1]
		matches = bf.knnMatch(query_des, indexed_des, k=2)
		similar_list = []
		for m, n in matches:
			if m.distance < self.threshold * n.distance:
				similar_list.append([m])
		ret = [id, len(similar_list)]
		del indexed_des, similar_list
		return ret
		
	def inmemory_search(self, query_path):
		query_img = cv2.imread(query_path, 0)
		query_des = self.extract(query_img)
		pkl_file = open("siftdump.pkl", "rb")
		indexed_list = []
		for idx, contents in enumerate(pickleloader(pkl_file)):
			id, indexed_des = parse_pkl(contents)
			if (indexed_des.all()) == None:
				continue
			indexed_list.append([id, indexed_des])
		pkl_file.close()
		start_time = time.time()
		match_list = list(map(lambda i: self.measure(query_des, i), indexed_list))
		ret_time = time.time() - start_time
		result = get_top_k_result(match_list=match_list, k=5)
		return result, ret_time
	
	def fast_search(self, query_path):
		query_img = cv2.imread(query_path, 0)
		query_des = self.extract(query_img)
		match_list = []
		indexed_list = os.listdir(self.indexedfolder)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=10)
		for idx, feature_file in enumerate(indexed_list):
			feature_path = os.path.join(self.indexedfolder, feature_file)
			features = self.read(feature_path)
			if (features.all()) == None:
				continue
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			matches = flann.knnMatch(query_des, features, k=2)
			similar_list = []
			for m,n in matches:
				if m.distance < self.threshold * n.distance:
					similar_list.append([m])
			match_list.append([feature_file, len(similar_list)])
			del features, similar_list
		result = get_top_k_result(match_list=match_list, k=5)
		return result

