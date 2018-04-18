import sift_controller
import time
import cv2
import glob
from utils import *
from multiprocessing.pool import ThreadPool
import search

def Linear_search(query_path):
	sift = sift_controller.SIFT()
	sift.search(query_path)
	del sift

def Linear_search_by_FLANN(query_path):
	sift = sift_controller.SIFT()
	sift.fast_search(query_path)
	del sift
	
def Linear_search_prefetching(query_path):
	sift = sift_controller.SIFT()
	sift.inmemory_search(query_path)
	del sift

def Parallel_search_prefetching(query_path, num_threads):
	input_list = prefetching(query_path)
	pool = ThreadPool(processes=num_threads)
	pool.map(search.multiprocessing_search, input_list)

if __name__ == "__main__":
	
	img_path = "./thumb/741_RPI1477299693.jpg"
	# Linear search
	Linear_search(img_path)
	
	# Linear search by FLANN
	Linear_search_by_FLANN(img_path)
	
	# Linear search + inmemory prefetching
	Linear_search_prefetching(img_path)
	
	# Parallel search + inmemory prefetching
	num_threads = cv2.getNumberOfCPUs()
	Parallel_search_prefetching(img_path, num_threads)
	