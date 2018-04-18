import pickle as pkl
import cv2
import sift_controller

def parse_pkl(dic):
	return dic["id"], dic["des"]

def pickleloader(pklfile):
	try:
		while True:
			yield pkl.load(pklfile)
	except EOFError:
		pass

def parse_glob(path):
	return path.split("/")[2]
	
def get_top_k_result(match_list=None, k=10):
    result = (sorted(match_list, key=lambda l: l[1], reverse=True))
    return result[:k]

def prefetching(query_path):
	pkl_file = open("siftdump.pkl", "rb")
	sift = sift_controller.SIFT()
	query_img = cv2.imread(query_path, 0)
	query_des = sift.extract(query_img)
	input_list = []
	for idx, contents in enumerate(pickleloader(pkl_file)):
		id, indexed_des = parse_pkl(contents)
		if (indexed_des.all()) == None:
			continue
		input_list.append([query_des, id, indexed_des])
	del sift
	
	pkl_file.close()
	
	return input_list