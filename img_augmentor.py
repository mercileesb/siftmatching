import Augmentor
from PIL import Image
from PIL import ImageDraw
import os
import glob
from utils import *

def add_logo(ori_img_path, logo_img_path):
	ori_img = Image.open(ori_img_path)
	logo_img = Image.open(logo_img_path)
	logo_img2 = logo_img.resize((40,40))
	ori_img.paste(logo_img2, (120, 10), logo_img2)
	path_tmp = parse_glob(ori_img_path)
	output_path = "./thumb/output/thumb_logo_" + path_tmp
	ori_img.save(output_path)

def add_caption(ori_img_path):
	ori_img = Image.open(ori_img_path)
	draw = ImageDraw.Draw(ori_img)
	draw.text((10,10), "Sogang University", fill=(0,0,0))
	draw.text((40, 140), "Multimedia System lab",fill=(0,0,0))
	path_tmp = parse_glob(ori_img_path)
	output_path = "./thumb/output/thumb_caption_" + path_tmp
	ori_img.save(output_path)
	
def add_border(ori_img_path):
	border_image = Image.open("resize_logo_300.jpg") 
	ori_img = Image.open(ori_img_path)
	border_image.paste(ori_img.resize((220, 220)), (40, 40))
	path_tmp = parse_glob(ori_img_path)
	output_path = "./thumb/output/thumb_border_" + path_tmp
	border_image.save(output_path)

if __name__ == "__main__":
	_THUMB_FOLDER = "./thumb"
	p = Augmentor.Pipeline(_THUMB_FOLDER)
	p.flip_left_right(probability=0.5)
	p.random_erasing(probability=0.5, rectangle_area=0.2)
	p.sample(1000)
	original_image_list = glob.glob("./thumb/*.jpg")
	for idx, img_path in enumerate(original_image_list):
		
		
		#add_caption(img_path)
		#add_logo(img_path, "minions_PNG84.png")
		add_border(img_path)
	print ("{} / {}".format(idx+1, len(original_image_list)))
