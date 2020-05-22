from __future__ import print_function
import runway

import os, glob, time, argparse, pdb, cv2
import numpy as np
from skimage.measure import label

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from functions import *
from networks import ResnetConditionHR

torch.set_num_threads(1)
#print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])

@runway.setup(options={'checkpoint': runway.file(extension='.pth')})
def setup(opts):
    print('begining of setup')
    #initialize network
    netM=ResnetConditionHR(input_nc=(3,3,1,4),output_nc=4,n_blocks1=7,n_blocks2=3)
    netM=nn.DataParallel(netM)
    checkpoint_path = opts['checkpoint']
    netM.load_state_dict(torch.load(checkpoint_path))
    netM.cuda(); netM.eval()
    cudnn.benchmark=True
    print('end of setup')
    return netM

inputs = {
    'input_subject': runway.image(description='An input image with the subject.'),
    'input_background': runway.image(description='The background of the input image without the subject.'),
    'input_segmentation': runway.image(description='Segmentation image of the input image'),
    'target_background': runway.image(description='Target background image'),
}

@runway.command('generate', inputs=inputs, outputs={'output': runway.image})
def generate(model, inputs):
	print('begining of generate')
	netM = model
	reso=(512,512) #input reoslution to the network
	# original input image
	input_subject = inputs['input_subject']
	bgr_img = np.array(input_subject)
	bgr_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)

	# captured background image
	input_background = inputs['input_background']
	bg_im0=np.array(input_background)
	bg_im0=cv2.cvtColor(bg_im0,cv2.COLOR_BGR2RGB)

	# segmentation mask
	input_segmentation = inputs['input_segmentation']
	rcnn = np.array(input_segmentation)

	#target background path
	target_background = inputs['target_background']
	back_img10=np.array(target_background); back_img10=cv2.cvtColor(back_img10,cv2.COLOR_BGR2RGB);
	#Green-screen background
	back_img20=np.zeros(back_img10.shape); back_img20[...,0]=120; back_img20[...,1]=255; back_img20[...,2]=155;

	## create the multi-frame
	multi_fr_w=np.zeros((bgr_img.shape[0],bgr_img.shape[1],4))
	multi_fr_w[...,0] = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY)
	multi_fr_w[...,1] = multi_fr_w[...,0]
	multi_fr_w[...,2] = multi_fr_w[...,0]
	multi_fr_w[...,3] = multi_fr_w[...,0]

	#crop tightly
	print('-----------------')
	print(bgr_img.shape)
	print(bgr_img)
	print('-----------------')
	print(rcnn.shape)
	print(rcnn)
	bgr_img0=bgr_img
	bbox=get_bbox(rcnn,R=bgr_img0.shape[0],C=bgr_img0.shape[1])

	crop_list=[bgr_img,bg_im0,rcnn,back_img10,back_img20,multi_fr_w]
	crop_list=crop_images(crop_list,reso,bbox)
	bgr_img=crop_list[0]; bg_im=crop_list[1]
	rcnn=crop_list[2]; back_img1=crop_list[3]; back_img2=crop_list[4]; multi_fr=crop_list[5]

	print('process segmentation mask')
	#process segmentation mask
	kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	rcnn=rcnn.astype(np.float32)/255; rcnn[rcnn>0.2]=1
	K=25

	zero_id=np.nonzero(np.sum(rcnn,axis=1)==0)
	del_id=zero_id[0][zero_id[0]>250]
	if len(del_id)>0:
		del_id=[del_id[0]-2,del_id[0]-1,*del_id]
		rcnn=np.delete(rcnn,del_id,0)
	rcnn = cv2.copyMakeBorder(rcnn, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)


	rcnn = cv2.erode(rcnn, kernel_er, iterations=10)
	rcnn = cv2.dilate(rcnn, kernel_dil, iterations=5)
	rcnn=cv2.GaussianBlur(rcnn.astype(np.float32),(31,31),0)
	rcnn=(255*rcnn).astype(np.uint8)
	rcnn=np.delete(rcnn, range(reso[0],reso[0]+K), 0)

	print('convert to torch')
	#convert to torch
	img=torch.from_numpy(bgr_img.transpose((2, 0, 1))).unsqueeze(0); img=2*img.float().div(255)-1
	bg=torch.from_numpy(bg_im.transpose((2, 0, 1))).unsqueeze(0); bg=2*bg.float().div(255)-1
	rcnn_al=torch.from_numpy(rcnn).unsqueeze(0).unsqueeze(0); rcnn_al=2*rcnn_al.float().div(255)-1
	multi_fr=torch.from_numpy(multi_fr.transpose((2, 0, 1))).unsqueeze(0); multi_fr=2*multi_fr.float().div(255)-1


	with torch.no_grad():
		img,bg,rcnn_al, multi_fr =Variable(img.cuda()),  Variable(bg.cuda()), Variable(rcnn_al.cuda()), Variable(multi_fr.cuda())
		input_im=torch.cat([img,bg,rcnn_al,multi_fr],dim=1)
		
		alpha_pred,fg_pred_tmp=netM(img,bg,rcnn_al,multi_fr)
		
		al_mask=(alpha_pred>0.95).type(torch.cuda.FloatTensor)

		# for regions with alpha>0.95, simply use the image as fg
		fg_pred=img*al_mask + fg_pred_tmp*(1-al_mask)

		alpha_out=to_image(alpha_pred[0,...]); 

		#refine alpha with connected component
		labels=label((alpha_out>0.05).astype(int))
		try:
			assert( labels.max() != 0 )
		except:
			pass
		largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
		alpha_out=alpha_out*largestCC

		alpha_out=(255*alpha_out[...,0]).astype(np.uint8)				

		fg_out=to_image(fg_pred[0,...]); fg_out=fg_out*np.expand_dims((alpha_out.astype(float)/255>0.01).astype(float),axis=2); fg_out=(255*fg_out).astype(np.uint8)

		#Uncrop
		R0=bgr_img0.shape[0];C0=bgr_img0.shape[1]
		alpha_out0=uncrop(alpha_out,bbox,R0,C0)
		fg_out0=uncrop(fg_out,bbox,R0,C0)

	print('compose')
	#compose
	back_img10=cv2.resize(back_img10,(C0,R0)); back_img20=cv2.resize(back_img20,(C0,R0))
	comp_im_tr1=composite4(fg_out0,back_img10,alpha_out0)
	comp_im_tr2=composite4(fg_out0,back_img20,alpha_out0)

	out = cv2.cvtColor(comp_im_tr1,cv2.COLOR_BGR2RGB)
	print('end of generate')
	return out

if __name__ == '__main__':
	runway.run(port=8888, model_options={'checkpoint': './real-fixed-cam.pth'})