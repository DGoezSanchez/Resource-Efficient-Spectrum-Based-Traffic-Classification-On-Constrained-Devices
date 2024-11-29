#!/usr/bin/env python3

import numpy as np
import trt_utils
import pandas as pd
import pickle
from time import time, sleep

FP = 'FP32'
NB = 64
task = 1
# Torch_model_VGG10MTL_8-11-2023_task1_FP32

PLAN_FILE_NAME = '/home/air-t/inference/pytorch/Torch_model_VGG10MTL_8-11-2023_task'+str(task)+'_'+FP+'.plan'
#PLAN_FILE_NAME = '/home/air-t/inference/pytorch/'+str(NB)+'Torch_model_VGG10MTL_08-11_dloss-2023'+str(FP)+'_batch'+str(NB)+'.plan'

BATCH_SIZE = NB
CPLX_SAMPLES_PER_INFER = 3000

with open('/media/air-t/SD-128/multi_task_learning/traffic_classification_test_X_b.pickle', 'rb') as archivo:
    xtest = pickle.load(archivo)


def main():

	trt_utils.make_cuda_context()
	buff_len = 2 * CPLX_SAMPLES_PER_INFER * BATCH_SIZE
	sample_buffer = trt_utils.MappedBuffer(buff_len , np.float32)
	dnn = trt_utils.TrtInferFromPlan(PLAN_FILE_NAME, BATCH_SIZE, sample_buffer)
	dt = []
	for i in xtest:
		for j in i:
			dt.append(j)

	predictTask = []

	start_time = time()
	for d in range(0,len(dt),NB):
		rp = np.reshape(dt[d:d+NB],(-1))
		np.copyto(sample_buffer.host, rp)
		dnn.feed_forward()
		output_task = dnn.output_buff.host
		predictTask.append(np.argmax(output_task.reshape(64,3),axis=1))

	elapsed_time = time() - start_time
	#print(" ")

	print("Elapsed time (seconds): %.10f :" % elapsed_time)
	print("Runtime one sample per seconds (seconds): %.10f :" % (elapsed_time/8192))

	#df_task = pd.DataFrame(predictTask)


	#df_task.to_csv('/media/air-t/SD-128/multi_task_learning/results/single_predict_task'+str(task)+'_'+str(FP)+'.csv', index=False, header=False)

	



if __name__ == '__main__':
	print("-----------------------------------------------------------------")
	print('Torch_model_VGG10MTL_8-11-2023_task'+str(task)+'_'+FP+'.plan')
	print("-----------------------------------------------------------------")
	#sleep(0.00001)
	main()
	print('To end!')
	



