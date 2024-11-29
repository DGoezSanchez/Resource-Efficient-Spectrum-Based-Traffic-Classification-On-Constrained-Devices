#!/usr/bin/env python3
# MTL_FP16_


import numpy as np
import trt_utils2
import pandas as pd
import pickle
from time import time

FP = 'FP16'
NB = 128

#Torch_model_VGG10MTL_8-11-2023_task1_FP16_batch64

#PLAN_FILE_NAME = '/home/air-t/inference/pytorch/Torch_model_VGG10MTL_8-11-2023_task1_'+str(FP)+'_batch'+str(NB)+'.plan'
PLAN_FILE_NAME = '/home/air-t/inference/pytorch/'+str(NB)+'Torch_model_VGG10MTL_08-11_dloss-2023'+str(FP)+'_batch'+str(NB)+'.plan'

BATCH_SIZE = NB
CPLX_SAMPLES_PER_INFER = 3000

with open('/media/air-t/SD-128/multi_task_learning/traffic_classification_test_X_b.pickle', 'rb') as archivo:
    xtest = pickle.load(archivo)


def main():

	trt_utils2.make_cuda_context()
	buff_len = 2 * CPLX_SAMPLES_PER_INFER * BATCH_SIZE
	sample_buffer = trt_utils2.MappedBuffer(buff_len , np.float32)
	dnn = trt_utils2.TrtInferFromPlan(PLAN_FILE_NAME, BATCH_SIZE, sample_buffer)
	dt = []
	for i in xtest:
		for j in i:
			dt.append(j)

	predictTask1 = []
	predictTask2 = []
	predictTask3 = []
	predictTask4 = []

	#for ctr in range(NUM_BATCHES):
	start_time = time()
	for d in range(0,len(dt),NB):
		rp = np.reshape(dt[d:d+NB],(-1))
		np.copyto(sample_buffer.host, rp)
		dnn.feed_forward()
		[output_task1, output_task2, output_task3, output_task4] = [dnn.output_buff_1.host, dnn.output_buff_2.host, dnn.output_buff_3.host, dnn.output_buff_4.host]	
		#predictTask1.append(np.argmax(output_task1.reshape(NB,3),axis=1))
		#predictTask2.append(np.argmax(output_task2.reshape(NB,3),axis=1))
		#predictTask3.append(np.argmax(output_task3.reshape(NB,3),axis=1))
		#predictTask4.append(np.argmax(output_task4.reshape(NB,7),axis=1))
		
	elapsed_time = time() - start_time
	print("Elapsed time (seconds): %.10f :" % elapsed_time)
	print("Runtime one sample per seconds (seconds): %.10f :" % (elapsed_time/8192))
	print(" ")
	#df_task1 = pd.DataFrame(predictTask1)
	#df_task2 = pd.DataFrame(predictTask2)
	#df_task3 = pd.DataFrame(predictTask3)
	#df_task4 = pd.DataFrame(predictTask4)

	#df_task1.to_csv('/media/air-t/SD-128/multi_task_learning/results/MTLpredict_task1_'+str(FP)+'_batch'+str(NB)+'.csv', index=False, header=False)
	#df_task2.to_csv('/media/air-t/SD-128/multi_task_learning/results/MTLpredict_task2_'+str(FP)+'_batch'+str(NB)+'.csv', index=False, header=False)
	#df_task3.to_csv('/media/air-t/SD-128/multi_task_learning/results/MTLpredict_task3_'+str(FP)+'_batch'+str(NB)+'.csv', index=False, header=False)
	#df_task4.to_csv('/media/air-t/SD-128/multi_task_learning/results/MTLpredict_task4_'+str(FP)+'_batch'+str(NB)+'.csv', index=False, header=False)


if __name__ == '__main__':
	print("-----------------------------------------------------------------")
	print('MTL_Torch_model_VGG10MTL_8-11-2023_'+str(FP)+'_batch'+str(NB)+'.plan')
	print("-----------------------------------------------------------------")
	main()
	print('To end!')



