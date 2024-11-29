#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
import pickle
from time import time
import pandas as pd

def get_instant_current():
    device = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_current0_input"
    total_current = 0
    with open(device, "r") as f:
        total_current += float(f.readline())
    return total_current

def main():
    with open('X_test_l_8192.pkl','rb') as f:
        X_test_l = pickle.load(f)
        print(X_test_l.shape)

    # Load the TFLite model in TFLite Interpreter
    tflite_model_name='./tflite_qnt_240303_140224_multi_task_classifier.h5.tflite'
    #interpreter = tflite.Interpreter(tflite_model_name, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1.0')])
    interpreter = tflite.Interpreter(model_path=tflite_model_name)    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    start_time = time()
    energy = []
    
    for i in range(0,8192):
        input_data = X_test_l[i]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if i%64 == 0:
            energy.append([get_instant_current(), time()-start_time])
            #print(i)
    df_energy = pd.DataFrame(energy)
    df_energy.to_csv('./Raspberry_energy_tflite_cpu.csv')

if __name__ == '__main__':
    init_time = time()
    main()
    to_end_time = time()
    print('To end!', to_end_time-init_time)
