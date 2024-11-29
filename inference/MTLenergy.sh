#!/bin/bash

file_path="/media/air-t/SD-128/multi_task_learning/energy/"
echo "Introduce el nombre del archivo:"
read file_name

full_path="${file_path}${file_name}"

echo "Guardando mediciones en $full_path"
sudo chmod a+r /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_current0_input

tiempo_inicial=$(date +%s%N)
tiempo_inicial=${tiempo_inicial:0:${#tiempo_inicial}-6}

#python running_inference_task.py &

python running_inference.py &
pid_python=$!


while kill -0 $pid_python 2> /dev/null; do
 
    valor=$(cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_current0_input)
    

    tiempo_actual=$(date +%s%N)
    tiempo_actual=${tiempo_actual:0:${#tiempo_actual}-6}
    

    tiempo_transcurrido_ms=$((tiempo_actual - tiempo_inicial))
    

    tiempo_transcurrido_s=$(echo "scale=3; $tiempo_transcurrido_ms / 1000" | bc)
    

    echo "$tiempo_transcurrido_s, $valor" >> "$full_path"
    
    #sleep 1  # Añadir un breve retraso para evitar una sobrecarga
done

echo "Script de Python to end. Measurements saved in  $full_path."


