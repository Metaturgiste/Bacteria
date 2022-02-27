for ((i=-2; 2-$i; i+=0.25)); do for ((j=-2; 2-$j; j+=0.25)); do 
screen -d -m -S w$i$j bash -c "python3 main.py $i $j" #& 
done; done;