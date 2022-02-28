for i in 2 -1.5 -1.25 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.25 1.5 2; do for i in 2 -1.5 -1.25 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.25 1.5 2; do 
screen -d -m -S w$i$j bash -c "python3 main.py $i $j" #& 
done; done;