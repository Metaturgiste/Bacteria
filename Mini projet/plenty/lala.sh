for ((i=0; 40-$i; i++)); do for ((j=0; 40-$j; j++)); do 
screen -d -m -S w$i$j bash -c "python3 ex.py $i $j" #& 
done; done;
