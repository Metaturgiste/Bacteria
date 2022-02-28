for ((i=0; 9-$i; i++)); do for ((j=0; 9-$j; j++)); do 
screen -d -m -S w$i$j bash -c "python3 ex.py $i $j" #& 
done; done;
