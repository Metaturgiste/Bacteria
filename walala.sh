for ((i=0; 4-$i; i++)); do for ((j=0; 3-$j; j++)); do for ((k=0; 9-$k; k++)); do 
screen -d -m -S w$i$j$k bash -c "python3 main.py $i $j $k" #& 
done; done; done;

