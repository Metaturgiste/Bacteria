for ((i=0; 4-$i; i++)); do for ((j=0; 3-$j; j++)); do for ((k=0; 7-$k; k++)); do 
screen -d -m -S w$i_$j_$k bash -c "python3 main.py $i $j $k" #& 
done; done; done;

