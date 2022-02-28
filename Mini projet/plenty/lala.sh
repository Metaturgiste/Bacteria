for i in {-2..2..0.1}; do for j in {-2..2..0.1}; do 
screen -d -m -S w$i$j bash -c "python3 ex.py $i $j" #& 
done; done;