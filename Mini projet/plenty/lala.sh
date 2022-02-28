for i in {-2..2..0.1}; do for i in {-2..2..0.1}; do 
screen -d -m -S w$i$j bash -c "python3 main.py $i $j" #& 
done; done;