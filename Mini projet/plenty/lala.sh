for i in {-2..2..0.1}; do for j in {-2..2..0.1}; do 
screen -d -m bash -c "python3 ex.py $i $j" #& 
done; done;