leftover = open('leftover.txt', 'r')

N_string = ['2', '5', '8', '11']
T_string = ['10', '100', '1000']
E_liste = ['cos(x)', 'tanh', '0', '1', 'croissant', 'random', 'Par_palier', 'sin_amorti','cos_petit']

for i in leftover:
    r = i.strip().split(',')
    print(str(N_string.index(r[0])) + ',' + str(T_string.index(r[1])) + ',' + str(E_liste.index(r[2])))

