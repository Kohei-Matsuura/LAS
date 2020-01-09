import numpy as np

def wer(hyp, ref):
    """
    hyp & ref must be word lists.
    """
    HYP_LEN = len(hyp)
    REF_LEN = len(ref)
    #print(HYP_LEN)
    #print(REF_LEN)
    #print(hyp)
    #print(ref)
    #sys.exit(0)

    d = np.array([[0 for i in range(HYP_LEN + 1)] for j in range(REF_LEN + 1)])

    # init distance table
    # each cells has [its distance, from where]
    for i in range(REF_LEN + 1):
        for j in range(HYP_LEN + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # calculate each cells
    for i in range(1, REF_LEN + 1):
        for j in range(1, HYP_LEN + 1):
            if hyp[j-1] == ref[i-1]:
                d[i][j] = min(d[i-1][j-1] + 0,
                              d[i][j-1] + 1,
                              d[i-1][j] + 1)
            else:
                d[i][j] = min(d[i-1][j-1] + 1,
                              d[i][j-1] + 1,
                              d[i-1][j] + 1)

    #print(d)
    route = []
    cell = [REF_LEN, HYP_LEN]
    val = d[cell[0]][cell[1]]
    search_token = [val, cell]
    #print(search_token)

    while(search_token[0] > 0):
        #print(search_token)
        cell_x = search_token[1][0]
        cell_y = search_token[1][1]
        if cell_x > -1 and cell_y > -1:
            prev_cell_val = min(d[cell_x-1][cell_y-1],
                        d[cell_x][cell_y-1],
                        d[cell_x-1][cell_y])
            if prev_cell_val == d[cell_x-1][cell_y-1]:
                if search_token[0] == prev_cell_val:
                    route.append('OK')
                    search_token = [prev_cell_val, [cell_x-1, cell_y-1]]
                else:
                    route.append('S')
                    search_token = [prev_cell_val, [cell_x-1, cell_y-1]]
            elif prev_cell_val == d[cell_x][cell_y-1]:
                route.append('I')
                search_token = [prev_cell_val, [cell_x, cell_y-1]]
            elif prev_cell_val == d[cell_x-1][cell_y]:
                route.append('D')
                search_token = [prev_cell_val, [cell_x-1, cell_y]]
        elif cell_x < 0:
            # Only way is Up (= Insert).
            route.append('I')
            search_token = [search_token[0] - 1, [cell_x, cell_y-1]]
        elif cell_y < 0:
            route.append('D')
            search_token = [search_token[0] - 1, [cell_x-1, cell_y]]
        else:
            print('ERROR: NO ROUTE')
            #sys.exit(0)

        route.reverse()

    S = route.count('S')
    D = route.count('D')
    I = route.count('I')
    WER = float((S + D + I) / REF_LEN)
    return S, D, I, WER
#print()
#print('REF_LEN:' + str(REF_LEN)) # equal to REF_LEN
#print('S: ' + str(S))
#print('D: ' + str(D))
#print('I: ' + str(I))
#print('-------------')
#print('WER: ' + str(WER))
