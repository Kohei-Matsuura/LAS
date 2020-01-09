with open('wav.list') as f:
    for l in f:
        l = l.strip()
        zero_one = l.split('.')[0]
        zero_one_list = zero_one.split('_')
        yes_no_list = []
        for e in zero_one_list:
            if e == '0':
                yes_no_list.append('no')
            else:
                yes_no_list.append('yes')
        yes_no =' '.join(yes_no_list)
        print('data/toy/{0} <sos> {1} <eos>'.format(l, yes_no))
