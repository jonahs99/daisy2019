def gen():
    possible = []
    prices = {1:0, 2: 2, 3:3, 4:4, 5:6}
    for i1 in range(1, 6):
        for i2 in range(1, 6):
            for i3 in range(1, 6):
                for i4 in range(1, 6):
                    for i5 in range(1, 6):
                        for i6 in range(1, 6):
                            test = [i1, i2, i3, i4, i5, i6]
                            sum = 0
                            for k in range(6):
                                sum += prices[test[k]]
                            if sum == 18:
                                possible.append(test)
    return possible
