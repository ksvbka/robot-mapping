from typing import List

def sum_nested_list(nested_list: List):
    summ = 0

    for i in range(len(nested_list)):
        for j in range(len(nested_list[i])):
            summ += nested_list[i][j]

    return summ