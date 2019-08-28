import sys

def quicksort(array):
  # 基线条件：为空或者只包含一个元素的数组是“有序”的
  if len(array) < 2:
    return array
  # 递归条件
  else:
    pivot = array[0]
    # 由所有小于等于基准值的元素组成的数组
    less = [i for i in array[1:] if i <= pivot]
    # 由所有大于基准值的元素组成的数组
    greater = [i for i in array[1:] if i > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)

if __name__ == '__main__':
  num = []
  num.append(list(map(int, sys.stdin.readline().strip().split())))
  print (quicksort(num[0]))
