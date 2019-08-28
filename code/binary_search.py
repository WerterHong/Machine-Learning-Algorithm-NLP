import sys

def binary_search(list, item):
  low = 0
  high = len(list) - 1
  while low <= high:
    mid = (low + high) // 2
    guess = list[mid]
    if guess == item:
      return mid
    if guess > item:
      high = mid - 1
    else:
      low = mid + 1
  return None

if __name__ == '__main__':
  list_num = int(sys.stdin.readline().strip())
  list_1 = []
  for _ in range(list_num):
    list_1.append(list(map(int, sys.stdin.readline().strip().split()))
    print(binary_search(list_1[_], 4))
