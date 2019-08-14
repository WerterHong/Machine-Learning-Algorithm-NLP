import sys
import math

def var(nums):
  num_avg, num_var = 0, 0
  num_avg = sum(nums) / len(nums)
  for i in nums:
    num_var += math.pow(i - num_avg, 2)
  return num_var / len(nums)

def cal_var(nums):
  if not nums:
    return 0
  min_var = float("inf")
  for i in range(len(nums) - 2):
    num_var = var([nums[i], nums[i + 1], nums[i + 2]])
    min_var = min(min_var,num_var)

  print ('% .2f' % min_var)
  return min_var

if __name__ == '__main__':
  nums = [10, -1, 0, 1, 3]
  cal_var(list(sorted(nums)))
