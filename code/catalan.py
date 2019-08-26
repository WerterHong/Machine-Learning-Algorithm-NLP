import sys

def catalan(n):
    if n == 0 or n == 1:
        return 1
    return int((4*n-2)*catalan(n-1)/(n+1))

if __name__ == '__main__':
  n = int(sys.stdin.readline().strip())
  print (catalan(n // 2))
