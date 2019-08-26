import sys

def solve(eq, var='x'):
  eq1 = eq.replace("=","-(")+")"
  c = eval(eq1,{var:1j})
  if (c.imag == 0):
    return -1
  else:
    ans = -c.real/c.imag
    if (ans - (int(ans)) == 0) and (ans > 0):
      return int(ans)
    else:
      return -1

if __name__ == '__main__':
  eq = sys.stdin.readline().strip()
  print (solve(eq))
