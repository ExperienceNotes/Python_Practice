import numpy as np
import os
def F(n,m):
    if n == 1 or m == 1 :
        return 1
    elif n <= m:
        return 1 + F(n,n-1)
    elif n > m:
        return F(n,m-1)+F(n-m,m)
n = eval(input('number:'))
ans = F(n,n)
print('ans:',ans)
os.system('pause')