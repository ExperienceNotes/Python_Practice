for i in range(2,101,1):
    flag = 0
    for x in range(2,i,1):
        if i%x==0:
            flag = 1
            break
    if flag == 0:
        print('{} 是質數'.format(i))

print('方法二')
ans = [1] * 10
ans[0] = ans[1] = 0
for i in range(2,int(10**0.5)+1):
    if ans[i] == 1:
        for j in range(i+i,10,i):
            ans[j] = 0
print(sum(ans))
