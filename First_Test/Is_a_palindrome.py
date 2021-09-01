word = input('請輸入一個單字:')
for i in range(len(word)//2):
    flag = 0
    print('{} word[{}],{} word[-{}]'.format(word[i],i,word[-(i+1)],i+1))
    if word[i]==word[-(i+1)]:
        flag = 1
        pass
    else:
        flag = 0
        break
if flag == 1:
    print('{}是迴文:'.format(word))
else:
    print('{}不是迴文:'.format(word))