word = input('請輸入一個句子')
upper = ""
for i in word:
    if i.isupper():
        upper = upper+i
print('{}為句子{}簡寫'.format(upper,word))