number = eval(input('請輸入1-1000的數字:'))
if number > 1000 or number < 1:
    print('必須輸出1-1000內的數字')
else:
    if number%2==0:
        print('{} 是偶數'.format(number))
    else:
        print('{} 是奇數'.format(number))