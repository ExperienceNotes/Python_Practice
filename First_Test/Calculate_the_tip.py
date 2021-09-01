bill = eval(input('請輸入帳單金額:'))
Rate = eval(input('請輸入費率x%:'))
tip = bill*Rate/100
print('在費率{}%,小費:{}元,總金額:{}'.format(Rate,tip,bill+tip))