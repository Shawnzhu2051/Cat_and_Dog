# coding: utf-8
import cv2
import numpy as np
import struct

n = 1
table1 = []
table2 = []
table3 = []
table4 = []
table5 = []
test_table = []
count = 0


def mybin(num):
    bstr = bin(num)
    bstr = '0' + bstr
    return bstr.replace('0b', '')


def myhex(num):
    hstr = hex(num).replace('0x', '')
    if len(hstr) == 1:
        hstr = '0' + hstr
    return hstr


def processing(n, table, flag):
    if flag == 0:
        sequence = str(n)
        string = 'cat.' + sequence
    else:
        sequence = str(n - 12500)
        string = 'dog.' + sequence
    image = cv2.imread('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train/' + string + '.jpg')
    thirtytwo = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)

    item = []
    flag = mybin(flag)
    item.append(flag)

    for i in range(32):
        for j in range(0, 32, 2):
            temp1 = myhex(thirtytwo[i, j, 0])
            temp2 = myhex(thirtytwo[i, j + 1, 0])
            item.append(temp1 + temp2)
    for i in range(32):
        for j in range(0, 32, 2):
            temp1 = myhex(thirtytwo[i, j, 1])
            temp2 = myhex(thirtytwo[i, j + 1, 1])
            item.append(temp1 + temp2)
    for i in range(32):
        for j in range(0, 32, 2):
            temp1 = myhex(thirtytwo[i, j, 2])
            temp2 = myhex(thirtytwo[i, j + 1, 2])
            item.append(temp1 + temp2)
    table.append(item)


while n < 25:
    if n < 5000:
        processing(n, table1, 0)
    elif n < 10000:
        processing(n, table2, 0)
    elif n < 12500:
        processing(n, table3, 0)
    elif n < 15000:
        processing(n, table3, 1)
    elif n < 20000:
        processing(n, table4, 1)
    else:
        processing(n, table5, 1)
    print(n)
    n = n + 1

with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data1.txt', 'w') as f:
    for item in table1:
        for ele in item:
            f.writelines(ele)

count1 = 1
with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data1.txt', 'r+') as filehandler:
    with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/newtxt1.txt','w') as filehandler2:
        for fh in filehandler:
            for f in fh:
                filehandler2.write(f)
                if count1 % 4 == 0:
                    filehandler2.write(' ')
                count1 = count1 + 1
                if (count1-1) % 32 == 0:
                    filehandler2.write('\n')



with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data2.txt', 'w') as f:
    for item in table2:
        for ele in item:
            f.writelines(ele)

count1 = 1
with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data2.txt', 'r+') as filehandler:
    with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/newtxt2.txt','w') as filehandler2:
        for fh in filehandler:
            for f in fh:
                filehandler2.write(f)
                if count1 % 4 == 0:
                    filehandler2.write(' ')
                count1 = count1 + 1
                if (count1-1) % 32 == 0:
                    filehandler2.write('\n')



with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data3.txt', 'w') as f:
    for item in table3:
        for ele in item:
            f.writelines(ele)
count1 = 1
with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data3.txt', 'r+') as filehandler:
    with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/newtxt3.txt','w') as filehandler2:
        for fh in filehandler:
            for f in fh:
                filehandler2.write(f)
                if count1 % 4 == 0:
                    filehandler2.write(' ')
                count1 = count1 + 1
                if (count1-1) % 32 == 0:
                    filehandler2.write('\n')


with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data4.txt', 'w') as f:
    for item in table4:
        for ele in item:
            f.writelines(ele)
count1 = 1
with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data4.txt', 'r+') as filehandler:
    with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/newtxt4.txt','w') as filehandler2:
        for fh in filehandler:
            for f in fh:
                filehandler2.write(f)
                if count1 % 4 == 0:
                    filehandler2.write(' ')
                count1 = count1 + 1
                if (count1-1) % 32 == 0:
                    filehandler2.write('\n')



with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data5.txt', 'w') as f:
    for item in table5:
        for ele in item:
            f.writelines(ele)
            count1 = 1
with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/train_data5.txt', 'r+') as filehandler:
    with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/newtxt5.txt','w') as filehandler2:
        for fh in filehandler:
            for f in fh:
                filehandler2.write(f)
                if count1 % 4 == 0:
                    filehandler2.write(' ')
                count1 = count1 + 1
                if (count1-1) % 32 == 0:
                    filehandler2.write('\n')


'''

while n <= 12500:
    sequence = str(n)
    image = cv2.imread('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/test/' + sequence + '.jpg')
    thirtytwo = cv2.resize(image,(32,32),interpolation = cv2.INTER_AREA)
    item = []
    item.append(2)
    for i in range(32):
       for j in range(32):
             item.append(thirtytwo[i,j,0])
    for i in range(32):
        for j in range(32):
            item.append(thirtytwo[i,j,1])
    for i in range(32):
        for j in range(32):
            item.append(thirtytwo[i,j,2])
    test_table.append(item)
    print(n)
    n = n + 1
with open('/Users/shawnzhu/Desktop/Project/Cat_and_Dog/test_data2.txt', 'w') as f:
    for item in test_table:
        f.write(str(item) + '\n')
        print(count)
        count = count + 1
'''
