import numpy as np
from sklearn.model_selection import train_test_split

ls3class1='./rd_group9/class1.txt'
ls3class2='./rd_group9/class2.txt'
ls3class3='./rd_group9/class3.txt'

lsc1=np.loadtxt(ls3class1)
lsc2=np.loadtxt(ls3class2)
lsc3=np.loadtxt(ls3class3)

trainlsc1,testlsc1= train_test_split(lsc1,test_size=.25)
trainlsc2,testlsc2= train_test_split(lsc2,test_size=.25)
trainlsc3,testlsc3= train_test_split(lsc3,test_size=.25)

np.savetxt('./rd_group9/class1_train.txt',trainlsc1)
np.savetxt('./rd_group9/class1_test.txt',testlsc1)

np.savetxt('./rd_group9/class2_train.txt',trainlsc2)
np.savetxt('./rd_group9/class2_test.txt',testlsc2)

np.savetxt('./rd_group9/class3_train.txt',trainlsc3)
np.savetxt('./rd_group9/class3_test.txt',testlsc3)

# trainlsc1=np.loadtxt('./LS_Group09/Class1_train.txt')
