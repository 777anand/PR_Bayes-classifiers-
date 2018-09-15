import numpy as np
import matplotlib.pyplot as plt
import math
import sys

if sys.argv[1]=='1':
	trainlsc1=np.loadtxt('./LS_Group09/Class1_train.txt')
	testlsc1=np.loadtxt('./LS_Group09/Class1_test.txt')
	trainlsc2=np.loadtxt('./LS_Group09/Class2_train.txt')
	testlsc2=np.loadtxt('./LS_Group09/Class2_test.txt')
	trainlsc3=np.loadtxt('./LS_Group09/Class3_train.txt')
	testlsc3=np.loadtxt('./LS_Group09/Class3_test.txt')
elif sys.argv[1]=='2':
	trainlsc1=np.loadtxt('./rd_group9/class1_train.txt')
	testlsc1=np.loadtxt('./rd_group9/class1_test.txt')
	trainlsc2=np.loadtxt('./rd_group9/class2_train.txt')
	testlsc2=np.loadtxt('./rd_group9/class2_test.txt')
	trainlsc3=np.loadtxt('./rd_group9/class3_train.txt')
	testlsc3=np.loadtxt('./rd_group9/class3_test.txt')
elif sys.argv[1]=='3':
	trainlsc1=np.loadtxt('rwclass1_train.txt')
	testlsc1=np.loadtxt('rwclass1_test.txt')
	trainlsc2=np.loadtxt('rwclass2_train.txt')
	testlsc2=np.loadtxt('rwclass2_test.txt')
	trainlsc3=np.loadtxt('rwclass3_train.txt')
	testlsc3=np.loadtxt('rwclass3_test.txt')

mean_c1=np.mean(trainlsc1,axis=0)
mean_c2=np.mean(trainlsc2,axis=0)
mean_c3=np.mean(trainlsc3,axis=0)

xs1 = [x[0] for x in trainlsc1]
ys1 = [x[1] for x in trainlsc1]
# plt.scatter(xs1, ys1)
xs2 = [x[0] for x in trainlsc2]
ys2 = [x[1] for x in trainlsc2]
# plt.scatter(xs2, ys2)
xs3 = [x[0] for x in trainlsc3]
ys3 = [x[1] for x in trainlsc3]



if sys.argv[2]=='1':
	variance_c1=np.var(trainlsc1,axis=0)
	variance_c2=np.var(trainlsc2,axis=0)
	variance_c3=np.var(trainlsc3,axis=0)
	total_variance=(variance_c1+variance_c2+variance_c3)/3
	common_variance=np.mean(total_variance)
	covariance_mat1=[[common_variance,0],[0,common_variance]]
	covariance_mat2=covariance_mat1
	covariance_mat3=covariance_mat1
elif sys.argv[2]=='2':
	covariance_mat1=(np.cov(xs1,ys1)+np.cov(xs2,ys2)+np.cov(xs3,ys3))/3
	covariance_mat2=covariance_mat1
	covariance_mat3=covariance_mat1s
elif sys.argv[2]=='3':
	covariance_mat1=np.cov(xs1,ys1)
	covariance_mat2=np.cov(xs2,ys2)
	covariance_mat3=np.cov(xs3,ys3)
	covariance_mat1[0][1]=0
	covariance_mat1[1][0]=0
	covariance_mat2[0][1]=0
	covariance_mat2[1][0]=0
	covariance_mat3[0][1]=0
	covariance_mat3[1][0]=0
elif sys.argv[2]=='4':
	covariance_mat1=np.cov(xs1,ys1)
	covariance_mat2=np.cov(xs2,ys2)
	covariance_mat3=np.cov(xs3,ys3)

prob1=len(trainlsc1)/(len(trainlsc1)+len(trainlsc2)+len(trainlsc3))
prob2=len(trainlsc2)/(len(trainlsc1)+len(trainlsc2)+len(trainlsc3))
prob3=len(trainlsc3)/(len(trainlsc1)+len(trainlsc2)+len(trainlsc3))

def g(x,mean_i,covariance_mat,prob_i):
	return -1*np.matmul(np.transpose(x-mean_i),np.matmul(np.linalg.inv(covariance_mat),(x-mean_i)))/2 - math.log(np.linalg.det(covariance_mat))/2 + math.log(prob_i)

min_y=np.amin(np.concatenate((ys1,ys2,ys3),axis=0))
min_x=np.amin(np.concatenate((xs1,xs2,xs3),axis=0))
max_y=np.amax(np.concatenate((ys1,ys2,ys3),axis=0))
max_x=np.amax(np.concatenate((xs1,xs2,xs3),axis=0))
pointsx=np.arange(min_x,max_x,10)
pointsy=np.arange(min_y,max_y,10)
xx,yy=np.meshgrid(pointsx,pointsy)
# points=np.concatenate((pointsx,pointsy),axis=1)

b_class1=[]
b_class2=[]
b_class3=[]
count=0
for i in pointsx:
	for j in pointsy:
		x=np.array([i,j])
		posterior_prob=[0,0,0]
		posterior_prob[0]=g(x,mean_c1,covariance_mat1,prob1)
		posterior_prob[1]=g(x,mean_c2,covariance_mat2,prob2)
		posterior_prob[2]=g(x,mean_c3,covariance_mat3,prob3)
		result=np.argmax(posterior_prob)
			
		if result==0:
			b_class1=b_class1+[x]
		elif result==1:
			b_class2=b_class2+[x]
		elif result==2:
			b_class3=b_class3+[x]


xc1 = [x[0] for x in b_class1]
yc1 = [x[1] for x in b_class1]

xc2 = [x[0] for x in b_class2]
yc2 = [x[1] for x in b_class2]

xc3 = [x[0] for x in b_class3]
yc3 = [x[1] for x in b_class3]

# plt.scatter(xx,yy)
plt.scatter(xc1,yc1)
plt.scatter(xc2,yc2)
plt.scatter(xc3,yc3)
plt.scatter(xs1, ys1)
plt.scatter(xs2, ys2)
plt.scatter(xs3, ys3)

# ----------------------------Contour plot---------------------------------------------------------------------

def density(x,mean_i,covariance_mat):
	return np.exp(-1*np.matmul((np.transpose(x-mean_i)),np.matmul((np.linalg.inv(covariance_mat)),(x-mean_i)))/2)/(math.sqrt(2*3.14*np.linalg.det(covariance_mat)))
z1=[]

for i in pointsx:
	for j in pointsy:
		x=np.array([i,j])
		z1=z1+[density(x,mean_c1,covariance_mat1)]
		# print(density(x,mean_c1,covariance_mat1))
	
z1 =np.transpose(np.reshape(z1,(len(pointsx), len(pointsy))))

z2=[]

for i in pointsx:
	for j in pointsy:
		x=np.array([i,j])
		z2=z2+[density(x,mean_c2,covariance_mat2)]
		# print(density(x,mean_c1,covariance_mat1))
	
z2 =np.transpose(np.reshape(z2,(len(pointsx), len(pointsy))))

z3=[]

for i in pointsx:
	for j in pointsy:
		x=np.array([i,j])
		z3=z3+[density(x,mean_c3,covariance_mat3)]
		# print(density(x,mean_c1,covariance_mat1))
	
z3 =np.transpose(np.reshape(z3,(len(pointsx), len(pointsy))))
# zz=density(z,mean_c1,covariance_mat1)
plt.contour(xx,yy,z1)
plt.contour(xx,yy,z2)
plt.contour(xx,yy,z3)

plt.show()
