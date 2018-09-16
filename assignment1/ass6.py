import numpy as np
import matplotlib.pyplot as plt
import math
import sys

if sys.argv[1]=='1':
	trainlsc1=np.loadtxt('./LS_Group09/Class1_train.txt')
	test1=np.loadtxt('./LS_Group09/Class1_test.txt')
	trainlsc2=np.loadtxt('./LS_Group09/Class2_train.txt')
	test2=np.loadtxt('./LS_Group09/Class2_test.txt')
	trainlsc3=np.loadtxt('./LS_Group09/Class3_train.txt')
	test3=np.loadtxt('./LS_Group09/Class3_test.txt')
elif sys.argv[1]=='2':
	trainlsc1=np.loadtxt('./rd_group9/class1_train.txt')
	test1=np.loadtxt('./rd_group9/class1_test.txt')
	trainlsc2=np.loadtxt('./rd_group9/class2_train.txt')
	test2=np.loadtxt('./rd_group9/class2_test.txt')
	trainlsc3=np.loadtxt('./rd_group9/class3_train.txt')
	test3=np.loadtxt('./rd_group9/class3_test.txt')
elif sys.argv[1]=='3':
	trainlsc1=np.loadtxt('rwclass1_train.txt')
	test1=np.loadtxt('rwclass1_test.txt')
	trainlsc2=np.loadtxt('rwclass2_train.txt')
	test2=np.loadtxt('rwclass2_test.txt')
	trainlsc3=np.loadtxt('rwclass3_train.txt')
	test3=np.loadtxt('rwclass3_test.txt')

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
	covariance_mat3=covariance_mat1
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

pointsx=np.arange(min_x,max_x,.03)
pointsy=np.arange(min_y,max_y,.03)
xx,yy=np.meshgrid(pointsx,pointsy)
# points=np.concatenate((pointsx,pointsy),axis=1)

b_class1=[]
b_class2=[]
b_class3=[]
for i in pointsx:
	for j in pointsy:
		x=np.array([i,j])
		posterior_prob=[0,0]
		posterior_prob[0]=g(x,mean_c1,covariance_mat1,prob1)
		posterior_prob[1]=g(x,mean_c2,covariance_mat2,prob2)
		#posterior_prob[2]=g(x,mean_c3,covariance_mat3,prob3)
		result=np.argmax(posterior_prob)
			
		if result==0:
			b_class1=b_class1+[x]
		elif result==1:
			b_class2=b_class2+[x]
		# elif result==2:
		# 	b_class3=b_class3+[x]
xc1 = [x[0] for x in b_class1] 
yc1 = [x[1] for x in b_class1]

xc2 = [x[0] for x in b_class2]
yc2 = [x[1] for x in b_class2]


plt.scatter(xc1,yc1,label='Class 1- decision surface')
plt.scatter(xc2,yc2,label='Class 2- decision surface')
plt.scatter(xs1,ys1,label='Class 1')
plt.scatter(xs2,ys2,label='Class 2')
plt.axis('equal')
plt.legend()
plt.show()


b_class1=[]
b_class2=[]
b_class3=[]
for i in pointsx:
	for j in pointsy:
		x=np.array([i,j])
		posterior_prob=[0,0]
		posterior_prob[0]=g(x,mean_c1,covariance_mat1,prob1)
		# posterior_prob[1]=g(x,mean_c2,covariance_mat2,prob2)
		posterior_prob[1]=g(x,mean_c3,covariance_mat3,prob3)
		result=np.argmax(posterior_prob)
			
		if result==0:
			b_class1=b_class1+[x]
		elif result==1:
			b_class2=b_class2+[x]
		# elif result==2:
		# 	b_class3=b_class3+[x]
xc1 = [x[0] for x in b_class1] 
yc1 = [x[1] for x in b_class1]

xc2 = [x[0] for x in b_class2]
yc2 = [x[1] for x in b_class2]

plt.scatter(xc1,yc1,label='Class 1- decision surface')
plt.scatter(xc2,yc2,label='Class 3- decision surface')
plt.scatter(xs1,ys1,label='Class 1')
plt.scatter(xs3,ys3,label='Class 3')
plt.axis('equal')

plt.legend()
plt.show()


b_class1=[]
b_class2=[]
b_class3=[]
for i in pointsx:
	for j in pointsy:
		x=np.array([i,j])
		posterior_prob=[0,0]
		# posterior_prob[0]=g(x,mean_c1,covariance_mat1,prob1)
		posterior_prob[0]=g(x,mean_c2,covariance_mat2,prob2)
		posterior_prob[1]=g(x,mean_c3,covariance_mat3,prob3)
		result=np.argmax(posterior_prob)
			
		if result==0:
			b_class1=b_class1+[x]
		elif result==1:
			b_class2=b_class2+[x]
		# elif result==2:
		# 	b_class3=b_class3+[x]
xc1 = [x[0] for x in b_class1] 
yc1 = [x[1] for x in b_class1]

xc2 = [x[0] for x in b_class2]
yc2 = [x[1] for x in b_class2]


plt.scatter(xc2,yc2,label='Class 3- decision surface')
plt.scatter(xc1,yc1,label='Class 2- decision surface')
plt.scatter(xs3,ys3,label='Class 3')
plt.scatter(xs2,ys2,label='Class 2')
plt.axis('equal')

plt.legend()
plt.show()


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
plt.scatter(xc1,yc1,label='Class 1- decision surface')
plt.scatter(xc2,yc2,label='Class 2- decision surface')
plt.scatter(xc3,yc3,label='Class 3- decision surface')
plt.scatter(xs1, ys1,label='Class 1')
plt.scatter(xs2, ys2,label='Class 2')
plt.scatter(xs3, ys3,label='Class 3')
plt.axis('equal')
plt.legend()
plt.show()


#--------------------Confusion matrix------------------------------------
c11 = []
c12 = []
c13 = []

c21 = []
c22 = []
c23 = []

c31 = []
c32 = []
c33 = []

for point in test1:
	posterior_prob=[0,0,0]
	posterior_prob[0]=g(point,mean_c1,covariance_mat1,prob1)
	posterior_prob[1]=g(point,mean_c2,covariance_mat2,prob2)
	posterior_prob[2]=g(point,mean_c3,covariance_mat3,prob3)
	result=np.argmax(posterior_prob)
		
	if result==0:
		c11=c11+[point]
	elif result==1:
		c12=c12+[point]
	elif result==2:
		c13=c13+[point]


for point in test2:
	posterior_prob=[0,0,0]
	posterior_prob[0]=g(point,mean_c1,covariance_mat1,prob1)
	posterior_prob[1]=g(point,mean_c2,covariance_mat2,prob2)
	posterior_prob[2]=g(point,mean_c3,covariance_mat3,prob3)
	result=np.argmax(posterior_prob)
		
	if result==0:
		c21=c21+[point]
	elif result==1:
		c22=c22+[point]
	elif result==2:
		c23=c23+[point]

for point in test3:
	posterior_prob=[0,0,0]
	posterior_prob[0]=g(point,mean_c1,covariance_mat1,prob1)
	posterior_prob[1]=g(point,mean_c2,covariance_mat2,prob2)
	posterior_prob[2]=g(point,mean_c3,covariance_mat3,prob3)
	result=np.argmax(posterior_prob)
		
	if result==0:
		c31=c31+[point]
	elif result==1:
		c32=c32+[point]
	elif result==2:
		c33=c33+[point]

confusion_matrix =[[len(c11),len(c12),len(c13)],[len(c21),len(c22),len(c23)],[len(c31),len(c32),len(c33)]]
print("Confusion Matrix")
print(confusion_matrix[0])
print(confusion_matrix[1])
print(confusion_matrix[2])


accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/(len(test1)+len(test2)+len(test3))
print("The accuracy is -",accuracy*100,"%")
print(" ")

precision_c1 = (confusion_matrix[0][0]/len(test1))
precision_c2 = (confusion_matrix[1][1]/len(test2))
precision_c3 = (confusion_matrix[2][2]/len(test3))
print("Precision of class 1-",precision_c1)
print("Precision of class 2-",precision_c2)
print("Precision of class 3-",precision_c3)
print("Mean Precision - ", (precision_c1+precision_c2+precision_c3)/3)
print(" ")

recall_c1 = (confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[2][0]))
recall_c2 = (confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1]+confusion_matrix[2][1]))
recall_c3 = (confusion_matrix[2][2]/(confusion_matrix[0][2]+confusion_matrix[1][2]+confusion_matrix[2][2]))
print("Recall of class 1-",recall_c1)
print("Recall of class 2-",recall_c2)
print("Recall of class 3-",recall_c3)
print("Mean Recall -",(recall_c1+recall_c2+recall_c3)/3)
print(" ")

f_measure1 = 2*(precision_c1*recall_c1)/(precision_c1+recall_c1)
f_measure2 = 2*(precision_c2*recall_c2)/(precision_c2+recall_c2)
f_measure3 = 2*(precision_c3*recall_c3)/(precision_c3+recall_c3)
print("F measure of class 1-",f_measure1)
print("F measure of class 2-",f_measure2)
print("F measure of class 3-",f_measure3)
print("Mean F Measure - ",(f_measure1+f_measure2+f_measure3)/3)


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
plt.scatter(xs1,ys1,label='Class 1')
plt.scatter(xs2,ys2,label='Class 2')
plt.axis('equal')
plt.legend()
plt.show()

plt.contour(xx,yy,z1)
plt.contour(xx,yy,z3)
plt.scatter(xs3,ys3,label='Class 3')
plt.scatter(xs1,ys1,label='Class 1')
plt.axis('equal')
plt.legend()
plt.show()

plt.contour(xx,yy,z2)
plt.contour(xx,yy,z3)
plt.scatter(xs3,ys3,label='Class 3')
plt.scatter(xs2,ys2,label='Class 2')
plt.axis('equal')
plt.legend()
plt.show()

plt.contour(xx,yy,z1)
plt.contour(xx,yy,z2)
plt.contour(xx,yy,z3)
plt.scatter(xs1,ys1,label='Class 1')
plt.scatter(xs2,ys2,label='Class 2')
plt.scatter(xs3,ys3,label='Class 3')
plt.legend()
plt.axis('equal')


plt.show()
