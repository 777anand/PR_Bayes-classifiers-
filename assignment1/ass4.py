import numpy as np
import matplotlib.pyplot as plt
import math

# trainlsc1=np.loadtxt('./LS_Group09/Class1_train.txt')
# testlsc1=np.loadtxt('./LS_Group09/Class1_test.txt')
# trainlsc2=np.loadtxt('./LS_Group09/Class2_train.txt')
# testlsc2=np.loadtxt('./LS_Group09/Class2_test.txt')
# trainlsc3=np.loadtxt('./LS_Group09/Class3_train.txt')
# testlsc3=np.loadtxt('./LS_Group09/Class3_test.txt')

# trainlsc1=np.loadtxt('./rd_group9/class1_train.txt')
# testlsc1=np.loadtxt('./rd_group9/class1_test.txt')
# trainlsc2=np.loadtxt('./rd_group9/class2_train.txt')
# testlsc2=np.loadtxt('./rd_group9/class2_test.txt')
# trainlsc3=np.loadtxt('./rd_group9/class3_train.txt')
# testlsc3=np.loadtxt('./rd_group9/class3_test.txt')

trainlsc1=np.loadtxt('./LS_Group09/Class1_train.txt')
testlsc1=np.loadtxt('./LS_Group09/Class1_test.txt')
trainlsc2=np.loadtxt('./LS_Group09/Class2_train.txt')
testlsc2=np.loadtxt('./LS_Group09/Class2_test.txt')
trainlsc3=np.loadtxt('./LS_Group09/Class3_train.txt')
testlsc3=np.loadtxt('./LS_Group09/Class3_test.txt')


mean_c1=np.mean(trainlsc1,axis=0)
mean_c2=np.mean(trainlsc2,axis=0)
mean_c3=np.mean(trainlsc3,axis=0)

# Case 1 ---------------------------------------------------------------------------------------------
# total_data=np.empty_like([[1,2]])
# total_data=np.concatenate((trainlsc1,trainlsc2,trainlsc3),axis=0)
# total_data=np.concatenate(trainlsc1,trainlsc2,axis=0)
# total_data=np.concatenate(total_data,trainlsc3,axis=0)
# print(total_data)
# total_variance=np.var(total_data,axis=0)
# print(total_variance)

variance_c1=np.var(trainlsc1,axis=0)
variance_c2=np.var(trainlsc2,axis=0)
variance_c3=np.var(trainlsc3,axis=0)
total_variance=(variance_c1+variance_c2+variance_c3)/3
common_variance=np.mean(total_variance)
print(common_variance)

w1=(mean_c1-mean_c2)/common_variance
w2=(mean_c2-mean_c3)/common_variance
w3=(mean_c3-mean_c1)/common_variance

w01=-1*(np.matmul(np.transpose(mean_c1),mean_c1) - np.matmul(np.transpose(mean_c2),mean_c2) )/(2*common_variance)
w02=-1*(np.matmul(np.transpose(mean_c2),mean_c2) - np.matmul(np.transpose(mean_c3),mean_c3) )/(2*common_variance)
w03=-1*(np.matmul(np.transpose(mean_c3),mean_c3) - np.matmul(np.transpose(mean_c1),mean_c1) )/(2*common_variance)

xs1 = [x[0] for x in trainlsc1]
ys1 = [x[1] for x in trainlsc1]
# plt.scatter(xs1, ys1)
xs2 = [x[0] for x in trainlsc2]
ys2 = [x[1] for x in trainlsc2]
# plt.scatter(xs2, ys2)
xs3 = [x[0] for x in trainlsc3]
ys3 = [x[1] for x in trainlsc3]

covariance_mat1=np.cov(xs1,ys1)
covariance_mat2=np.cov(xs2,ys2)
covariance_mat3=np.cov(xs3,ys3)


# print(covariance_mat)

def g1(x):
	return np.matmul(np.transpose(w1),x) + w01

def g2(x):
	return np.matmul(np.transpose(w2),x) + w02

def g3(x):
	return np.matmul(np.transpose(w3),x) + w03


def g(x,mean_i,covariance_mat):
	return -1*np.matmul(np.transpose(x-mean_i),np.matmul(np.linalg.inv(covariance_mat),(x-mean_i)))/2 - math.log(np.linalg.det(covariance_mat))/2

def classify(x,mean_c1,mean_c2,mean_c3,covariance_mat):
	posterior_prob=[0,0,0]
	posterior_prob[0]=g(i,mean_c1,covariance_mat1)
	posterior_prob[1]=g(i,mean_c2,covariance_mat2)
	posterior_prob[2]=g(i,mean_c3,covariance_mat3)
	# print(posterior_prob)
	return np.argmax(posterior_prob)


# class1=[]

# for i in testlsc1:
# 	posterior_prob=[0,0,0]
# 	posterior_prob[0]=g(i,mean_c1,covariance_mat)
# 	posterior_prob[1]=g(i,mean_c2,covariance_mat)
# 	posterior_prob[2]=g(i,mean_c3,covariance_mat)
# 	class1=class1+[np.argmax(posterior_prob)+1]

# print(class1)

# class2=[]

# for i in testlsc2:
# 	posterior_prob=[0,0,0]
# 	posterior_prob[0]=g(i,mean_c1,covariance_mat)
# 	posterior_prob[1]=g(i,mean_c2,covariance_mat)
# 	posterior_prob[2]=g(i,mean_c3,covariance_mat)
# 	class2=class2+[np.argmax(posterior_prob)+1]

# print(class2)

# class3=[]

# for i in testlsc3:
# 	posterior_prob=[0,0,0]
# 	posterior_prob[0]=g(i,mean_c1,covariance_mat)
# 	posterior_prob[1]=g(i,mean_c2,covariance_mat)
# 	posterior_prob[2]=g(i,mean_c3,covariance_mat)
# 	class3=class3+[np.argmax(posterior_prob)+1]

# print(class3)

# plt.scatter(xs3, ys3)
# plt.show()

min_y=np.amin(np.concatenate((ys1,ys2,ys3),axis=0))
min_x=np.amin(np.concatenate((xs1,xs2,xs3),axis=0))
max_y=np.amax(np.concatenate((ys1,ys2,ys3),axis=0))
max_x=np.amax(np.concatenate((xs1,xs2,xs3),axis=0))
pointsx=np.arange(min_x,max_x,.1)
pointsy=np.arange(min_y,max_y,.1)
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
		posterior_prob[0]=g(i,mean_c1,covariance_mat1)
		posterior_prob[1]=g(i,mean_c2,covariance_mat2)
		posterior_prob[2]=g(i,mean_c3,covariance_mat3)
		result=np.argmax(posterior_prob)
			
			


		# result=classify(x,mean_c1,mean_c2,mean_c3,covariance_mat)
		if result==0:
			b_class1=b_class1+[x]
		elif result==1:
			b_class2=b_class2+[x]
		elif result==2:
			b_class3=b_class3+[x]

# print(b_class1)
# print(b_class2)
# print(b_class3)
# print(len(b_class2))
# print(len(b_class3))
# np.delete()
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
plt.show()
