import numpy as np
import matplotlib.pyplot as plt
import sys

np.random.seed(123)

def true_f(x):
    return 4*np.sin(np.pi * x)*np.cos(6* np.pi * x**2)

def k_poly(x, z, d):
    return (1 + np.dot(x,z.T))**d

def k_rbf(x, z, gamma):
    return np.exp(-gamma * (x-z.T)**2)

def fit(y, K, lam):
    return np.linalg.solve( K + lam*np.identity(y.size),y)

def predict(K, alpha):
    return K.dot(alpha)

def error(K, y, alpha):
    n = y.size
    f_hat = predict(K,alpha)
    return 1/n * ((y-f_hat)**2).sum()

def KFold(x, y, lam, param, is_poly = True, k=10):
    step = int(y.size/k)
    err = np.zeros(k)
    for i in range(k):
        train_x = np.delete(x, np.arange(i*step,(i+1)*step) ,axis = 0)
        valid_x = x[i*step:(i+1)*step]
        train_y = np.delete(y, np.arange(i*step,(i+1)*step)  ,axis = 0)
        valid_y = y[i*step:(i+1)*step]

        if is_poly:
            K = k_poly(train_x, train_x, param)
            K_valid = k_poly(valid_x, train_x,param)
        else:
            K = k_rbf(train_x, train_x, param)
            K_valid =  k_rbf(valid_x,train_x, param)
        # print(i)
        a = fit(train_y, K, lam)
        # print(a)
        err[i] = error(K_valid, valid_y, a)
    # print(err)
    print(err.mean())
    return err.mean()

#####################################
# a(i)
n = 30
x_fine = np.arange(0,1.01,0.01)
x = np.random.uniform(0, 1, size=(n,1))
# x = np.sort(x,axis =  0)
e = np.random.normal(0,1,size=(n,1))
# f_star = 4*np.sin(np.pi * x)*np.cos(6* np.pi * x**2)
y = true_f(x) + e 

min_err = sys.float_info.max
for lam in np.arange(0.001,0.5,0.003):
    # min_err.append( LOO(x, y, lam,param=0))
    for d in np.arange(1,50,1):
        err = KFold(x, y, lam,param=d, is_poly = True, k=y.size)
        if err<min_err:
            min_err = err
            min_lam = lam
            min_d = d

print(min_err) 
print(min_lam)
print(min_d)
# min_d = 49
# min_lam=0.036


K_poly = k_poly(x, x, min_d)
a_poly = fit(y, K_poly, min_lam)
K_poly_fine =  k_poly(x_fine.reshape(x_fine.size, 1), x, min_d)
y_poly = predict(K_poly_fine, a_poly)

plt.plot(x, y, '.',c='black')
plt.plot(x_fine,y_poly, label = '$\hat{f}_{poly}(x)$')
plt.plot(x_fine, true_f(x_fine), label = 'True $f(x)$')
plt.legend( frameon=False)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Kernel d = '+str(min_d)+', $\lambda$ = '+str(round(min_lam,4)))
# plt.ylim((-10,10))
plt.savefig('A3b-1_new.png')
plt.show()
#####################################
# a(ii)

min_err = sys.float_info.max
for lam in np.arange(0.001,0.5,0.005):
    # min_err.append( LOO(x, y, lam,param=0))
    for g in np.arange(50,300,5):
        err = KFold(x, y, lam, param=g, is_poly=False, k=y.size)
        if err < min_err:
            min_err = err
            min_lam_rbf = lam
            min_g = g

print(min_err) 
print(min_lam_rbf)
print(min_g)

# lam = 0.001, gamma = 26 


K_rbf= k_rbf(x, x, min_g)
a_rbf = fit(y, K_rbf, min_lam_rbf)
# y_rbf = predict(K_rbf, a_rbf)
K_rbf_fine =  k_rbf(x_fine.reshape(x_fine.size, 1), x, min_g)
y_rbf = predict(K_rbf_fine, a_rbf)

# b

plt.plot(x, y, '.',c='black')
plt.plot(x_fine, true_f(x_fine), label = 'True $f(x)$')
plt.plot(x_fine, y_rbf , label =  '$\hat{f}_{rbf}(x)$')
plt.legend( frameon=False)
plt.xlabel('x')
plt.ylabel('y')
plt.title('RBF Kernel $\gamma$ = '+str(min_g)+', $\lambda$ = '+str(round(min_lam_rbf,4)))

plt.savefig('A3b-2_new.png')
plt.show()
#############################
# c

B = 300
n = 30
y_poly_list=[]
y_rbf_list=[]

for i in range(B):
    sample_index = np.random.choice(np.arange(n), n)
    train_x = x[sample_index]
    train_y = y[sample_index]

    K_train_poly = k_poly(train_x, train_x, min_d)
    a_poly = fit(train_y, K_train_poly, min_lam)
    K_poly = k_poly(x_fine.reshape(x_fine.size, 1), train_x,  min_d)
    y_poly_list.append( predict(K_poly, a_poly).reshape(x_fine.size) )


    K_train_rbf= k_rbf(train_x, train_x, min_g)
    a_rbf = fit(train_y, K_train_rbf, min_lam_rbf)
    K_rbf =  k_rbf(x_fine.reshape(x_fine.size, 1), train_x,  min_g)
    y_rbf_list.append( predict(K_rbf, a_rbf).reshape(x_fine.size))

poly_low = np.percentile(y_poly_list, 5, axis=0)
poly_high = np.percentile(y_poly_list, 95, axis=0)

rbf_low = np.percentile(y_rbf_list, 5, axis=0)
rbf_high = np.percentile(y_rbf_list, 95, axis=0)


plt.plot(x, y, '.', c='black')
plt.plot(x_fine, true_f(x_fine), label = 'True $f(x)$')
plt.plot(x_fine,y_poly, label = '$\hat{f}_{poly}(x)$')
plt.fill_between(x_fine, poly_low, poly_high,color='grey', alpha = 0.25, label='95% CI')
# plt.plot(x, poly_low,'--', c='grey')
# plt.plot(x, poly_high, '--',label = '95% CI', c='grey')
plt.legend( frameon=False)
plt.xlabel('x')
plt.ylabel('y')
plt.title('95% Confidence Interval for $\hat{f}_{poly}(x)$')
# plt.xlim((-0.1,1.1))
plt.ylim((-8,30))
plt.savefig('A3c-1_filled.png')
plt.show()

plt.plot(x, y, '.', c='black')
plt.plot(x_fine, true_f(x_fine), label = 'True $f(x)$')
plt.plot(x_fine, y_rbf , label =  '$\hat{f}_{rbf}(x)$')
plt.fill_between(x_fine, rbf_low, rbf_high,color='grey', alpha = 0.25, label='95% CI')
# plt.plot(x, rbf_low, '--', c='grey', alpha = 0.2)
# plt.plot(x, rbf_high, '--',label = '95% CI', c='grey', alpha = 0.2)
plt.legend( frameon=False)
# plt.xlim((-0.1,1.1))
plt.xlabel('x')
plt.ylabel('y')
plt.title('95% Confidence Interval for $\hat{f}_{rbf}(x)$')
plt.savefig('A3c-2_filled.png')
plt.show()
######################################################
# d
n = 300
x = np.random.uniform(0, 1, size=(n,1))
# x = np.sort(x,axis =  0)
e = np.random.normal(0,1,size=(n,1))
# f_star = 4*np.sin(np.pi * x)*np.cos(6* np.pi * x**2)
y = true_f(x) + e 

plt.scatter(x, y)

min_err = sys.float_info.max
for lam in np.arange(0.0001,1,0.005):
    # min_err.append( LOO(x, y, lam,param=0))
    for d in np.arange(1,40,2):
        err = KFold(x, y, lam, param=d, k=10)
        if err<min_err:
            min_err = err
            min_lam = lam
            min_d = d

print(min_err) 
print(min_lam)
print(min_d)
# min_d=35
K_poly = k_poly(x, x, min_d)
a_poly = fit(y, K_poly, min_lam)
K_poly_fine =  k_poly(x_fine.reshape(x_fine.size, 1), x, min_d)
y_poly = predict(K_poly_fine, a_poly)


plt.plot(x, y, '.',c='black')
plt.plot(x_fine, true_f(x_fine), label = 'True $f(x)$')
plt.plot(x_fine,y_poly, label = '$\hat{f}_{poly}(x)$')
plt.legend( frameon=False)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Kernel d = '+str(min_d)+', $\lambda$ = '+str(round(min_lam,4)))
# plt.ylim((-10,10))

plt.savefig('A3d-1_new.png')
plt.show()

#####################################

min_err = sys.float_info.max
for lam in np.arange(0.001,0.5,0.005):
    # min_err.append( LOO(x, y, lam,param=0))
    for g in np.arange(0,500,5):
        err = KFold(x, y, lam, param=g, is_poly=False)
        if err < min_err:
            min_err = err
            min_lam_rbf = lam
            min_g = g

print(min_err) 
print(min_lam_rbf)
print(min_g)

# lam = 0.001, gamma = 26 

K_rbf= k_rbf(x, x, min_g)
a_rbf = fit(y, K_rbf, min_lam_rbf)
# y_rbf = predict(K_rbf, a_rbf)
K_rbf_fine =  k_rbf(x_fine.reshape(x_fine.size, 1), x, min_g)
y_rbf = predict(K_rbf_fine, a_rbf)

# b


plt.plot(x_fine, y_rbf , label =  '$\hat{f}_{rbf}(x)$')
plt.plot(x, y, '.',c='black')
plt.plot(x_fine, true_f(x_fine), label = 'True $f(x)$')
plt.legend( frameon=False)
plt.xlabel('x')
plt.ylabel('y')
plt.title('RBF Kernel $\gamma$ = '+str(min_g)+', $\lambda$ = '+str(round(min_lam_rbf,4)))

plt.savefig('A3d-2_new.png')
plt.show()

##########################################


B = 300
n = 300

y_poly_list=[]
y_rbf_list=[]

for i in range(B):
    sample_index = np.random.choice(np.arange(n), n)
    train_x = x[sample_index]
    train_y = y[sample_index]

    K_train_poly = k_poly(train_x, train_x, min_d)
    a_poly = fit(train_y, K_train_poly, min_lam)
    K_poly = k_poly(x_fine.reshape(x_fine.size, 1), train_x,  min_d)
    y_poly_list.append( predict(K_poly, a_poly).reshape(x_fine.size) )


    K_train_rbf= k_rbf(train_x, train_x, min_g)
    a_rbf = fit(train_y, K_train_rbf, min_lam_rbf)
    K_rbf =  k_rbf(x_fine.reshape(x_fine.size, 1), train_x,  min_g)
    y_rbf_list.append( predict(K_rbf, a_rbf).reshape(x_fine.size))

poly_low = np.percentile(y_poly_list, 5, axis=0)
poly_high = np.percentile(y_poly_list, 95, axis=0)

rbf_low = np.percentile(y_rbf_list, 5, axis=0)
rbf_high = np.percentile(y_rbf_list, 95, axis=0)


plt.plot(x, y, '.', c='black')
plt.plot(x_fine, true_f(x_fine), label = 'True $f(x)$')
plt.plot(x_fine,y_poly, label = '$\hat{f}_{poly}(x)$')
plt.fill_between(x_fine, poly_low, poly_high,color='grey', alpha = 0.25, label='95% CI')
# plt.plot(x, poly_low,'--', c='grey')
# plt.plot(x, poly_high, '--',label = '95% CI', c='grey')
plt.legend( frameon=False)
plt.xlabel('x')
plt.ylabel('y')
plt.title('95% Confidence Interval for $\hat{f}_{poly}(x)$')
# plt.xlim((-0.1,1.1))
plt.ylim((-8,8))
plt.savefig('A3d-1_filled.png')
plt.show()

plt.plot(x, y, '.', c='black')
plt.plot(x_fine, true_f(x_fine), label = 'True $f(x)$')
plt.plot(x_fine, y_rbf , label =  '$\hat{f}_{rbf}(x)$')
plt.fill_between(x_fine, rbf_low, rbf_high,color='grey', alpha = 0.25, label='95% CI')
# plt.plot(x, rbf_low, '--', c='grey', alpha = 0.2)
# plt.plot(x, rbf_high, '--',label = '95% CI', c='grey', alpha = 0.2)
plt.legend( frameon=False)
# plt.xlim((-0.1,1.1))
plt.xlabel('x')
plt.ylabel('y')
plt.title('95% Confidence Interval for $\hat{f}_{rbf}(x)$')
plt.savefig('A3d-2_filled.png')
plt.show()


##############################################################
# e

m = 1000
xm = np.random.uniform(0, 1, size=(m,1))
xm = np.sort(xm,axis =  0)
em = np.random.normal(0,1,size=(m,1))
# f_star_m = 4*np.sin(np.pi * xm)*np.cos(6* np.pi * xm**2)
ym = true_f(xm) + em 
# plt.scatter(x, y)

K_poly = k_poly(x, x, min_d)
a_poly = fit(y, K_poly, min_lam)
K_rbf= k_rbf(x, x, min_g)
a_rbf = fit(y, K_rbf, min_lam_rbf)

E_list = []
B = 300
for i in range(B):
    sample_index = np.random.choice(np.arange(m), m)
    sample_x = xm[sample_index]
    sample_y = ym[sample_index]

    K_poly_pred = k_poly(sample_x, x, min_d)
    y_poly = predict(K_poly_pred, a_poly)

    K_rbf_pred = k_rbf(sample_x, x, min_g)
    y_rbf = predict(K_rbf_pred, a_rbf)

    E_list.append(((sample_y - y_poly)**2 - (sample_y - y_rbf)**2).mean())

low = np.percentile(E_list, 5, axis=0)
high = np.percentile(E_list, 95, axis=0)

print(low, high)
# -0.056619967819317905 0.0045735956864309075
