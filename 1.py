import sys,os,time
import numpy as np
sys.path.append(os.pardir)
from mnist import load_mnist
class ksmnetwork:
    def __init__(self, input_dim, conv_param, hidden_size, output_size):
        filter_num=conv_param["filter_num"]
        filter_size=conv_param["filter_size"]
        filter_pad=conv_param["filter_pad"]
        filter_stride=conv_param["filter_stride"]
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size+ 2*filter_pad)/filter_stride+1
        pool_output_size = int(filter_num*(conv_output_size/2)*(conv_output_size/2))
        W1=np.random.randn(filter_num,input_dim[0],filter_size,filter_size)
        W2=np.random.randn(pool_output_size,hidden_size)/np.sqrt(pool_output_size)
        W3=np.random.randn(hidden_size,output_size)/np.sqrt(hidden_size)
        b1=np.zeros(filter_num)
        b2=np.zeros(hidden_size)
        b3=np.zeros(output_size)
        self.layers=[
            Convolution(W1,b1,conv_param["filter_stride"],conv_param["filter_pad"]),
            Relu(),
            Pooling(pool_h=2,pool_w=2,stride=2),
            Affine(W2,b2),
            Relu(),
            Affine(W3,b3),
        ]
        self.last_layer = SoftmaxWithLoss()
        self.grads=[]
        self.params=[]
        for layer in self.layers:
            self.grads += layer.grads
            self.params += layer.params
       

    def predict(self,x):
        x= self.layers[0].forward(x)
        x= self.layers[1].forward(x)
        x= self.layers[2].forward(x)
        x= self.layers[3].forward(x)
        x= self.layers[4].forward(x)
        x= self.layers[5].forward(x)
        return x
    
    def forward(self,x,t):
        score = self.predict(x)
        loss = self.last_layer.forward(score, t)
        return loss

    def backward(self,dout=1):
        dout = self.last_layer.backward(dout)
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def accuracy(self,x,t,batch_size=100):
        if t.ndim != 1 : 
            t = np.argmax(t, axis=1)
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.params=[W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.stride = stride
        self.pad = pad
    
    def forward(self,x):
        FN, C, FH, FW = self.params[0].shape 
            # フィルターの(バッチ数、チャンネル、高さ、横幅)
        N, C ,H ,W = x.shape
            # 入力層の(バッチ数、チャンネル、高さ、横幅)
        out_h = int(1+(H+2*self.pad - FH)/self.stride)
        out_w = int(1+(W+2*self.pad - FW)/self.stride)
        #出力サイズを求める

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.params[0].reshape(FN, -1).T
        out = np.dot(col,col_W)+self.params[1]
        out = out.reshape(N, out_h,out_w, -1).transpose(0,3,1,2)
        #Transposeは軸の順番を指定 出力を整形
        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.params[0].shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        self.grads[0][...] = self.dW
        self.grads[1][...] = self.db

        return dx

class Affine:
    def __init__(self,W,b):
        self.params=[W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.original_x_shape = None

    
    def forward(self, x):
        w, b = self.params
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        out = np.dot(x,w)+b
        self.x = x
        return out
    def backward(self,dout):
        w,b = self.params
        dx = np.dot(dout,w.T)
        dw = np.dot(self.x.T,dout)
        db = np.sum(dout,axis=0)
        self.grads[0][...]=dw
        self.grads[1][...]=db
        dx = dx.reshape(*self.original_x_shape)
        return dx 

class Sigmoid:
    def __init__(self):
        self.params ,self.grads= [],[]
    
    def forward(self, x):
        out = 1 / ( 1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        return dout * self.out * (1-self.out)

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size
        return dx

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

def cross_entropy_error(y,t):
    delta= 1e-7
    return - np.sum(t* np.log(y+ delta))

class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self,lr = 0.01, momentum = 0.9):
        self.lr=lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for val in params.items():
                self.v[key]=np.zeros_like(val)
        for key in params.keys():
            self.v[key]= self.momentum * self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class AdaGrad:
    '''
    AdaGrad
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)

class Dropout:
    def __init__(self,dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
        self.mask = None
    def forward(self, x,train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self,dout):
        return  dout *self.mask
    
class Relu:
    def __init__(self):
        self.mask=None
        self.params ,self.grads= [],[]
    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out 
    def backward(self, dout):
        dout[self.mask]=0
        dx = dout
        return dx
        
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Pooling:
    def __init__(self, pool_h, pool_w,stride=2,pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride = stride
        self.pad = pad
        self.params ,self.grads= [],[]
        
    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1+(H-self.pool_h)/self.stride)
        out_w = int(1+(W-self.pool_w)/self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
        self.x = x
        self.arg_max = arg_max
        return out 
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class Trainer:
    def __init__(self,model, x_train, t_train, x_test, t_test,epochs=20, optimizer_param=0.01, evaluate_sample_num_per_epoch=None, verbose=True):
        self.model = model
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
    
        self.x_train=x_train
        self.x_test=x_test
        self.t_train=t_train
        self.t_test=t_test
        self_mini_batch_size=100
        self.optimizer_palam=optimizer_param
        self.optimizer = AdaGrad(optimizer_param)

    def fit(self, x_train,t_train, x_test,t_test, max_epoch=100, batch_size=32, max_grad=None, eval_interval=2):
        train_data_size = len(x_train)
        test_data_size = len(x_test)
        max_iters = train_data_size // batch_size
        mini_batch_size = test_data_size // max_iters
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer

        start_time = time.time()
        for epoch in range(max_epoch):
            # シャッフル
            idx = np.random.permutation(np.arange(train_data_size))
            x_train = x_train[idx]
            t_train = t_train[idx]
            idxt = np.random.permutation(np.arange(test_data_size))
            x_test=x_test[idxt]
            t_test=t_test[idxt]

            for iters in range(max_iters):
                batch_x = x_train[iters*batch_size:(iters+1)*batch_size]
                batch_t = t_train[iters*batch_size:(iters+1)*batch_size]
                batch_x_t = x_test[iters*mini_batch_size:(iters+1)*mini_batch_size]
                batch_t_t = t_test[iters*mini_batch_size:(iters+1)*mini_batch_size]

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)

                
                # 評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = model.forward(batch_x_t, batch_t_t) / batch_size
                    elapsed_time = time.time() - start_time
                    print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))

            self.current_epoch += 1

def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate



max_epoch = 1
batch_size = 30
hidden_size = 100
learning_rate = 1.0
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True,flatten=False)
model = ksmnetwork(input_dim=[1,28,28],conv_param={"filter_num":int(30),"filter_size":int(5),"filter_pad":int(0),"filter_stride":int(1)} ,
                  hidden_size=100, output_size=10)
trainer = Trainer(model, x_train, t_train, x_test, t_test,
                 epochs=10, optimizer_param=0.01, 
                 evaluate_sample_num_per_epoch=None, verbose=True)
trainer.fit(x_train,t_train,x_test,t_test,max_epoch=300,batch_size=1000,eval_interval=10)



