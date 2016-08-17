__author__ = 'p2admin'
import scipy.io
import numpy as np
import theano
import theano.tensor as T
import random


force = scipy.io.loadmat('true.mat')
neurons = scipy.io.loadmat('hidstates5th_WB_recon_30neuron')
data1=neurons['xtr'].T
data2=np.asarray(force['skt']).reshape(-1).T
sgd_optimization()



class RBM_NN(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out),dtype=theano.config.floatX),name='W',borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
        #self.W=W
        #self.b=b
        self.y_pred = T.dot(input, self.W)+self.b
        self.params = [self.W, self.b]

    def difference(self,y):
        return T.mean(np.abs(T.neq(self.y_pred, y))) #[T.arange(y.shape[0],y)]



def load_data(data1, data2):
    #data1=neurons['xtr']
    #data2=force['skt']
    def shared_dataset(data1,data2,borrow=True):
        shared_x = theano.shared(np.asarray(data1, dtype=theano.config.floatX),borrow=True)
        shared_y = theano.shared(np.asarray(data2, dtype=theano.config.floatX),borrow=True)
        return shared_x, T.cast(shared_y,'int32')

    train_set_x, train_set_y = shared_dataset(data1,data2)
    rval = [(train_set_x, train_set_y)]
    return rval

def sgd_optimization(learning_rate=0.1, n_epochs=5, data1=data1, data2=data2, batch_size=10):
    datasets = load_data(data1, data2)
    train_set_x, train_set_y = datasets[0]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = RBM_NN(input=x, n_in=30 * 1, n_out=10)
    cost = classifier.difference(y)

    #test_model = theano.function(inputs=[index], outputs=cost,
    #                            givens={x:test_set_x[index*batch_size:(index+1)*batch_size],
    #                                   y:test_set_y[index*batch_size:(index+1)*batch_size]}
    #                            )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W-learning_rate*g_W),
               (classifier.b, classifier.b-learning_rate*g_b)]

    train_model = theano.function(inputs=[index],outputs=cost,updates=updates,
                                 givens={x:train_set_x[index*batch_size:(index+1)*batch_size],
                                         y:train_set_y[index*batch_size:(index+1)*batch_size]}
                                 )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')

    epoch=0
    while (epoch<n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost=train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            #print iter
            if (iter + 1) % 5 == 0:
                # compute zero-one loss on validation set
                losses = [train_model(i) for i in range(n_train_batches)]
                this_loss = np.mean(losses)

                print "Epoch {0}, Minibatch {1}/{2}, Test Error= {3}".format(epoch,minibatch_index+1,n_train_batches,
                                                                       this_loss)

