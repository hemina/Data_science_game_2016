"""model

Author: Mina HE

modified from the version MNIST
"""

import os
import sys
import time
import pylab
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
#import theano.config as config
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
import pandas as pd
import logging
import numpy as np  # Make sure that numpy is imported
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation

class LeNetConvPoolLayer(object):

    """Pool Layer of a convolutional network """



    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(5, 5)):

        """

        Allocate a LeNetConvPoolLayer with shared variable internal parameters.



        :type rng: numpy.random.RandomState

        :param rng: a random number generator used to initialize weights



        :type input: theano.tensor.dtensor4

        :param input: symbolic image tensor, of shape image_shape



        :type filter_shape: tuple or list of length 4

        :param filter_shape: (number of filters, num input feature maps,

                              filter height, filter width)



        :type image_shape: tuple or list of length 4

        :param image_shape: (batch size, num input feature maps,

                             image height, image width)



        :type poolsize: tuple or list of length 2

        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        """



        assert image_shape[1] == filter_shape[1]

        self.input = input



        # there are "num input feature maps * filter height * filter width"

        # inputs to each hidden unit

        fan_in = np.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:

        # "num output feature maps * filter height * filter width" /

        #   pooling size

        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /

                   np.prod(poolsize))

        # initialize weights with random weights

        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(

            np.asarray(

                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),

                dtype=theano.config.floatX

            ),

            borrow=True

        )



        # the bias is a 1D tensor -- one bias per output feature map

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self.b = theano.shared(value=b_values, borrow=True)



        # convolve input feature maps with filters

        conv_out = conv.conv2d(

            input=input,

            filters=self.W,

            filter_shape=filter_shape,

            image_shape=image_shape

        )



        # downsample each feature map individually, using maxpooling

        pooled_out = downsample.max_pool_2d(

            input=conv_out,

            ds=poolsize,

            ignore_border=True

        )



        # add the bias term. Since the bias is a vector (1D array), we first

        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will

        # thus be broadcasted across mini-batches and feature map

        # width & height

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))



	# L1 norm ; one regularization option is to enforce L1 norm to

        # be small

        self.L1 = 0

        self.L1 = abs(self.W).sum()



        # square of L2 norm ; one regularization option is to enforce

        # square of L2 norm to be small

        self.L2_sqr = 0

        self.L2_sqr = (self.W ** 2).sum()



        # store parameters of this layer

        self.params = [self.W, self.b]





def evaluate_lenet5(learning_rate=0.01, n_epochs=400,
                    trainset='train.csv',
                    nkerns=[20, 30, 40], batch_size=100):
    """ 
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', trainset), header=0, \
                    delimiter=";", quoting=3)
    rng = np.random.RandomState(23455)
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(train['data'], train['label'], test_size=0.3, random_state=0)


    feature_r = 200
    feature_c = 200
    #theano variable
    train_set_x = theano.shared(value=x_train, borrow=True) 
    train_set_y = theano.shared(y_train) 

    valid_set_x = theano.shared(value=x_test, borrow=True)
    valid_set_y = theano.shared(y_test) 
    # compute number of minibatches for training, validation and testing
    test_set_x = train_set_x
    test_set_y = train_set_y

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    n_test_batches = n_train_batches
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    # start-snippet-1

    #x = T.matrix('x')   # the data is presented as rasterized images

    x = T.tensor4('x') 
    y = T.ivector('y')  # the labels are presented as 1D vector of

    ######################

    # BUILD ACTUAL MODEL #

    ######################

    print '... building the model'



    #layer0_input = x.reshape((batch_size, 1, 100, 300))

    layer0_input = x
    # Construct the first convolutional pooling layer:

    # filtering reduces the image size to (50-11+1 , 100-11+1) = (40, 90)

    # maxpooling reduces this further to (40/2, 90/2) = (20, 45)

    # 4D output tensor is thus of shape (batch_size, nkerns[0], 20, 45)

    layer0 = LeNetConvPoolLayer(

        rng,

        input=layer0_input,

        image_shape=(batch_size, 1, feature_r, feature_c),

        filter_shape=(nkerns[0], 1, 11, 11),

        poolsize=(2, 2)

    )



    l1feature_r=feature_r-11+1

    l1feature_c=feature_c-11+1

    l1feature_r=l1feature_r/2

    l1feature_c=l1feature_c/2

    # Construct the second convolutional pooling layer

    # filtering reduces the image size to (20-5+1, 45-5+1) = (16, 41)

    # maxpooling reduces this further to (16/2, 41/2) = (8, 20)

    # 4D output tensor is thus of shape (batch_size, nkerns[1], 8, 20)

    layer1 = LeNetConvPoolLayer(

        rng,

        input=layer0.output,

        image_shape=(batch_size, nkerns[0], l1feature_r, l1feature_c),

        filter_shape=(nkerns[1], nkerns[0], 5, 5),

        poolsize=(2, 2)

    )



    # Construct the third convolutional pooling layer

    # filtering reduces the image size to (8-3+1, 20-3+1) = (6, 18)

    # maxpooling reduces this further to (6/2, 18/2) = (3, 9)

    # 4D output tensor is thus of shape (batch_size, nkerns[2], 3, 9)

    l2feature_r=l1feature_r-5+1

    l2feature_c=l1feature_c-5+1

    l2feature_r=l2feature_r/2

    l2feature_c=l2feature_c/2



    layer2 = LeNetConvPoolLayer(

        rng,

        input=layer1.output,

        image_shape=(batch_size, nkerns[1], l2feature_r, l2feature_c),

        filter_shape=(nkerns[2], nkerns[1], 3, 3),

        poolsize=(2, 2)

    )
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 3 * 9),
    # or (100, 40 * 3 * 9) = (100, 1080) with the default values.

    layer3_input = layer2.output.flatten(2)

    l3feature_r=l2feature_r-3+1

    l3feature_c=l2feature_c-3+1

    l3feature_r=l3feature_r/2

    l3feature_c=l3feature_c/2

    # construct a fully-connected sigmoidal layer

    layer3 = HiddenLayer(

        rng,

        input=layer3_input,

        n_in=nkerns[2] * l3feature_r * l3feature_c,

        n_out=500,

        activation=T.tanh

    )



    # classify the values of the fully-connected sigmoidal layer



    layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=2)



#    # the cost we minimize during training is the NLL of the model

#    cost = layer4.negative_log_likelihood(y)



    # create a function to compute the mistakes that are made by the model

    test_model = theano.function(

        [index],

        layer4.errors(y),

        givens={

            x: test_set_x[index * batch_size: (index + 1) * batch_size],

            y: test_set_y[index * batch_size: (index + 1) * batch_size]

        }

    )



    validate_model = theano.function(

        [index],

        layer4.errors(y),

        givens={

            x: valid_set_x[index * batch_size: (index + 1) * batch_size],

            y: valid_set_y[index * batch_size: (index + 1) * batch_size]

        }

    )

    

    # create a list of all model parameters to be fit by gradient descent

    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params



#    prm=[]

#    for i in xrange(len(params)):

#	if i%2==0:

#	    p=params[i].get_value()

#	    pp=[item for sublist in p for item in sublist]

#	    prm.extend(pp)



#    len_prm=len(prm)

#    prm_list=[]

#    for sublist in prm:

#	if sublist.size==1:

#	    prm_list.append(sublist)

#	else:

#	    tmp=[itm for subli in sublist for itm in subli]

#	    prm_list.extend(tmp)

#    regular=0

#    regular=layer4.L2_sqr + layer3.L2_sqr + layer2.L2_sqr + layer1.L2_sqr + layer0.L2_sqr 

  

    lamda=0.0061



    # the cost we minimize during training is the NLL of the model

#    cost = layer4.negative_log_likelihood(y)



    # the cost with regularization

#    cost=lamda*regular+layer4.negative_log_likelihood(y)



    # create a list of gradients for all model parameters

#    grads = T.grad(cost, params)



#    for param_i in zip(params, grads):

#        regular=regular+param_i*param_i

#    regular=sum(it*it for it in prm_list)



    # the cost with regularization

    cost=(layer4.negative_log_likelihood(y)+(layer4.L2_sqr + layer3.L2_sqr + layer2.L2_sqr + layer1.L2_sqr + layer0.L2_sqr)*lamda) 

    

    # create a list of gradients for all model parameters

    grads = T.grad(cost, params)  



    # train_model is a function that updates the model parameters by

    # SGD Since this model has many parameters, it would be tedious to

    # manually create an update rule for each model parameter. We thus

    # create the updates list by automatically looping over all

    # (params[i], grads[i]) pairs.

    updates = [

        (param_i, param_i - learning_rate * grad_i)

        for param_i, grad_i in zip(params, grads)

    ]



    train_model = theano.function(

        [index],

        cost,

        updates=updates,

        givens={

            x: train_set_x[index * batch_size: (index + 1) * batch_size],

            y: train_set_y[index * batch_size: (index + 1) * batch_size]

        }

    )



    print 'end'

    # end-snippet-1





##    print '... debug'

    eval_layer0 = theano.function(

            [index],

            layer0.output,

            givens={

                x: train_set_x[index * batch_size: (index + 1) * batch_size]

            }

    )



    eval_layer1 = theano.function(

            [index],

            layer1.output,

            givens={

                x: train_set_x[index * batch_size: (index + 1) * batch_size]

            }

    )



    eval_layer2 = theano.function(

            [index],

            layer2.output,

            givens={

                x: train_set_x[index * batch_size: (index + 1) * batch_size]

            }

    )



    eval_layer2_flatten = theano.function(

            [index],

            layer2.output.flatten(2),

            givens={

                x: train_set_x[index * batch_size: (index + 1) * batch_size]

            }

    )

    



    eval_layer3 = theano.function(

            [index],

            layer3.output,

            givens={

                x: train_set_x[index * batch_size: (index + 1) * batch_size]

            }

    )

    



    ###############

    # TRAIN MODEL #

    ###############

    print '... training'

#    print 'n_valid_batches='

#    print n_valid_batches

##    # early-stopping parameters

    patience = 10000  # look as this many examples regardless

    patience_increase = 2  # wait this much longer when a new best is

                           # found

    improvement_threshold = 0.995  # a relative improvement of this much is

                                   # considered significant

    validation_frequency = min(n_train_batches, patience / 2)

##                                  # go through this many

##                                  # minibatche before checking the network

##                                  # on the validation set; in this case we

##                                  # check every epoch

##

    best_validation_loss = np.inf

    best_iter = 0

    test_score = 0.

    start_time = time.clock()

##

    epoch = 0

    done_looping = False

##

    while (epoch < n_epochs) and (not done_looping):

        epoch = epoch + 1

        for minibatch_index in xrange(n_train_batches):



            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 10 == 0:

                print 'training @ iter = ', iter

##            eval_0=eval_layer0(minibatch_index)

##            print eval_0.shape

##            eval_1=eval_layer1(minibatch_index)

##            print eval_1.shape

##            eval_2=eval_layer2(minibatch_index)

##            print eval_2.shape

##            eval_2_flatten=eval_layer2_flatten(minibatch_index)

##            print eval_2_flatten.shape

##            eval_3=eval_layer3(minibatch_index)

##            print eval_3.shape

            

            cost_ij = train_model(minibatch_index)

            print cost_ij

            

            #if (iter + 1) % validation_frequency == 0:

            if (iter + 1) % 50 == 0:



                # compute zero-one loss on validation set

                validation_losses = [np.mean(validate_model(i)) for i

                                     in xrange(n_valid_batches)]

##	        print 'size of validate_error:' 

##              print len(validate_error)

#		print 'validation_losses='

#                print validation_losses



                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %

                      (epoch, minibatch_index + 1, n_train_batches,

                       this_validation_loss * 100.))



                # if we got the best validation score until now

                if this_validation_loss < best_validation_loss:



                    #improve patience if loss improvement is good enough

                    if this_validation_loss < best_validation_loss *  improvement_threshold:
                       patience = max(patience, iter * patience_increase)
                    # save best validation score and iteration number

                    best_validation_loss = this_validation_loss

                    best_iter = iter



##                    # test it on the test set

##                    test_losses = [

##                        test_model(i)

##                        for i in xrange(n_test_batches)

##                    ]

##                    test_score = np.mean(test_losses)

##                    print(('     epoch %i, minibatch %i/%i, test error of '

##                           'best model %f %%') %

##                          (epoch, minibatch_index + 1, n_train_batches,

##                           test_score * 100.))





                    

##

            if patience <= iter:

		print	"patience <= iter"	

                done_looping = True

                break

##

    validate_error = []

    for i in xrange(n_valid_batches):

#	print i

#	print 'validate_model(i)='

#	print validate_model(i)

        validate_error.extend(validate_model(i))

#	print len(validate_error)

    print 'validate_error='

    print np.mean(validate_error)



    train_set_error = []

    for i in xrange(n_train_batches):

#       print i

#       print 'validate_model(i)='

#       print validate_model(i)

        train_set_error.extend(test_model(i))

#        print len(train_set_error)

    print 'train_set_error='

    print np.mean(train_set_error)



    end_time = time.clock()

    print 'end_time='

    print end_time

    print('Optimization complete.')

    print('Best validation score of %f %% obtained at iteration %i, '

         %(best_validation_loss * 100., best_iter + 1.))

#    output = pd.DataFrame( data={"id":train["id"][20000:25000],"review":train["review"][20000:25000], "error":validate_error} )

#    output.to_csv( "cnn_BoW_error.csv", index=False, quoting=3 )
#    output.to_csv( "cnn_BoW_error.csv", index=False )
#    print "Wrote cnn_BoW_error.csv"

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
