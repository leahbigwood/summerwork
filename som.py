import numpy as np


def distSq(SOM, x):
    return (np.square(SOM - x)).sum(axis=-1)

def find_BMU(SOM,x):
    '''
    Return the (g,h) index of the BMU in the grid
    '''
    _distSq = distSq(SOM, x)
    
    return  np.unravel_index(np.argmin(_distSq, axis=None), _distSq.shape)
    
def update_weights(SOM, zSOM, train_ex, learn_rate, radius_sq, 
                   BMU_coord, step=3):
    '''
    Update the weights of the SOM cells when given a single training example
    and the model parameters along with BMU coordinates as a tuple
    '''
    
    g, h        = BMU_coord
    
    _distSq     = distSq(zSOM, zSOM[g,h])
    dist_func   = np.exp(-np.sqrt(_distSq / radius_sq))
            
    return  learn_rate * dist_func[:,:,None] * (train_ex[None,:] - SOM)   

def train_SOM(SOM,\
              train_data,\
              nepochs = 10,\
              rand=None):    

    '''
    Main routine for training an SOM. It requires an initialized SOM grid
    or a partially trained grid as parameter
    '''
    
    if rand == None:
        rand = np.random.RandomState(0)
    
    nvec         = SOM.shape[0]
    dsom         = 1. / nvec
    
    zSOM         = dsom * np.array([[i, j] for i in range(nvec) for j in range(nvec)]).reshape(nvec,nvec,2)
    
    learn_rate   = learn_rate_0 = 1.
    radius_sq    = radius_sq_0  = 1.
    
    lr_decay     = radius_decay = 10. / nepochs
    
    rates = []
    radii = []
    
    for epoch in range(nepochs):
        draws            = rand.choice(np.arange(len(train_data)), size=len(train_data), replace=False)
        train_data_epoch = train_data[draws]
                
        for train_ex in train_data_epoch:
            g, h   = find_BMU(SOM, train_ex)
            SOM   += update_weights(SOM, zSOM, train_ex, learn_rate, radius_sq, (g,h))
            
        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq  =  radius_sq_0 * np.exp(-epoch * radius_decay)       
        
        rates.append(learn_rate)
        radii.append(np.sqrt(radius_sq))
        
    return nvec, zSOM, SOM, rates, radii, lr_decay