import numpy as np


def find_BMU(SOM,x):
    '''
    Return the (g,h) index of the BMU in the grid
    '''
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)
    
def update_weights(SOM, train_ex, learn_rate, radius_sq, 
                   BMU_coord, step=3):
    '''
    Update the weights of the SOM cells when given a single training example
    and the model parameters along with BMU coordinates as a tuple
    '''
    
    g, h = BMU_coord
    
    # If radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM[g,h,:] += learn_rate * (train_ex - SOM[g,h,:])
        return SOM
    
    # Change all cells in a small neighborhood of BMU
    for i in range(max(0, g-step), min(SOM.shape[0], g+step)):
        for j in range(max(0, h-step), min(SOM.shape[1], h+step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            SOM[i,j,:] += learn_rate * dist_func * (train_ex - SOM[i,j,:])   
            
    return SOM    

def train_SOM(SOM,\
              train_data,\
              learn_rate = .1,\
              radius_sq = 1, 
              lr_decay = .1,\
              radius_decay = .1,\
              epochs = 10,\
              rand=None):    

    '''
    Main routine for training an SOM. It requires an initialized SOM grid
    or a partially trained grid as parameter
    '''
    
    if rand == None:
        rand = np.random.RandomState(0)
    
    learn_rate_0 = learn_rate
    radius_0     = radius_sq
    
    for epoch in np.arange(0, epochs):
        train_data_epoch = rand.choice(train_data, size=len(train_data), replace=True)
                
        for train_ex in train_data_epoch:
            g, h = find_BMU(SOM, train_ex)
            SOM  = update_weights(SOM, train_ex, learn_rate, radius_sq, (g,h))
            
        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq  =     radius_0 * np.exp(-epoch * radius_decay)       
        
    return SOM