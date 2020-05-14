import numpy as np
import multiprocessing
from numpy.polynomial import polynomial as P
import time


'''
Downsides include increased need for Thikonov filter. 
'''

def polyfit_deriv_multiprocess(stack, defvals):
    stack = np.array(stack)
    dim_y, dim_x = np.shape(stack[0])
    derivatives = np.zeros(np.shape(stack[0]))

    pool = multiprocessing.Pool()
    results = [pool.apply_async(run_polyfit, args=(i,defvals, stack[:,i],2,)) for i in range(dim_y)]
    for p in results:
        i, val = p.get()
        derivatives[i] = val
    return derivatives


def run_polyfit(i,x, y, deg):
    print(np.shape(P.polyfit(x, y, deg)[1]))
    return(i, P.polyfit(x, y, deg)[1])

def polyfit_deriv(stack, defvals):
    stack = np.array(stack)
    dim_y, dim_x = np.shape(stack[0])
    derivatives = np.zeros(np.shape(stack[0]))
    starttime = time.time()
    print('00.00%')
    for i in range(dim_y):
        if time.time() - starttime >= 5:
            print('{:.2f}%'.format(i/dim_y*100))
            starttime = time.time()
        
        unf_d = P.polyfit(defvals,stack[:,i],2)
        derivatives[i] = unf_d[1]
    print('100.0%')
    return derivatives