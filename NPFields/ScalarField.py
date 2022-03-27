import numpy as np
def partial_derivative(scalarfield:np.ndarray,axis:int)->np.ndarray:
    """
    Partial derivative of a field along an axis
    Returns a vector field of shape (x1,x2,x3,...,xn) for
    a field of shape (x1,x2,x3,...,xn)
    """
    f = np.zeros(scalarfield.shape)
    x0 = tuple([slice(None) if dimension!=axis else slice(1,-1) for dimension in range(len(scalarfield.shape))])
    x1 = tuple([slice(None) if dimension!=axis else slice(2,None) for dimension in range(len(scalarfield.shape))])
    x_1 = tuple([slice(None) if dimension!=axis else slice(0,-2) for dimension in range(len(scalarfield.shape))])
    f[x0] = ((scalarfield[x1]-scalarfield[x_1]))/2
    return f
def gradient(scalarfield:np.ndarray)->np.ndarray:
    """
    Gradient of a scalar field.
    Returns a vector field of shape (x1,x2,x3,...,xn,n) for
    a field of shape (x1,x2,x3,...,xn)
    """
    grad = np.zeros(tuple(list(scalarfield.shape)+[len(scalarfield.shape)]))
    alldims = [slice(None) for _ in scalarfield.shape]
    for dimension in range(len(scalarfield.shape)):
        grad[tuple(alldims+[dimension])] = partial_derivative(scalarfield,dimension)
    return grad