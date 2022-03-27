import NPFields.ScalarField as ScalarField
import numpy as np
def partial_derivative(vectorfield:np.ndarray,axis:int,component:int)->np.ndarray:
    """
    d[Field_component]/d[axis]
    Returns a scalar field of shape (x1,x2,x3,...,xn)
    """
    return ScalarField.partial_derivative(
        vectorfield[tuple([slice(None) for _ in vectorfield.shape[:-1]]+[component])],
        axis)
def divergent(vectorfield:np.ndarray):
    """
    Returns the divergent of a vector field
    Returns a scalar field of shape (x1,x2,x3,...,xn)
    """
    field = np.zeros(vectorfield.shape[:-1])
    for dimension in range(len(field.shape)):
        field+=partial_derivative(vectorfield,dimension,dimension)
    return field