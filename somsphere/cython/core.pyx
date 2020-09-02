#cython: language_level=3
import numpy

def get_best_cell(inputs, importance, n_pix, weights, return_vals=1):
    """
    Return the closest cell to the input object
    It can return more than one value if needed
    """
    activations = numpy.sum(numpy.transpose([importance]) * (
            numpy.transpose(numpy.tile(inputs, (n_pix, 1))) - weights) ** 2, axis=0)

    return numpy.argmin(activations) if return_vals == 1 else numpy.argsort(activations)[0:return_vals], activations
