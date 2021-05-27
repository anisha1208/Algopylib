
def fact(n : int) -> int:
    val : int = 1
    for i in range(2, n+1):
        val *= i
    return val

def nCr(n : int, r : int) -> int:
    '''
    Returns the Binomial Coefficient.

    Parameters:
        n (int): 0 <= n .

    Returns:
        (int) : return nCr(n, r) values and 1 for -ve values.
    '''
    return fact(n)//(fact(n-r)*fact(r))

def nPr(n : int, r : int) -> int:
    return  fact(n)//fact(n-r)


    
