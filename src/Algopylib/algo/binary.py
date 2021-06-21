def binary_sum(x : int ,y : int) -> str : 
    """
    Given two binary number
    return their sum .

    For example,
    x = 11
    y = 111
    Return result as "1000".
    
    Parameters:
        x(int) : binary number
        y(int) : binary number

    Returns:
        s(str) : sum of a and b in string 
    
    """
    a : str = str(x) 
    b : str = str(y)
    s : str = ""
    c : int = 0
    i : int = len(a)-1
    j : int = len(b)-1
    zero : int = ord('0')

    while (i >= 0 or j >= 0 or c == 1):
        if (i >= 0):
            c += ord(a[i]) - zero
            i -= 1
        if (j >= 0):
            c += ord(b[j]) - zero
            j -= 1
        s = chr(c % 2 + zero) + s
        c //= 2 
        
    return s