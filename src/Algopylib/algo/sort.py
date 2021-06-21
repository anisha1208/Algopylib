from typing import List, Union
from random import randint

def bubble_sort(arr : List , simulation : bool = False) -> List:
    """Sorts A List using bubble sort algorithm
    https://en.wikipedia.org/wiki/Bubble_sort
    Worst-case performance: O(N^2)

    Parameters:
        arr(List) : Unsorted List 
        simulation(bool) : to enable simulation (default argument is False) 

    Returns:
        arr(List) : Returns sorted List

    """

    def swap(i : int, j : int) -> None:
        """Swaps two element of List 

        Parameters:
            i(int) : index of first element
            j(int) : index of second element

        Returns:
            None : Function returns nothing

        """
        arr[i], arr[j] = arr[j], arr[i]

    n : int = len(arr)
    swapped : bool = True
    
    iteration : int = 0
    if simulation:
        print("iteration",iteration,":",*arr)
    x : int = -1
    while swapped:
        swapped = False
        x = x + 1
        for i in range(1, n-x):
            if arr[i - 1] > arr[i]:
                swap(i - 1, i)
                swapped = True
                if simulation:
                    iteration = iteration + 1
                    print("iteration",iteration,":",*arr)
                    
    return arr

def insertion_sort(arr : List , simulation : bool = False) -> List:
    """ Insertion Sort
    Complexity: O(n^2)
    1: Iterate from arr[1] to arr[n] over the array. 
    2: Compare the current element (key) to its predecessor. 
    3: If the key element is smaller than its predecessor, compare it to the elements before. Move the greater elements one position up to make space for the swapped element.
    """
    
    iteration : int = 0
    if simulation:
        print("iteration", iteration, ":", *arr)
        
    for i in range(len(arr)):
        cursor : Union[int, float, complex, str] = arr[i]
        pos : int = i
        """ Move elements of arr[0..i-1], that are greater than key, to one position ahead of their current position"""
        
        while pos > 0 and arr[pos - 1] > cursor:
            """ Swap the number down the list"""
            arr[pos] = arr[pos - 1]
            pos = pos - 1
        """ Break and do the final swap"""
        arr[pos] = cursor
        
        if simulation:
                iteration = iteration + 1
                print("iteration",iteration,":",*arr)

    return arr

def merge_sub(a : List[int], b : List[int]) -> List[int]:
    """A subroutine of merge sort 
    Worst-case performance: O(N)

    Parameters:
        a(List) : Unsorted List  
        b(List) : Unsorted List

    Returns:
        c(List) : Two pointer sorted List

    """
    n = len(a)
    m = len(b)
    i, j, k = 0, 0, 0
    c = [ 0 for i in range(n+m)]

    while(i < n or j < m):
        if(j == m or (i < n and a[i] < b[j])):
            c[k] = a[i]
            k += 1
            i += 1
        else:
            c[k] = b[j]
            k += 1
            j += 1
    
    return c

def merge_sort(a : List[int]) -> List[int]:
    """Sorts A List using merge sort algorithm
    https://en.wikipedia.org/wiki/Merge_sort
    Worst-case performance: O(Nlog(N))

    Parameters:
        a(List) : Unsorted List  

    Returns:
        a(List) : Returns sorted List

    """
    n = len(a)

    if(n < 2):
        return a
    
    b = [ 0 for i in range(n//2)]
    c = [ 0 for i in range(n - n//2)]

    for i in range(n):
        if (i < n//2):
            b[i] = a[i]
        else:
            c[i-(n//2)] = a[i]
    
    return merge_sub(merge_sort(b), merge_sort(c)) 

def insert_heap(h : List[int], x : int) -> None:
    """Inserting an element to heap 
    https://en.wikipedia.org/wiki/Heap_(data_structure)
    Worst-case performance: O(log(N))

    Parameters:
        h(List) : List that represents heap
        x(int) : element to be inserted  

    """
    h.append(x)
    n = len(h)
    i = n-1

    while(i > 0 and h[i] < h[(i-1)//2]):
        h[i], h[(i-1)//2] = h[(i-1)//2], h[i]
        i = (i-1)//2

def remove_min(h : List[int]) -> int:
    """Removing an element from heap
    https://en.wikipedia.org/wiki/Heap_(data_structure)
    Worst-case performance: O(log(N))

    Parameters:
        h(List) : List that represents heap  

    Returns:
        last_element(int) : Returns the smallest element of the heap

    """
    n = len(h)
    last_element = h[0]

    h[0], h[n-1] = h[n-1], h[0]
    h.pop()
    n = len(h)

    i, j = 0, 0

    while(2*i+1 < n):
        j = 2*i+1
        if((2*i+2 < n) and h[2*i+2] < h[j]):
            j = 2*i+2
        if(h[i] <= h[j]):
            break
        h[i], h[j] = h[j], h[i]
        i = j
    
    return last_element

def heap_sort(a : List[int]) -> None:
    """Sorts A List using heap sort algorithm
    https://en.wikipedia.org/wiki/Heapsort
    Worst-case performance: O(Nlog(N))

    Parameters:
        a(List) : Unsorted List  

    """
    temp = []

    for i in a:
        insert_heap(temp, i)
    
    for i in range(len(a)):
        a[i] = remove_min(temp)

def quick_sort(a : List[int], l : int, r : int) -> None:
    """Sorts A List using quick sort algorithm
    https://en.wikipedia.org/wiki/Quicksort
    Worst-case performance: O(N^2)
    Average performance: O(Nlog(N))

    Parameters:
        arr(List) : Unsorted List
        l(int) : left index (present in List) 
        r(int) : Right index (not present in List) 

    """
    if(r - l <= 1):
        return None
    
    idx = randint(l, r-1)

    x = a[idx]
    m = l

    for i in range(l, r, 1):
        if (a[i] < x):
            a[i], a[m] = a[m], a[i]
            m += 1

    quick_sort(a, l, m)
    quick_sort(a, m, r)