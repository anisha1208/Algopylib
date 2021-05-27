def insertion_sort(arr, simulation=False):
    """ Insertion Sort
        Complexity: O(n^2)
        1: Iterate from arr[1] to arr[n] over the array. 
        2: Compare the current element (key) to its predecessor. 
        3: If the key element is smaller than its predecessor, compare it to the elements before. Move the greater elements one position up to make space for the swapped element.
    """
    
    iteration = 0
    if simulation:
        print("iteration",iteration,":",*arr)
        
    for i in range(len(arr)):
        cursor = arr[i]
        pos = i
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        
        while pos > 0 and arr[pos - 1] > cursor:
            # Swap the number down the list
            arr[pos] = arr[pos - 1]
            pos = pos - 1
        # Break and do the final swap
        arr[pos] = cursor
        
        if simulation:
                iteration = iteration + 1
                print("iteration",iteration,":",*arr)

    return arr
