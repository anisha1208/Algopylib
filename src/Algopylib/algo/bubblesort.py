from typing import List

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
