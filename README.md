![logo](https://raw.githubusercontent.com/still-n0thing/Algopylib/main/logo.png)

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![PyPI version](https://badge.fury.io/py/Algopylib.svg)](https://badge.fury.io/py/Algopylib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/algopylib)](https://pepy.tech/project/algopylib)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

> A basic Python library that has modules for math functions, algorithms, and puzzles.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Algopylib.

```bash
pip install Algopylib 
```

## Usage

```python
import Algopylib.algo.check as ck

ck.is_power_of_2(32) # returns True 
ck.is_even(3) # returns False 
```

## Package Structure

```bash
├── Algopylib
│   ├── algo
│   │   ├── __init__.py
│   │   ├── binary.py
│   │   ├── check.py
│   │   └── sort.py
│   ├── math
│   │   ├── __init__.py
│   │   ├── combinatorics.py
│   │   ├── gcd_lcm.py
│   │   ├── geometry.py
│   │   ├── matrix.py
│   │   ├── series.py
│   │   └── stat.py
│   └── puzzle
│       ├── __init__.py
│       ├── sudoku.py
│       └── tictaktoe.py
└── __init__.py
```

### Contents
## [**Algo**](https://github.com/still-n0thing/Algopylib/tree/main/src/Algopylib/algo)

#### Implementation of simple Algorithms.
   - [Binary](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/algo/binary.py)
      - Binary sum
   - [Check](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/algo/check.py)
      - Is power of 2
      - Is power of N
      - Is square
      - Is palindrome
      - Is subsequence
      - Is even
      - Is odd
      - Is prime
   - [Sort](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/algo/sort.py)
      - Bubble sort
      - Insertion sort
      - Merge sort
      - Heap sort
      - Quick sort

## [**Math**](https://github.com/still-n0thing/Algopylib/tree/main/src/Algopylib/math)

#### Implementation of useful Math functions.
   - [Combinatorics](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/math/combinatorics.py)
      - Factorial
      - nCr
      - nPr
   - [GCD LCM](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/math/gcd_lcm.py)
      - GCD
      - LCM
      - Trailing zeroes
      - GCD using bitwise operator
   - [Geometry](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/math/geometry.py)
      - Distance
      - Is collinear
      - Equation of line
      - Is inside circle
      - Area of triangle
   - [Matrix](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/math/matrix.py)
      - Determinant of 2x2 matrix
      - Determinant of 3x3 matrix
      - Inverse of 2x2 matrix
      - Inverse of 3x3 matrix
   - [Series](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/math/series.py)
      - Nth term of AP
      - Nth term of GP
      - Sum of AP 
      - Sum of GP
   - [Statistics](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/math/stat.py)
      - Mean
      - Float mean
      - Geometric mean
      - Harmonic mean
      - Median
      - Low and high median
      - Median-grouped
      - Mode 
      - Multimode
      - Quantiles
      - Population variance
      - Sample variance
      - Population standard deviation
      - Sample Standard deviation
      - Sample covariance
       
## [**Puzzle**](https://github.com/still-n0thing/Algopylib/tree/main/src/Algopylib/puzzle)

#### Implementation of simple Puzzles.
   - [Sudoku](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/puzzle/sudoku.py)
   - [Tic tac toe](https://github.com/still-n0thing/Algopylib/blob/main/src/Algopylib/puzzle/tictaktoe.py)
---


## Setup for devs

#### For Windows,
1. Use ```pip install virtualenv```  
2. Run ```virtualenv nameofvenv``` in terminal  
3. Activate that virtualenv (different for different os)  
4. Run ```pip install -r requirements.txt``` in terminal (virtualenv should be activated)  
5. To compile library, run ```python setup.py sdist bdist_wheel``` in terminal   


#### For other OS,
1. Use ```pip install virtualenv```  
2. Run ```virtualenv nameofvenv``` in terminal  
3. Activate that virtualenv  
4. Run following command in terminal,  
    * ```python -m pip install -upgrade pip```
    * ```pip install wheel```  
    * ```pip install twine```
    * ```pip install mypy```  
5. To compile library, run ```python setup.py sdist bdist_wheel``` in terminal   
---

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.  
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)