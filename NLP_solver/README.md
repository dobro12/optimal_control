# NLP solver

There are a lot of NLP solvers like slsqp, pyOpt, scipy, etc.

- slsqp <https://github.com/jacobwilliams/slsqp> (Fortan)

- pyOpt <http://www.pyopt.org/> (python2)

- scipy <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html> (python)

- NLopt <https://github.com/stevengj/nlopt> (c)

I decide to use NLopt solver for c++.

The source code can be found here. <https://github.com/stevengj/nlopt>

## Installation

``` bash
git clone https://github.com/stevengj/nlopt
cd nlopt
mkdir build
cd build
cmake ..
make
sudo make install
```

## 2020.12.16

NLopt takes 3.6 seconds to calculate the "bangbang_control" optimization problem.

scipy takes 3.7 seconds... There is no advantage to use c++.
