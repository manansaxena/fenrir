# fenrir
Package for Scalable Inference for Bayesian Multinomial Logistic-Normal Dynamic Linear Models

## Fenrir Package Requirements

### R Package Dependencies
Ensure these packages are installed from CRAN:

- **Rcpp**: version 1.0.10 or higher
- **RcppEigen**
- **RcppDist**
- **RcppNumerical**

### C++ Library Dependencies
The following C++ libraries are required:

- **Standard Libraries**:
  - `<iostream>`
  - `<vector>`
  - `<cmath>`
  - `<chrono>`
  - `<fstream>`
  - `<iomanip>`
  - `<tuple>`
  - `<functional>`
  - `<limits>`
  - `<random>`

- **Boost Libraries**:
  - `boost/math/special_functions/gamma.hpp`
  - `boost/math/distributions.hpp`
  - `boost/math/distributions/beta.hpp`
  - `boost/random.hpp`

### Local Headers
These headers are part of the Fenrir package:

- `"MultDirichletBoot.h"`
- `"helper.h"`
- `"fenrir.h"`
- `"fenrir_grad.h"`

Make sure that the Boost libraries are correctly installed and configured in your system to compile this package. 
To configure the use of Boost libraries please make changes in the Makevars file.
```
PKG_CPPFLAGS+=-I<path to boost library stored>
```
