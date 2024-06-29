#include <iostream>
#include <vector>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip> 

#include <tuple>
#include <boost/math/distributions.hpp>
#include <boost/random.hpp>
#include <omp.h>
#include <functional>
#include <boost/math/distributions/beta.hpp>
#include <limits>

#include "Rcpp.h"
#include "RcppEigen.h"
#include "RcppNumerical.h"

#include "MultDirichletBoot.h"
#include "helper.h"
#include "fenrir.h"
#include "fenrir_grad.h"

#include "target.h"
#include "mh_general.h"

