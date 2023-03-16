from opendp.transformations import *
from opendp.mod import enable_features
enable_features('contrib') # we are using un-vetted constructors

num_tests = 3  # d_in=symmetric distance; we are told this is public knowledge
budget = 1. # d_out=epsilon

num_students = 50  # we are assuming this is public knowledge
size = num_students * num_tests  # 150 exams
bounds = (0., 100.)  # range of valid exam scores- clearly public knowledge
constant = 70. # impute nullity with a guess

transformation = (
    make_split_dataframe(',', col_names=['Student', 'Score']) >>
    make_select_column(key='Score', TOA=str) >>
    make_cast(TIA=str, TOA=float) >>
    make_impute_constant(constant=constant) >>
    make_clamp(bounds) >>
    make_bounded_resize(size, bounds, constant=constant) >>
    make_sized_bounded_mean(size, bounds)
)

from opendp.measurements import make_base_laplace
from opendp.mod import enable_features, binary_search_param

# Please make yourself aware of the dangers of floating point numbers
enable_features("floating-point")

# Find the smallest noise scale for which the relation still passes
# If we didn't need a handle on scale (for accuracy later),
#     we could just use binary_search_chain and inline the lambda
make_chain = lambda s: transformation >> make_base_laplace(s)
scale = binary_search_param(make_chain, d_in=num_tests, d_out=budget) # -> 1.33
measurement = make_chain(scale)

# We already know the privacy relation will pass, but this is how you check it
assert measurement.check(num_tests, budget)

# How did we get an entire class full of Salils!? ...and 2 must have gone surfing instead
mock_sensitive_dataset = "\n".join(["Salil,95"] * 148)

# Spend 1 epsilon creating our DP estimate on the private data
release = measurement(mock_sensitive_dataset) # -> 95.8

