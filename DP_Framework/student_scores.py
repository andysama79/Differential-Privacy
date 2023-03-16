
#? compute the DP mean of a csv dataset of student exam scores,
#? using a privacy budget of 1 epsilon.
#? Accuracy estimate with 95% confidence

#* public knowledge that the class only has three exams, each student
#* may contribute at most three records, so our symmetric distance
#* 'd_in' is 3

#TODO:  1) parse a csv, 2) select a column, 3) cast,
#TODO:  4) impute, 5) clamp, 6) resize
#TODO:  Then aggregate with the mean.
#%%
#*   imports
from opendp.transformations import *
from opendp.mod import enable_features, binary_search_param
enable_features('contrib')  #   we are using = un-vetted constructors
from opendp.measurements import make_base_laplace
enable_features('floating-point')
from opendp.accuracy import laplacian_scale_to_accuracy
#%%
#*   constants: public knowledge
num_tests = 3   #   d_in = symmetric distance
budget = 1  #   d_out=epsilon
alpha = .05

num_students = 50   #   assuming public knowledge
size = num_students * num_tests #   150 exams
bounds = (0., 100.) #   range of valid exam scores - clearly public knowledge
constant = 70.  #   impute nullity with a guess
#%%
#*  transformations
transformation = (
    make_split_dataframe(',', col_names=['Student', 'Score']) >>
    make_select_column(key='Score', TOA=str) >>
    make_cast(TIA=str, TOA=float) >>
    make_impute_constant(constant=constant) >>
    make_clamp(bounds) >>
    make_bounded_resize(size, bounds, constant=constant) >>
    make_sized_bounded_mean(size, bounds)
)
#%%
#* find the smallest noise scale for which the relation still passes
#* if we didn't need a handle on scale (for accuracy later)
#* we could just use binary_search_chain and inline the lambda

make_chain = lambda s: transformation >> make_base_laplace(s)
scale = binary_search_param(make_chain, d_in=num_tests, d_out=budget)   #->1.33
measurement = make_chain(scale)

#!   We already know the privacy realtion will pass, but this is how you chect it
assert measurement.check(num_tests, budget)

#!  How did we get an entire class full of Salils!? ... and 2 must have gone surfing instead
mock_sensitive_dataset = "\n".join(["Salil, 95"] * 148)
release = measurement(mock_sensitive_dataset)   #   -> 95.8
#%%
#*  also wanted an accuracy estimate
#*  'laplacian_scale_to_accuracy' can be used to convert the earlier discorevered noise sclae parameter inot an accuracy estimate
accuracy = laplacian_scale_to_accuracy(scale, alpha)
(f"When the laplace scale is {scale}, "
 f"the DP estimate differs from the true value by no more than {accuracy} "
 f"at a statistical significance level alpha of {alpha}, "
 f"or with (1 - {alpha})100% = {(1 - alpha) * 100}% confidence")