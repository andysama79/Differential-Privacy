import opendp
from opendp.mod import enable_features
enable_features('contrib')

from opendp.transformations import make_identity
from opendp.typing import VectorDomain, AllDomain, SymmetricDistance

identity = make_identity(D=VectorDomain[AllDomain[str]], M=SymmetricDistance)
identity(["Hello, World!"])

print(identity[0])

#   import some types to have them in scope. "make_identity" is a constructor function,
#   and the imports from opendp.typing are necessary for disambiguating the types the transformation 
#   will work with

#   calling "make_identity()" to construct an indentiy Transformation
#   important: OpenDP is statically typed, some type information needs to be specified.
#   this is achieved by supplying some key-value arguments
#   D=VectorDomain[AllDomain[str]] says that we want the Transormation to have an input
#   and output domain consisting of alal string vectors,and
#   M=SymmetricDistance says that we wnat the resulting Transformation to use the OpenDP type SymmetricDistance
#   for its input and output Metric