#   *Enclose your strings in " " or """ """!

s1 = "I am a string"

s2 = """I am also a string"""

#   !   single quotes will produce an error

#   *   use the '$' sign to insert existing variable into stirng and to evaluate expressions within a string

name = "Arnab"
fingers = 10
toes = 10

println("Hello, my name is $name.")
println("I have $fingers fingers and $toes toes.")

#   *   String Concatenation

println(string("How many cats ", "are too many cats?"))

println(string("I don't know, but ", 10, " are too few."))

#   *   We can also use '*' or string interpolation!

s3 = "How many cats ";
s4 = "are too many cats?";

println(s3*s4)

"$s3$s4"
