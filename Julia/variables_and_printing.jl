#   use 'println()' to print
println("Hello World!")

first_name = "Arnab"
last_name = "Dey"

val1 = 64
println(typeof(val1))   #   Int64

val2 = 3.1425
println(typeof(val2))   #   Float64

println("Hello, ", first_name, " ", last_name)

#convert(Int64, "1")    #! error: cannot 'convert' an object of type String to an object of type Int64

parse(Int64, "1")