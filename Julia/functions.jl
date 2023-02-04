#   *   ways to make functions

function f(x)
    x^2
end

println(f(2))
#   println(f("Arnab")) #   defined in some cases otherwise or not

sayhi(name) = println("hello $name")

sayhi("Arnab")
#   works on different data types
sayhi(5)

#   anonymous functions
sayhi2 = name -> println("Hello $name, nice to see you!")

sayhi2("Arnab")
sayhi2(5)

#   *   Duck typing: if it quacks it is probably a Duck

vec = [1, 2, 3]
mat = rand(Int8, 3, 3)

vec2 = [3, 5, 2]

# println(f(vec)) #!  will throw an error
println(mat)
# matres = f(mat)
# matres2 = f.(mat)  #   *   broadcasting
vecres2 = f.(vec)

#   *   mutating and non-mutating functions

sort(vec2)
println(vec2)
sort!(vec2)
println(vec2)


