x = 3
y = 10

if x > y
    println("$x is greater that $y")
elseif x < y
    println("$x is lesser than $y")
else
    println("$x and $y are equal")
end

res = x > y ? x : y
println(res)