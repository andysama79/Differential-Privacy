# n = 0
# while n < 10
#     global n += 1
#     println(n * 5)
# end

# names = ["Arnab", "Ankit", "Hriday", "Mangesh", "Siddharth"]

# for name in names
#     println(name)
# end

#   *   Some syntactic sugar

m, n = 5, 5
A = zeros(m, n)

for i in 1:m, j in 1:n
    A[i, j] = i + j
end
A

#   *   more 'Julia' way

A = [i + j for i in 1:m, j in 1:n]
A