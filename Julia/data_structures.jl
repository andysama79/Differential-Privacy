#   *   Types covered: Dictionaries, Tuples, Arrays

myPhoneBook = Dict("Arnab" => "0014", "Hriday" => "0074", "Siddharth" => "0080")

#   *   add a new key-value pair

myPhoneBook["Mangesh"] = "0038"

myPhoneBook

#   *   get the value of a key

println(myPhoneBook["Mangesh"])

#   *   get a key and simulataneously delete
pop!(myPhoneBook, "Siddharth")

myPhoneBook

myFavoriteAnimals = ("penguins", "cats", "sugargliders")

#   *   indexing

println(myFavoriteAnimals[1])

#   *   immmutability

#   myFavoriteAnimals[1] = "otters" #!  will throw an error since tuples are immutable

myFriends = ["Mangesh", "Siddharth", "Hriday", "Ankit"]
fibonacci = [1, 1, 2, 3, 5, 8, 13]
mix = [1, 2, 3.0, "hi"]

#   *   indexing allowed; mutable - arrays

#   *   can use push! and pop! functions; former adds an elemetn to the end of an array and latter remvoes the last element of an array

#   *   can create multi-dimensional arrays

println(rand(Int16, 4, 3))