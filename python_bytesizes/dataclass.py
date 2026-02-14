from dataclasses import dataclass

class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __repr__(self): # representation
        return f"User(name={self.name}, age={self.age})"

    def __eq__(self, other): # equality
        return self.name == other.name and self.age == other.age

@dataclass
class UserWithDataclass:
    name: str
    age: int

if __name__ == "__main__":
    user1 = User("Thanh", 20)
    user2 = User("Thanh", 20)
    user3 = User("John", 21)
    print(user1 == user2)
    print(user1 == user3)
    print(user1)
    print("=" * 10)
    user4 = UserWithDataclass("Thanh", 20)
    user5 = UserWithDataclass("Thanh", 20)
    user6 = UserWithDataclass("John", 21)
    print(user4 == user5)
    print(user4 == user6)
    print(user4)