#######################
# without decorator syntax
#######################
def greet(name):
    return f"Hello {name}"

def wrapper(func):
    def inner(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return inner

greet = wrapper(greet)

print(greet("Thanh"))

#######################
# with decorator syntax
#######################
@wrapper
def greet2(name):
    return f"Hello 2 {name}"

print("=" * 10)
print(greet2("Thanh"))

#######################
# decorator factory
#######################
def wrapper_factory(message):
    def wrapper(func):
        def inner(*args, **kwargs):
            print(message)
            return func(*args, **kwargs)
        return inner
    return wrapper

@wrapper_factory("Hello")
def greet3(name):
    return f"Hello 3 {name}"

#######################
# tool calling framework
#######################
REGISTRY = []

def tool():
    def decorator(func):
        REGISTRY.append(func)
        return func
    return decorator


@tool()
def hello():
    print("hi")

print(REGISTRY)  # contains hello