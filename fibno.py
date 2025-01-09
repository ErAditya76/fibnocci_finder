def fibonacci_generator(n):
    """
    Generate Fibonacci series up to n terms.
    :param n: Number of terms in the Fibonacci sequence
    """
    if n <= 0:
        print("Enter a positive integer.")
        return
    a, b = 0, 1
    for _ in range(n):
        print(a, end=" ")
        a, b = b, a + b


# Input for number of terms
num_terms = int(input("Enter the number of terms: "))
fibonacci_generator(num_terms)
