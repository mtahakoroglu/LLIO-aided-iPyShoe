import matplotlib.pyplot as plt
import numpy as np

# Define the function
def func(x):
    return np.sin(x)

# Generate x values
x = np.linspace(0, 10, 100)

# Compute y values
y = func(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sin(x)')

# Add dashed grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add labels and title
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Plot of sin(x) with Dashed Grid Lines')

# Add a legend
plt.legend()

# Show the plot
plt.show()
