# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
#### Developed by A sasidharan
#### Register no: 212221240049
#### Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
~~~
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/content/globaltemper.csv')

# Use a numerical column from the DataFrame instead of 'City'
y = df['AverageTemperature'].values 
x = np.arange(len(y))

A = np.vstack([x, np.ones(len(x))]).T
linear_trend = np.linalg.lstsq(A, y, rcond=None)[0]

y_linear_trend = linear_trend[0] * x + linear_trend[1]

degree = 2
coeffs = np.polyfit(x, y, degree)
y_poly_trend = np.polyval(coeffs, x)

plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Original Data', marker='o')
plt.plot(x, y_linear_trend, label='Linear Trend', linestyle='--')
plt.plot(x, y_poly_trend, label='Polynomial Trend', linestyle='--')
plt.title('Linear and Polynomial Trend Lines')
plt.xlabel('Index')
plt.ylabel('AverageTemperature') # Changed y-axis label to reflect the data used
plt.legend()
plt.show()

print("Linear Trend Equation: y = {:.2f}x + {:.2f}".format(linear_trend[0], linear_trend[1]))
print("Polynomial Trend Equation: y = {:.2f}x^2 + {:.2f}x + {:.2f}".format(coeffs[0], coeffs[1], coeffs[2]))
~~~
### OUTPUT
![image](https://github.com/user-attachments/assets/fc07fab7-9c98-4bec-8373-7bf9b274fd32)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
