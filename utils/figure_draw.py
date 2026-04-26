import matplotlib.pyplot as plt

# Data
heads = [1, 2, 4, 6, 8]
res2net101_csra = [90.68, 90.97, 91.21, 91.84, 90.92]
res2net50_csra = [90.41, 90.84, 90.94, 90.55, 90.32]
resnet152_csra = [90.23, 90.37, 89.83, 89.72, 88.33]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(heads, res2net101_csra, marker='o', color='black', label='Res2Net101+CSRA')
plt.plot(heads, res2net50_csra, marker='s', color='darkblue', label='Res2Net50+CSRA')
plt.plot(heads, resnet152_csra, marker='^', color='gray', label='ResNet152+CSRA')

# Labels and Title
plt.xlabel('Number of Attention Heads')
plt.ylabel('Mean Average Accuracy (%)')
# plt.title('Influence of the Number of Attention Heads in CSRA on Rescuenet Dataset')
plt.legend()
plt.grid(True)
plt.ylim(80, 100)

# Show plot
plt.show()
