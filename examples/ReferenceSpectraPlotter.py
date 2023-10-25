import json
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('GTK3Agg') # or any other backend that you prefer
# Given a JSON file with various reference spectra, this program graphs their reflectance as a function of wavelength/band
# Assumes that the reflectance values are IN ORDER (i.e. 0001 -> 0002 -> etc.)

# Opening JSON file
#assumes running from main hypertools 
f = open('json/spectral_database1.json')

jsonAsDict = json.load(f)
'''
# For each reference spectrum
for i in jsonAsDict["Spectral_Information"]:
    print(i)
    # For each reflectance value
    for j in jsonAsDict["Spectral_Information"][i]:
        print(j)
'''

firstClass = list(jsonAsDict['Spectral_Information'].keys())[0]
plt.figure().set_figwidth(15)

x = [int(key) for key in jsonAsDict['Spectral_Information'][firstClass].keys()] # assuming all inner dictionaries have the same keys
ys = []
for i in jsonAsDict["Spectral_Information"]:
    #print(i)
    ys.append([jsonAsDict["Spectral_Information"][i][key] for key in jsonAsDict["Spectral_Information"][i].keys()])
    plt.plot(x, ys[-1], label=i)

# Customize chart
plt.xlabel('Band')
plt.ylabel('Reflectance (0-255)')
plt.title('Spectra')
plt.legend()
# Display chart
plt.savefig("images/ReferenceSpectraPlot.png")
plt.show()

#y2 = [data['Item2'][key] for key in data['Item2'].keys()]

# Closing file
f.close()