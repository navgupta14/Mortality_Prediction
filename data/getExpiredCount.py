import pandas as pd
import numpy as np

data = pd.read_csv("combined_features.csv")
expired = np.array(data.expired)
first30k = expired[:30000]
rest = expired[30000:]

expiredCount = 0
livingCount = 0
for e in first30k:
        if int(e) == 1:
                expiredCount += 1
        if int(e) == 0:
                livingCount += 1

print "For first 30k"
print "Expired Count is ", expiredCount
print "Living Count is ", livingCount
print expiredCount + livingCount
print "\n"

expiredCount = 0
livingCount = 0
for e in rest:
        if int(e) == 1:
                expiredCount += 1
        if int(e) == 0:
                livingCount += 1

print "For rest"
print "Expired Count is ", expiredCount
print "Living Count is ", livingCount
print expiredCount + livingCount
print "\n"

expiredCount = 0
livingCount = 0
for e in expired:
	if int(e) == 1:
		expiredCount += 1
	if int(e) == 0:
		livingCount += 1

print "For overall data"
print "Expired Count is ", expiredCount
print "Living Count is ", livingCount
print expiredCount + livingCount
