import pandas as pd
import sys
import numpy

CAP_DISTANCE = True

def loadCSV(filename):
    df = pd.read_csv(filename, sep=',', usecols=['ObservedValue', 'TrueValue'])

    return df

def Tiers(df):
    correctTier1 = [0.0, 0.0, 0.0]
    totalsTier1  = [0.0, 0.0, 0.0]
    correctTier2 = [0.0, 0.0, 0.0]
    totalsTier2  = [0.0, 0.0, 0.0]
    correctTier3 = [0.0, 0.0, 0.0, 0.0, 0.0]
    totalsTier3  = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    for x in df.index:
        tVal = df.at[x, 'TrueValue']
        oVal = df.at[x, 'ObservedValue']
        
        if CAP_DISTANCE:
            if tVal > 10.0:
                tVal = 10.0
            if oVal > 10.0:
                oVal = 10.0
                
        diff = oVal - tVal
        
        # Tier 1
        # 0 miles to 1 mile
        if (0.0 <= tVal <= 1.0):
            if (abs(diff) <= 0.25):
                correctTier1[0] += 1
            totalsTier1[0] += 1
        # 1 mile to 3 miles
        elif (1.0 < tVal <= 3.0):
            if (-0.5 <= diff <= 1.0):
                correctTier1[1] += 1
            totalsTier1[1] += 1
        # 3+ miles
        elif (3.0 < tVal):
            if (abs(diff) <= 1.5):
                correctTier1[2] += 1
            totalsTier1[2] += 1
 
        #Tier 2
        # 0 miles to 1 mile
        if (0.0 <= tVal <= 1.0):
            if (abs(diff) <= 0.25):
                correctTier2[0] += 1
            totalsTier2[0] += 1
        # 1 mile to 3 miles
        elif (1.0 < tVal <= 3.0):
            if (abs(diff) <= 0.5):
                correctTier2[1] += 1
            totalsTier2[1] += 1
        # 3+ miles
        elif (3.0 < tVal):
            if (abs(diff) <= 1.0):
                correctTier2[2] += 1
            totalsTier2[2] += 1
                
                
        if (1.25 < tVal < 1.5 or 1.75 < tVal < 2.0 or
            2.5 < tVal < 3.0 or 3.5 < tVal < 4.0):
            continue
        
        if tVal > 10.0:
            tVal = 10.0
        if oVal > 10.0:
            oVal = 10.0
        
        diff = oVal - tVal
        
        #Tier 3
        # 0 miles to 1.25 miles
        if (0.0 <= tVal <= 1.25):
            if (abs(diff) <= 0.25):
                correctTier3[0] += 1
            totalsTier3[0] += 1
        # 1.5 miles to 1.75 miles
        elif (1.5 <= tVal <= 1.75):
            if (-0.5 <= diff <= 0.25):
                correctTier3[1] += 1
            totalsTier3[1] += 1
        # 2 miles to 2.5 miles
        elif (2.0 <= tVal <= 2.5):
            if (abs(diff) <= 0.5):
                correctTier3[2] += 1
            totalsTier3[2] += 1
        # 3 miles to 3.5miles
        elif (3.0 <= tVal <= 3.5):
            if (-1.0 <= diff <= 0.5):
                correctTier3[3] += 1
            totalsTier3[3] += 1
        # 4+ miles
        elif (4.0 <= tVal <= 10.0):
            if (abs(diff) <= 1):
                correctTier3[4] += 1
            totalsTier3[4] += 1

    confidenceTier1 = numpy.divide(correctTier1, totalsTier1)
    confidenceTier2 = numpy.divide(correctTier2, totalsTier2)
    confidenceTier3 = numpy.divide(correctTier3, totalsTier3)

    print("Tier 1: ", "%.4f" % (sum(correctTier1) / sum(totalsTier1)))
    print("   Block 1: ", "%.4f" % confidenceTier1[0])
    print("      0 to 1 mile: ", "%.4f" % confidenceTier1[0])
    print("   Block 2: ", "%.4f" % confidenceTier1[1])
    print("      >1 to 3 miles: ", "%.4f" % confidenceTier1[1])
    print("   Block 3: ", "%.4f" % confidenceTier1[2])
    print("      >3 miles: ", "%.4f" % confidenceTier1[2])
    
    print("\nTier 2: ", "%.4f" % (sum(correctTier2) / sum(totalsTier2)))
    print("   Block 1: ", "%.4f" % confidenceTier2[0])
    print("      0 to 1 mile: ", "%.4f" % confidenceTier2[0])
    print("   Block 2: ", "%.4f" % confidenceTier2[1])
    print("      >1 to 3 miles: ", "%.4f" % confidenceTier2[1])
    print("   Block 3: ", "%.4f" % confidenceTier2[2])
    print("      >3 miles: ", "%.4f" % confidenceTier2[2])
    
    print("\nTier 3: ", "%.4f" % (sum(correctTier3) / sum(totalsTier3)))
    print("   Block 1: ", "%.4f" % confidenceTier3[0])
    print("      0 to 1.25 miles: ", "%.4f" % confidenceTier3[0])
    print("   Block 2: ", "%.4f" % confidenceTier3[1])
    print("      1.5 to 1.75 miles: ", "%.4f" % confidenceTier3[1])
    print("   Block 3: ", "%.4f" % (sum(correctTier3[2:5]) / sum(totalsTier3[2:5])))
    print("      2.0 to 2.5 miles : ", "%.4f" % confidenceTier3[2])
    print("      3.0 to 3.5 miles : ", "%.4f" % confidenceTier3[3])
    print("      4.0 to 10.0 miles: ", "%.4f" % confidenceTier3[4])


df = loadCSV(sys.argv[1])
Tiers(df)
