import pandas as pd
def loadCSV(filename):
    df = pd.read_csv(filename, sep='\t', usecols=['TrueValue', 'ObservedValue'])
    return df

def Tiers(df):
    correctCounterTier1 = 0
    correctCounterTier2 = 0
    correctCounterTier3 = 0
    total = len(df.index)
    for x in df.index:
        # Tier 1
        # 0 miles to 1 mile
        if (df.loc[x, 'TrueValue'] <= 1):
            if (abs(df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue']) <= 0.25):
                correctCounterTier1 += 1
        # 1 mile to 3 miles
        elif (df.loc[x, 'TrueValue'] <= 3):
            if (df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue'] >= -1 and df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue'] <= .5):
                correctCounterTier1 += 1
        # 3+ miles
        elif (df.loc[x, 'TrueValue'] > 3):
            # accounting for the fact that any number greater than 10 gets rounded down to 10
            if (df.loc[x, 'TrueValue'] >= 8.5 and df.loc[x, 'ObservedValue'] >= 7):
                correctCounterTier1 += 1
            elif (abs(df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue']) <= 1.5):
                correctCounterTier1 += 1
        else:
            print("error: no category found for this value.")
        #Tier 2
        # 0 miles to 1 mile
        if (df.loc[x, 'TrueValue'] <= 1):
            if (abs(df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue']) <= 0.25):
                correctCounterTier2 += 1
        # 1 mile to 3 miles
        elif (df.loc[x, 'TrueValue'] <= 3):
            if (abs(df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue']) <= 0.5):
                correctCounterTier2 += 1
        # 3+ miles
        elif (df.loc[x, 'TrueValue'] > 3):
            # accounting for the fact that any number greater than 10 gets rounded down to 10
            if (df.loc[x, 'TrueValue'] >= 9 and df.loc[x, 'ObservedValue'] >= 8):
                correctCounterTier2 += 1
            elif (abs(df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue']) <= 1):
                correctCounterTier2 += 1
        else:
            print("error: no category found for this value.")
        #Tier 3
        # 0 miles to 1.49999999 miles
        if (df.loc[x, 'TrueValue'] < 1.5):
            if (abs(df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue']) <= 0.25):
                correctCounterTier3 += 1
        # 1.5 miles to 1.9999999 miles
        elif (df.loc[x, 'TrueValue'] < 2):
            if (df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue'] >= -0.25 and df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue'] <= 0.5):
                correctCounterTier3 += 1
        # 2 miles to 2.999999 miles
        elif (df.loc[x, 'TrueValue'] < 3):
            if (abs(df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue']) <= 0.5):
                correctCounterTier3 += 1
        # 3 miles to 3.999999 miles
        elif (df.loc[x, 'TrueValue'] < 4):
            if (df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue'] >= -.5 and df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue'] <= 1):
                correctCounterTier3 += 1
        # 4+ miles
        elif (df.loc[x, 'TrueValue'] >= 4):
            # accounting for the fact that any number greater than 10 gets rounded down to 10
            if (df.loc[x, 'TrueValue'] >= 9 and df.loc[x, 'ObservedValue'] >= 8):
                correctCounterTier3 += 1
            elif (abs(df.loc[x, 'TrueValue'] - df.loc[x, 'ObservedValue']) <= 1):
                correctCounterTier3 += 1  
    print("Tier 1:", ". Total number of correct samples: ", correctCounterTier1, "Total number of samples: ", total)
    print("Tier 1 Accuracy: ", correctCounterTier1/total*100, "%")
    print("Tier 2:", ". Total number of correct samples: ", correctCounterTier2, "Total number of samples: ", total)
    print("Tier 2 Accuracy: ", correctCounterTier2/total*100, "%")
    print("Tier 3:", ". Total number of correct samples: ", correctCounterTier3, "Total number of samples: ", total)
    print("Tier 3 Accuracy: ", correctCounterTier3/total*100, "%")

df = loadCSV('zachFiles/testVisibility.csv')
Tiers(df)
