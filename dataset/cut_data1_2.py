import pandas as pd
import os

here = os.path.abspath(os.path.dirname(__file__))

try:
    os.stat(os.path.join(here, "ten_countries"))  # directory to save shorted dataset exists
except:
    os.mkdir(os.path.join(here, "ten_countries"))  # if not: create a new directory


total_dataS = pd.read_csv("original/cities_original_subset.txt", sep="#").T  # macht ein DataFrame(eine Art Tabelle) aus der Datei und dreht es um("T")
#  um indexieren zu koennen

total_dataS.index=["id", "Name(Unicode)", "Name(ASCII)", "grad1", "grad2", "Country"]  # labelt(indexiert) dataframe, um nach bestimmten Spalten zu greifen
total_dataS = total_dataS.T  # dreht die Tabelle in ihre urspruengliche Position

items_counts = total_dataS['Country'].value_counts().sort_values()  # frequency of each country
max_countries = items_counts[len(items_counts)-10:len(items_counts)].index  # 10 most frequent countries for the shorted dataset
shorted1 = total_dataS[total_dataS['Country'].isin(list(max_countries))]  # information about each city in each selected country
shorted2 = pd.DataFrame(list(shorted1['Country']), index=list(shorted1["Name(ASCII)"]))  # erwuenschte labels in die Liste einfuegen
shorted2 = shorted2.sample(frac=1).dropna() #Liste mischen

i_tr = int(len(shorted2.index)*0.7)  #Proportionen fuer train test und validdata bestimmen
i_v = int(len(shorted2.index)*0.85)
path_new = "ten_countries/"
shorted2[1:i_tr].to_csv(path_new+"train", sep='#', index=True, header=True)   #verschiedene Daten als csv! speichern
shorted2[i_tr:i_v].to_csv(path_new+"valid", sep='#', index=True, header=True)
shorted2[i_v:len(shorted2)].to_csv(path_new+"test", sep='#', index=True, header=True)