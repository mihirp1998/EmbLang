import random
import csv
pokemon = ["watertotle","squirtle","pikachu","bulbasor","charmander"]
species = ["water","water","magnet","green","fire"]
writer = csv.writer(open('metadata.tsv', 'w'), delimiter='\t', lineterminator='\n')
writer.writerow(["Pokemon","Species"])
for i in range(5):
    writer.writerow([pokemon[i],species[i]])




# with open('values.tsv', 'w') as tsvfile:
#     writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
#     for i in range(5):
#         writer.writerow([random.random(),random.random()])