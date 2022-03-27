import json
import pprint
f = open("Data/val2/init_jr_test.json")

data = json.load(f)

pprint.pprint(data)
