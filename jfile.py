import json
import os
import os.path as loc
class jfile:
    def __init__(self, name):
        self.name = name
        self.data = {}
        if loc.isfile(self.name + ".json"):
            self.fname = open(self.name+".json", "r+")
            if os.stat(self.name+".json").st_size != 0:
                self.data = json.loads(self.fname.read())
            self.fname.close()

            

    def save(self):
        fname = open(self.name + ".json", "w")
        fname.write(json.dumps(self.data, sort_keys=True, indent=4,separators=(',',': ')))
        fname.close()   
