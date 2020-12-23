import ipdb
from lib.tree import Tree
st = ipdb.set_trace
import copy
import pickle
supplement = ["Red Rubber Cylinder to the left front of Blue Rubber Cube to the left front of Green Rubber Cylinder to right front of Blue Rubber Cube","Red Rubber Cube to the left front of the Blue Rubber Sphere to the right front of Cyan Metal Cylinde"
"Purple Cylinder to the left behind of Brown Cube to the left front of Purple Sphere","Purple Cylinder to the left behind of Cyan Cube to the left front of Cyan Cube","Cyan Cube to the left behind of Gray Sphere to the left front of Blue Cube"
"Red Sphere to the left behind of Cyan Cylinder to the left front of Red Sphere","red cylinder to the right behind of green cube","pink cylinder to the left front of red cylinder","brown sphere to the left of purple sphere",
"red cylinder to the left front of gray cylinder","blue sphere to the right front of blue cylinder to the left front of blue sphere","purple sphere to the left front of red cube to the left behind of purple sphere"
]

paper = ["red sphere to the left front of brown sphere to the left behind of brown sphere","green sphere to the left front of red sphere to the left behind of green sphere",
"brown cylinder to the left front of green cube to the left front of brown cylinder to the left behind of brown cube","blue cube to the left behind of gray sphere to the the left of gray cube"
"cyan sphere to the right front of red sphere too the right front of red cube to the left of cyan cube","cyan cube to the right front of cyan sphere to the left front of blue cube"
,"red cube inside blue bowl","red cube inside green bowl","red bowl inside blue cube","red bowl inside green cube","Purple cylinder to the left front of red sphere to the left of blue sphere"]

# paper =[["red","sphere","left-front","brown","sphere"],["red","sphere","left-front","brown","sphere","left-behind","brown","sphere"],["blue","cube","left-front","red","sphere","left-front","brown","sphere","left-behind","brown","sphere"]]
dictionary_colors = ['brown','yellow','cyan','blue','gray','purple','green','red']

paper =[["cyan","cylinder","right","red","sphere","left-front","green","sphere"]
		,["red","cylinder","right-front","red","sphere","left-front","blue","sphere"],
		["blue","cylinder","left-front","red","sphere"],["cyan","cylinder","right","yellow","sphere"],["blue","sphere","left-behind","green","cube"],["cyan","sphere","left-behind","red","cube"],["cyan","sphere","left-behind","red","cylinder"]]

# paper =[["blue","cylinder","left-front","red","sphere","left-front","cyan","sphere","left-behind","purple","sphere","left","green","cylinder","right","brown","sphere","left","yellow","sphere"]
# 		,["cyan","cylinder","right","yellow","sphere","left-front","green","sphere","left-behind","blue","sphere","left","gray","cylinder","right","red","sphere"]]
# paper = [["blue","cylinder","left-front","red","sphere"],["cyan","cylinder","right","yellow","sphere"],["blue","sphere","left-behind","green","cube"]]
paper_d = [["cyan","cylinder"],["green","cylinder"],["blue","cylinder"],["cyan","sphere"],["green","sphere"],["blue","sphere"]]

paper =[["red","sphere","left-front","green","sphere"],["red","sphere","right-front","green","sphere"],\
["red","sphere","left-behind","green","sphere"],["red","sphere","right-behind","green","sphere"],\
["red","sphere","left","green","sphere"],["red","sphere","right","green","sphere"],\
["red","sphere","right","green","sphere"],["red","sphere","left-behind","green","sphere"]]

# paper = [["yellow","sphere","left-front","green","sphere","left-behind","blue","sphere","left-front","blue","cylinder","left-behind","red","cube","left-front","gray","cube"],
# ["gray","sphere","left-front","blue","sphere","left-front","red","sphere","left-behind","cyan","sphere","left-behind","green","sphere"]]

# paper = [["yellow","sphere","right","green","sphere","left","blue","sphere","right","blue","cylinder","left-behind","red","sphere","left","gray","sphere"],
# ["gray","sphere","right-front","blue","sphere","left-behind","red","sphere","right","cyan","sphere","right","green","sphere"],["yellow","sphere","left-front","green","sphere","left-behind","blue","sphere","left-front","blue","cylinder","left-behind","red","cube","left-front","gray","cube"],
# ["gray","sphere","left-front","blue","sphere","left-front","red","sphere","left-behind","cyan","sphere","left-behind","green","sphere"],["red","sphere","left-front","green","sphere"],["red","sphere","right-front","green","sphere"],\
# ["red","sphere","left-behind","green","sphere"],["red","sphere","right-behind","green","sphere"],\
# ["red","sphere","left","green","sphere"],["red","sphere","right","green","sphere"],\
# ["red","sphere","right","green","sphere"],["red","sphere","left-behind","green","sphere"],["cyan","cylinder","right","red","sphere","left-front","green","sphere"],["red","cylinder","right-front","red","sphere","left-front","blue","sphere"],\
# ["blue","cylinder","left-front","brown","sphere","left-front","cyan","sphere"],["red","sphere","right","brown","sphere","left-front","green","sphere"],["cyan","cylinder","right","red","sphere","left-front","green","sphere","left-front","blue","sphere"],["blue","cylinder","right-front","red","sphere","left-front","blue","sphere","left-front","green","sphere"],\
# ["blue","cylinder","left-front","brown","sphere","left-front","cyan","sphere","left-front","yellow","sphere"],["red","sphere","right","brown","sphere","left-front","green","sphere","left-behind","green","sphere"],
# ["green","sphere","right","yellow","sphere","left-front","green","sphere","left-behind","green","sphere"],["red","cylinder","right","red","sphere","left-front","green","sphere"],["yellow","cylinder","right","red","sphere","left-front","blue","sphere"],\
# ["yellow","cylinder","left","brown","sphere","left-front","cyan","sphere"],["green","sphere","right","brown","sphere","left-front","green","sphere"],["brown","cylinder","right","red","sphere","left-front","green","sphere","left-front","blue","sphere"],["gray","cylinder","right-front","red","sphere","left-front","blue","sphere","left-front","green","sphere"],\
# ["red","cylinder","right","brown","sphere","left-front","cyan","sphere","left-front","yellow","sphere"],["yellow","sphere","right","brown","sphere","left-front","green","sphere","left-behind","green","sphere"],
# ["blue","sphere","right","yellow","sphere","left-front","green","sphere","left-behind","green","sphere"]]

# st()
paper =[
		["red","cylinder","right","yellow","sphere","left-front","green","sphere","right","blue","sphere"],
		["red","cylinder","right","cyan","sphere","left-front","green","sphere","right","blue","sphere"],
		["red","cylinder","right","yellow","sphere","left-front","brown","sphere","right","blue","sphere"],
		["red","cylinder","right","green","sphere","left-front","brown","sphere","right","blue","sphere"],
		["red","cylinder","right","blue","sphere","left-front","brown","sphere","right","gray","sphere"],
		["red","cylinder","right","yellow","sphere","left-front","cyan","sphere","right","blue","sphere"],
		["red","cylinder","right","green","sphere","left-front","brown","sphere","right","blue","sphere"],
		["red","cylinder","right","cyan","sphere","left-front","green","sphere","right","blue","sphere"],

		["red","cylinder","right","yellow","sphere","left-front","green","sphere","right","blue","sphere","left-front","yellow","sphere"],
		["red","cylinder","right","cyan","sphere","left-front","green","sphere","right","blue","sphere","left-front","yellow","sphere"],
		["red","cylinder","right","yellow","sphere","left-front","brown","sphere","right","blue","sphere","left-front","green","sphere"],
		["red","cylinder","right","green","sphere","left-front","brown","sphere","right","blue","sphere","left-front","gray","sphere"],
		["red","cylinder","right","blue","sphere","left-front","brown","sphere","right","gray","sphere","left-front","purple","sphere"],
		["red","cylinder","right","yellow","sphere","left-front","cyan","sphere","right","blue","sphere","left-front","green","sphere"],
		["red","cylinder","right","green","sphere","left-front","brown","sphere","right","blue","sphere","left-front","gray","sphere"],
		["red","cylinder","right","cyan","sphere","left-front","green","sphere","right","blue","sphere","left-front","yellow","sphere"],
		]

paper = [["blue","sphere","left-behind","green","cube"],
["red","cylinder","right","cyan","sphere"],
["red","cylinder","right-front","red","sphere","left-front","blue","sphere"],
["blue","sphere","right-behind","red","sphere","left-behind","green","cylinder"],
["red","cylinder","right","yellow","sphere","left-front","brown","sphere","right","blue","sphere"],
["yellow","sphere","left-front","blue","sphere","right","green","sphere","left-front","yellow","sphere","right","red","cylinder"],
]

paper = [["blue","sphere","left-behind","green","cube"],
["red","cylinder","right","cyan","sphere"],
["red","cylinder","right-front","red","sphere","left-front","blue","sphere"],
["blue","sphere","right-behind","red","sphere","left-behind","green","cylinder"],
["red","cylinder","right","yellow","sphere","left-front","brown","sphere","right","blue","sphere"],
["yellow","sphere","left-front","blue","sphere","right","green","sphere","left-front","yellow","sphere","right","red","cylinder"]]

paper =[["red","sphere","left-front","green","sphere"],["red","sphere","right-front","green","sphere"],\
["red","sphere","left-behind","green","sphere"],["red","sphere","right-behind","green","sphere"],\
["red","sphere","left","green","sphere"],["red","sphere","right","green","sphere"],\
["red","sphere","right","green","sphere"],["red","sphere","left-behind","green","sphere"],\
]

dictionary_colors = ['brown','yellow','cyan','blue','gray','purple','green','red']
dictionary_customVae = ['left-front','right-front','right-behind','right','left-behind','left']
dictionary_shapes = ['sphere']
import random
func2Obj = lambda:[random.choice(dictionary_colors),random.choice(dictionary_shapes),random.choice(dictionary_customVae),random.choice(dictionary_colors),random.choice(dictionary_shapes)]
func3Obj = lambda:[random.choice(dictionary_colors),random.choice(dictionary_shapes),random.choice(dictionary_customVae),random.choice(dictionary_colors),random.choice(dictionary_shapes),random.choice(dictionary_customVae),random.choice(dictionary_colors),random.choice(dictionary_shapes)]
func4Obj = lambda:[random.choice(dictionary_colors),random.choice(dictionary_shapes),random.choice(dictionary_customVae),random.choice(dictionary_colors),random.choice(dictionary_shapes),random.choice(dictionary_customVae),random.choice(dictionary_colors),random.choice(dictionary_shapes),random.choice(dictionary_customVae),random.choice(dictionary_colors),random.choice(dictionary_shapes)]

# st()
# obj2 = [func2Obj() for i in range(50)]
obj3 = [func4Obj() for i in range(50)]
paper = obj3
# paper = [["blue","sphere","right-behind","red","sphere","left-behind","green","cylinder"]]
# st()
describe = pickle.load(open("describe.p","rb"))
# describe.children[0].word = "small"
layout = pickle.load(open("layout.p","rb"))
# st()
def combine_mod(val):
	color,noun = val
	tree_temp = copy.deepcopy(describe)	
	tree_temp.children[0].children[0].word = color
	tree_temp.word = noun
	return tree_temp

def layout_mod(val):
	# st()
	trees,pos = val
	tree_temp = copy.deepcopy(layout)
	tree_temp.word = pos
	tree_temp.children = trees
	return tree_temp
layout_tree=True
if layout_tree:
	# val = []
	tree = Tree()
	for sent in paper:
		sent_temp = sent
		val = len(sent)
		assert (val+1)%3 ==0
		numObj = (val+1)//3
		# st()
		temp = []
		pos_l = []
		for obj in range(numObj):
			temp.append(combine_mod(sent[obj*3:obj*3+2]))
			if (obj*3+2) < val:
				pos_l.append(sent[obj*3+2])
		print([i.word for i in temp])
		print(pos_l)
		iterval = reversed(range(len(pos_l)))
		for i in iterval:
			pos = pos_l[i]
			temp[i:i+2] = [layout_mod((temp[i:i+2][::-1],pos))]
		# st()
		name_sent = "_".join(sent_temp)
		assert len(temp) ==1
		tree = temp[0].wordVal = name_sent
		pickle.dump(temp[0],open("only4" +"/"+name_sent+".p","wb"))
	print("check")
else:
	for sent in paper_d:
		tree = combine_mod(sent)
		tree.wordVal ="_".join(sent)
		pickle.dump(tree,open("tree_single" +"/"+tree.wordVal+".p","wb"))

		# print(i)