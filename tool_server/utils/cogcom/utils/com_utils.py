"""answer question base on the content of the image
@File    :   com_utils.py
@Time    :   2024/2/4
@Author  :   Ji Qi 
@Contact :   qj20@mails.tsinghua.edu.cn
"""
from PIL import Image
from grounding_parser import unrefine_box

HARD_PROMPT = 'Based on this new image and the reasoning history:'

class Node:
    def __init__(self, data, key):
        self.key = key
        self.data = data

def build_tree(dict_tree):
    dtree = {}
    max_depth = 0
    for k,v in dict_tree.items():
        k = k.replace('*', '~')
        dtree[k] = v
        max_depth  = (k.split('--')[1].split(',')[0])

    tree = {}
    for key in sorted(dtree.keys()):
        fid, curid = key.split('--')
        flevel, ford = fid.split(',')
        clevel, cord = curid.split(',')
        if ford == '~':
            for i in range(1000): # maximum
                fid = f'{flevel},{i}'
                if fid not in tree:
                    if fid.split(',')[0] == '-1' and i==0: # one root
                        tree[fid] = [Node(dtree[key], key)]
                        if clevel != max_depth:
                            tree[curid] = [] # add current node !
                    break
                tree[fid].append(Node(dtree[key], key))
                if clevel != max_depth:
                    tree[curid] = []
        else:
            tree[fid] = tree.get(fid, []) + [Node(dtree[key], key)]
            if clevel != max_depth:
                tree[curid] = []
    return tree


def find_paths(tree):
    paths = []
    level = 0
    path = []
    def dfs(node):
        nonlocal tree, path
        fid, curid = node.key.split('--')
        path.append(node)
        if 'found' in node.data and node.data['found']:
            # paths.append([n.data['found'] for n in path])
            paths.append([n.data for n in path])
        if curid in tree:
            for child in tree[curid]:
                dfs(child)
        path.pop()
    # for root in tree[0]:
    for root in tree.get('-1,0', []):
        dfs(root)
    return paths

def manipulate_image(pil_image, func, param, unrefine=True) -> Image:
    """ Manipulate original image and produce a new image if possible, otherwise return the original image.
      Args:
        @pil_image: an image instance of PIL.Image
        @func: name of the manipulation function, where currently supporting: [crop_and_zoomin, ]
        @para: input parameter of the manipulation function
      Return:
        an produced (or original) image instance of PIL.Image
    """
    new_img = pil_image
    try:
        if func and 'crop_and_zoomin' in func.lower():
            box = None
            if isinstance(param, list) and param and (isinstance(param[0], int) or isinstance(param[0], float)):
                box = param
            elif isinstance(param, list) and param and isinstance(param[0], list) and (isinstance(param[0][0], int) or isinstance(param[0][0], float)):
                box = param[0]
            if box:
                if unrefine:
                    box = unrefine_box(box, new_img.size[0], new_img.size[1])
                new_img = new_img.crop(box)
                new_img = new_img.resize((new_img.size[0]*2, new_img.size[1]*2), Image.Resampling.BICUBIC)
    except:
        new_img = pil_image
        print(f'Failed to manipulate image with func: {func}, param: {param}')
    print(f'Manipulating image from original size of {pil_image.size} to new size {new_img.size} ...')
    return new_img
