from dreamcoder.program import Primitive
from dreamcoder.type import baseType, tint, tbool, arrow, t0, t1, TypeConstructor


def tarray(t): return TypeConstructor("array", [t])


tobj = baseType("tobj")  # the objects from the gym minigrid env
tgameobj = baseType("tgameobj")  # the objects from the gym minigrid env
trow = tarray(tobj)
tmap = tarray(trow)
taction = baseType("taction")
tx = baseType("tx")
ty = baseType("ty")
tboolop = baseType("tboolop")


class GameObject():
    def __init__(self, obj, x, y):
        self.obj = obj
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        return self.obj == other.obj


def _if(c): return lambda t: lambda f: t if c else f
def _and(x): return lambda y: x and y
def _not(x): return not x
def _id(x): return x
def _or(x): return lambda y: x or y
def _get(m): return lambda x: lambda y: GameObject(m[x][y], x, y)
def _eq(x): return lambda y: x == y
def _eq_obj(x: GameObject): return lambda y: x.obj == y
def _greater_than(x): return lambda y: x > y
def _get_obj_x(obj: GameObject): return obj.x
def _get_obj_y(obj: GameObject): return obj.y
def _get_game_obj(obj: GameObject): return obj.obj
def _check_game_obj(obj: GameObject): return obj.obj


action_mapping = {
    'n': 'no-op', 'l': 'left', 'u': 'up', 'r': 'right', 'd': 'down', 'f': 'fire'
}

channels = {
    'asterix': {
        'empty': -1,
        'player': 0,
        'enemy': 1,
        'trail': 2,
        'gold': 3
    },
    'breakout': {
        'empty': -1,
        'paddle': 0,
        'ball': 1,
        'trail': 2,
        'brick': 3,
    },
    'freeway': {
        'empty': -1,
        'chicken': 0,
        'car': 1,
        'speed1': 2,
        'speed2': 3,
        'speed3': 4,
        'speed4': 5,
        'speed5': 6,
    },
    'space_invaders': {
        'empty': -1,
        'cannon': 0,
        'alien': 1,
        'alien_left': 2,
        'alien_right': 3,
        'friendly_bullet': 4,
        'enemy_bullet': 5
    },
    'seaquest': {
        'empty': -1,
        'sub_front': 0,
        'sub_back': 1,
        'friendly_bullet': 2,
        'trail': 3,
        'enemy_bullet': 4,
        'enemy_fish': 5,
        'enemy_sub': 6,
        'oxygen_guage': 7,
        'diver_guage': 8,
        'diver': 9
    }
}

actions = {
    'breakout': ['n', 'l', 'r'],
    'asterix':  ['n', 'l', 'u', 'r', 'd'],
    'freeway': ['n', 'u', 'd'],
    'space_invaders': ['n', 'l', 'r', 'f'],
    'seaquest': ['n', 'l', 'u', 'r', 'd', 'f']
}


def basePrimitives(env_name):
    action_keys = list(action_mapping.keys())
    return [
        Primitive(str(j), tint, j) for j in range(10)
    ] + [
        Primitive(f'{action_mapping[action]}-action', taction, action_keys.index(action)) for action in ['n', 'l', 'u', 'r', 'd', 'f']
    ] + [
        # we need to add 1 as 0 zero indicates empty grid positions so we always start with 1
        Primitive(f'{obj}-obj', tobj, idx + 1) for obj, idx in channels[env_name].items()
    ] + [
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("eq-obj?", arrow(tgameobj, tobj, tbool), _eq_obj),
        Primitive("get", arrow(tmap, tint, tint, tgameobj), _get),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        Primitive("gt-x?", arrow(tx, tx, tbool), _greater_than),
        Primitive("gt-y?", arrow(ty, ty, tbool), _greater_than),
        Primitive("eq-x?", arrow(tx, tx, tbool), _eq),
        Primitive("eq-y?", arrow(ty, ty, tbool), _eq),
        Primitive("get-obj-x", arrow(tgameobj, tx), _get_obj_x),
        Primitive("get-obj-y", arrow(tgameobj, ty), _get_obj_y),
        Primitive("get-game-obj", arrow(tgameobj, tobj), _get_game_obj),
    ]


def basePrimitivesBoolOp(env_name):
    action_keys = list(action_mapping.keys())
    return [
        Primitive(str(j), tint, j) for j in range(10)
    ] + [
        Primitive(f'{action_mapping[action]}-action', taction, action_keys.index(action)) for action in actions[env_name]
    ] + [
        # we need to add 1 as 0 zero indicates empty grid positions so we always start with 1
        Primitive(f'{obj}-obj', tobj, idx + 1) for obj, idx in channels[env_name].items()
    ] + [
        Primitive("if", arrow(tboolop, t0, t0, t0), _if),
        Primitive("eq-obj?", arrow(tgameobj, tobj, tbool), _eq_obj),
        Primitive("get", arrow(tmap, tint, tint, tgameobj), _get),
        Primitive("not", arrow(tbool, tbool), _not),
        # Primitive("id", arrow(tbool, tboolop), _id),
        Primitive("and", arrow(tbool, tbool, tboolop), _and),
        Primitive("or", arrow(tbool, tbool, tboolop), _or),
        Primitive("gt-x?", arrow(tx, tx, tbool), _greater_than),
        Primitive("gt-y?", arrow(ty, ty, tbool), _greater_than),
        Primitive("eq-x?", arrow(tx, tx, tbool), _eq),
        Primitive("eq-y?", arrow(ty, ty, tbool), _eq),
        Primitive("get-obj-x", arrow(tgameobj, tx), _get_obj_x),
        Primitive("get-obj-y", arrow(tgameobj, ty), _get_obj_y),
        Primitive("get-game-obj", arrow(tgameobj, tobj), _get_game_obj),
    ]


def basePrimitivesCheckObject(env_name):
    action_keys = list(action_mapping.keys())
    return [
        Primitive(str(j), tint, j) for j in range(10)
    ] + [
        Primitive(f'{action_mapping[action]}-action', taction, action_keys.index(action)) for action in actions[env_name]
    ] + [
        # we need to add 1 as 0 zero indicates empty grid positions so we always start with 1
        Primitive(f'{obj}-obj', tobj, idx + 1) for obj, idx in channels[env_name].items()
    ] + [
        Primitive("if", arrow(tboolop, t0, t0, t0), _if),
        Primitive("eq-obj?", arrow(tgameobj, tobj, tbool), _eq_obj),
        Primitive("get", arrow(tmap, tint, tint, tgameobj), _get),
        Primitive("not", arrow(tbool, tbool), _not),
        # Primitive("id", arrow(tbool, tboolop), _id),
        Primitive("and", arrow(tbool, tbool, tboolop), _and),
        Primitive("or", arrow(tbool, tbool, tboolop), _or),
        Primitive("gt-x?", arrow(tx, tx, tbool), _greater_than),
        Primitive("gt-y?", arrow(ty, ty, tbool), _greater_than),
        Primitive("eq-x?", arrow(tx, tx, tbool), _eq),
        Primitive("eq-y?", arrow(ty, ty, tbool), _eq),
        Primitive("get-obj-x", arrow(tgameobj, tx), _get_obj_x),
        Primitive("get-obj-y", arrow(tgameobj, ty), _get_obj_y),
        Primitive("get-game-obj", arrow(tgameobj, tobj), _get_game_obj),
    ]
