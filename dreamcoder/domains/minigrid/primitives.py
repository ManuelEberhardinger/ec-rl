from dreamcoder.program import Primitive, Program
from dreamcoder.grammar import Grammar
from dreamcoder.type import baseType, tint, tbool, arrow, t0, TypeConstructor
from gym_minigrid.minigrid import MiniGridEnv, IDX_TO_OBJECT


idx_to_action = {int(a): a.name for a in MiniGridEnv.Actions}


def tarray(t): return TypeConstructor("array", [t])


class GameObject():
    def __init__(self, obj, x, y):
        self.obj = obj
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        return self.obj == other.obj


tdirection = baseType("tdirection")  # the objects from the gym minigrid env
tobj = baseType("tobj")  # the objects from the gym minigrid env
tgameobj = baseType("tgameobj")  # the objects from the gym minigrid env
trow = tarray(tobj)
tmap = tarray(trow)
taction = baseType("taction")


def _loop(n):
    def f(start, body):
        if start >= n - 1:
            return  # ??
        return body()(f(start + 1, body))
    return lambda b: f(0, b)


def _if(c): return lambda t: lambda f: t if c else f
def _and(x): return lambda y: x and y
def _not(x): return not x
def _or(x): return lambda y: x or y
def _get(m): return lambda x: lambda y: GameObject(m[x][y], x, y)
# def _get(m): return lambda x: lambda y: m[x][y]
def _eq(x): return lambda y: x == y
def _eq_direction(x): return lambda y: x == y
def _eq_obj(x: GameObject): return lambda y: x.obj == y
# def _eq_obj(x): return lambda y: x == y
def _get_action(idx): return idx_to_action[idx]
def _get_game_obj(obj: GameObject): return obj.obj


def basePrimitives():
    return [
        Primitive(str(j), tint, j) for j in range(5)
    ] + [
        Primitive(f'm-{action}-action', taction, idx) for idx, action in idx_to_action.items() if action in ['left', 'right', 'forward']
    ] + [
        Primitive(f'direction-{idx}', tdirection, idx) for idx in range(4)
    ] + [
        Primitive(f'm-{obj}-obj', tobj, idx) for idx, obj in IDX_TO_OBJECT.items() if obj in ['empty', 'wall', 'goal']
    ] + [
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("eq-direction?", arrow(tdirection,
                  tdirection, tbool), _eq_direction),
        Primitive("eq-obj?", arrow(tgameobj, tobj, tbool), _eq_obj),
        Primitive("get", arrow(tmap, tint, tint, tgameobj), _get),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        Primitive("get-game-obj", arrow(tgameobj, tobj), _get_game_obj),
        # Primitive("loop", arrow(tint, t0, t0), _loop),
        # Primitive("eq?", arrow(tint, tint, tbool), _eq),
    ]
