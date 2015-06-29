import heapq

import numpy as np

from chainer import function
from chainer.functions import basic_math
from chainer import variable


class DotNode(object):
    def _shape(self):
        if isinstance(self.node, variable.Variable):
            return "oval"
        elif isinstance(self.node, function.Split):
            return "hexagon"
        else:
            return "box"

    def _label(self):
        if isinstance(self.node, variable.Variable):
            if self.node.data.shape == tuple():
                return str(self.node.data.dtype)
            return "%s, %s" % (str(self.node.data.shape),
                               str(self.node.data.dtype))
        elif isinstance(self.node, basic_math.Add):
            return "+"
        elif isinstance(self.node, basic_math.AddConstant):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "+ %s" % str(value)
            elif isinstance(value, variable.Variable):
                return "+ %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, basic_math.Sub):
            return "-"
        elif isinstance(self.node, basic_math.SubFromConstant):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "* (-1) + %s" % str(value)
            elif isinstance(value, variable.Variable):
                return "* (-1) + %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, basic_math.Mul):
            return "*"
        elif isinstance(self.node, basic_math.MulConstant):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "* %s" % str(value)
            elif isinstance(value, variable.Variable):
                return "* %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, basic_math.Div):
            return "/"
        elif isinstance(self.node, basic_math.DivFromConstant):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "/ %s" % str(value)
            elif isinstance(value, variable.Variable):
                return "/ %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, basic_math.PowVarVar):
            return "**"
        elif isinstance(self.node, basic_math.PowVarConst):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "** %s" % str(value)
            elif isinstance(value, variable.Variable):
                return "** %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, basic_math.PowConstVar):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "%s **" % str(value)
            elif isinstance(value, variable.Variable):
                return "%s **" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, basic_math.Exp):
            return "exp"
        elif isinstance(self.node, basic_math.Log):
            return "log"
        else:
            return str(type(self.node))

    def __init__(self, node):
        self.node = node
        self.id_ = id(node)
        self.attribute = {
            "label": self._label(),
            "shape": self._shape()
        }

    def __str__(self):
        attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                      in self.attribute.items()]
        return "%s [%s];" % (self.id_, ",".join(attributes))


class ComputationalGraph(object):
    def __init__(self, edges):
        self.edges = edges

    def _to_dot(self):
        ret = "digraph graphname{"
        for edge in self.edges:
            head, tail = edge
            assert (isinstance(head, variable.Variable)
                    and isinstance(tail, function.Function)) or \
                   (isinstance(head, function.Function)
                    and isinstance(tail, variable.Variable))
            head_node = DotNode(head)
            tail_node = DotNode(tail)
            ret += str(head_node)
            ret += str(tail_node)
            ret += "%s -> %s;" % (head_node.id_, tail_node.id_)
        ret += "}"
        return ret

    def __str__(self):
        return self._to_dot()

    def __len__(self):
        return len(self.edges)

    def __contains__(self, e):
        return e in self.edges


def computational_graph(outputs, remove_split=False):
    cands = []
    seen_edges = set()

    def _add_cand(cand):
        heapq.heappush(cands, (-cand.rank, len(seen_edges), cand))

    for o in outputs:
        heapq.heappush(cands, (-o.rank, len(seen_edges), o))

    while cands:
        _, _, cand = heapq.heappop(cands)
        if isinstance(cand, variable.Variable):
            creator = cand.creator
            if remove_split and isinstance(creator, function.Split):
                # assume that function.Split has only one input
                next_cand = creator.inputs[0]
                _add_cand(next_cand)
                continue
            if creator is not None and (creator, cand) not in seen_edges:
                _add_cand(creator)
                seen_edges.add((creator, cand))
        elif isinstance(cand, function.Function):
            if remove_split and isinstance(cand, function.Split):
                next_cand = creator.inputs[0]
                _add_cand(next_cand)
                continue
            for input_ in cand.inputs:
                if input_ != cand and (input_, cand) not in seen_edges:
                    creator = input_.creator
                    if remove_split and\
                       creator is not None and\
                       isinstance(creator, function.Split):
                        input_ = creator.inputs[0]
                    _add_cand(input_)
                    seen_edges.add((input_, cand))
    return ComputationalGraph(seen_edges)
