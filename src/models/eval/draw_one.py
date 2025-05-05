import json
from graphviz import Digraph
from graphviz import Source

# Edge list provided by the user
edges = [
    ["a0.h2", "a1.h3.k"], ["a0.h2", "a1.h2.v"], ["a0.h2", "a1.h3.v"],
    ["a0.h3", "a1.h2.k"], ["a0.h3", "a1.h5.v"], ["a0.h5", "a1.h2.k"],
    ["a0.h5", "a1.h5.v"], ["a0.h7", "a1.h7.k"], ["m0", "a1.h2.k"],
    ["m0", "a1.h5.k"], ["m0", "m1"], ["a1.h1", "m1"], ["a1.h2", "m1"],
    ["a1.h5", "m1"], ["a1.h7", "m1"], ["a0.h2", "a2.h4.k"],
    ["a0.h2", "a2.h5.v"], ["a0.h3", "a2.h3.v"], ["a0.h4", "a2.h4.k"],
    ["a0.h7", "a2.h3.k"], ["a0.h7", "a2.h7.k"], ["m0", "a2.h3.k"],
    ["m0", "a2.h5.k"], ["m0", "a2.h7.k"], ["m0", "a2.h5.v"], ["m0", "a2.h7.v"],
    ["a1.h1", "a2.h4.k"], ["a1.h1", "a2.h3.v"], ["a1.h1", "a2.h7.v"],
    ["a1.h2", "a2.h3.k"], ["a1.h2", "a2.h4.k"], ["a1.h2", "a2.h5.k"],
    ["a1.h2", "a2.h7.k"], ["a1.h2", "a2.h0.v"], ["a1.h2", "a2.h1.v"],
    ["a1.h2", "a2.h3.v"], ["a1.h2", "a2.h5.v"], ["a1.h3", "a2.h4.k"],
    ["a1.h3", "a2.h7.k"], ["a1.h5", "a2.h4.q"], ["a1.h5", "a2.h7.v"],
    ["a1.h7", "a2.h5.v"], ["m1", "a2.h4.q"], ["m1", "a2.h3.k"],
    ["m1", "a2.h5.k"], ["m1", "a2.h7.k"], ["m1", "a2.h3.v"], ["m1", "a2.h5.v"],
    ["m1", "a2.h7.v"], ["a0.h5", "m2"], ["m0", "m2"], ["a1.h5", "m2"],
    ["a1.h7", "m2"], ["m1", "m2"], ["a2.h1", "m2"], ["a2.h2", "m2"],
    ["a2.h3", "m2"], ["a2.h5", "m2"], ["a2.h7", "m2"], ["a0.h2", "a3.h3.v"],
    ["m0", "a3.h3.k"], ["m0", "a3.h7.k"], ["m0", "a3.h3.v"], ["m0", "a3.h7.v"],
    ["a1.h1", "a3.h7.v"], ["a1.h2", "a3.h3.v"], ["a1.h2", "a3.h7.v"],
    ["a1.h3", "a3.h7.v"], ["a1.h5", "a3.h3.k"], ["a1.h5", "a3.h3.v"],
    ["a1.h7", "a3.h7.v"], ["m1", "a3.h3.k"], ["m1", "a3.h7.k"],
    ["m1", "a3.h3.v"], ["m1", "a3.h7.v"], ["a2.h0", "a3.h7.v"],
    ["a2.h1", "a3.h3.k"], ["a2.h1", "a3.h7.k"], ["a2.h1", "a3.h7.v"],
    ["a2.h2", "a3.h3.k"], ["a2.h2", "a3.h7.k"], ["a2.h3", "a3.h3.v"],
    ["a2.h3", "a3.h7.v"], ["a2.h4", "a3.h3.k"], ["a2.h4", "a3.h7.k"],
    ["a2.h4", "a3.h3.v"], ["a2.h5", "a3.h3.v"], ["a2.h5", "a3.h7.v"],
    ["a2.h7", "a3.h3.v"], ["a2.h7", "a3.h7.v"], ["m2", "a3.h3.k"],
    ["m2", "a3.h3.v"], ["m2", "a3.h7.v"], ["a3.h3", "m3"], ["a3.h7", "m3"],
    ["a3.h3", "m4"], ["a3.h7", "m4"], ["m3", "m4"], ["a4.h6", "m4"],
    ["a2.h4", "m5"], ["a2.h7", "m5"], ["a3.h3", "m5"], ["a3.h7", "m5"],
    ["m3", "m5"], ["m4", "m5"], ["a5.h6", "m5"], ["a2.h4", "m6"],
    ["a3.h3", "m6"], ["a3.h7", "m6"], ["m3", "m6"], ["m4", "m6"],
    ["m5", "m6"], ["a2.h1", "m7"], ["a2.h4", "m7"], ["a3.h3", "m7"],
    ["a3.h7", "m7"], ["m3", "m7"], ["m4", "m7"], ["a5.h6", "m7"],
    ["m5", "m7"], ["m6", "m7"], ["a1.h2", "m8"], ["a3.h3", "m8"],
    ["a3.h7", "m8"], ["m4", "m8"], ["m5", "m8"], ["m6", "m8"], ["m7", "m8"],
    ["a1.h2", "m9"], ["a3.h3", "m9"], ["a3.h7", "m9"], ["m3", "m9"],
    ["m4", "m9"], ["m5", "m9"], ["m6", "m9"], ["m7", "m9"], ["m8", "m9"],
    ["a2.h7", "m10"], ["m2", "m10"], ["a3.h3", "m10"], ["a3.h7", "m10"],
    ["m3", "m10"], ["m4", "m10"], ["m5", "m10"], ["m6", "m10"], ["m7", "m10"],
    ["m8", "m10"], ["m9", "m10"], ["a2.h7", "m11"], ["a3.h3", "m11"],
    ["a3.h7", "m11"], ["m3", "m11"], ["m4", "m11"], ["m5", "m11"],
    ["m6", "m11"], ["m7", "m11"], ["m8", "m11"], ["m9", "m11"], ["m10", "m11"],
    ["a2.h1", "resid_post"], ["a2.h7", "resid_post"], ["a3.h2", "resid_post"],
    ["a3.h3", "resid_post"], ["a3.h7", "resid_post"], ["m3", "resid_post"],
    ["m4", "resid_post"], ["m5", "resid_post"], ["m6", "resid_post"],
    ["m7", "resid_post"], ["m8", "resid_post"], ["m9", "resid_post"],
    ["m10", "resid_post"], ["m11", "resid_post"]
]

# Build the directed graph
dot = Digraph(comment='Residual Stream', format='png')
dot.attr(rankdir='LR')  # Layout from left to right

# Add nodes
for src, dst in edges:
    dot.node(src, src)
    dot.node(dst, dst)

# Add edges
for src, dst in edges:
    dot.edge(src, dst)

# Render inline
src = Source(dot.source, format='png')
src
