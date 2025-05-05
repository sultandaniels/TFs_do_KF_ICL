import json
import os
import argparse
import graphviz

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--in_path", "-i", type=str, required=True)
    parser.add_argument(
        "--out_path", "-o",
        type=str,
        default="vis/one_after_edges",
        help="Output path without extension (Graphviz will append .<format>)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["png", "svg", "jpg", "pdf", "dot"],
        default="png",
        help="Graphviz output format"
    )
    parser.add_argument("--constants", "-c", default=[], nargs="+")
    parser.add_argument("--no-sanitize", "-ns", action="store_true")
    
    args = parser.parse_args()
    args.constants = [c for c in args.constants if c.lower() != "[none]"]
    return args

def get_circuit_colors(
    tok_embeds_color="grey",
    pos_embeds_color="lightsteelblue",
    mlp_color="cadetblue2",
    q_color="plum1",
    k_color="lightpink",
    v_color="khaki1",
    o_color="darkslategray3",
    resid_post_color="azure"
):
    def decide_color(node_name):
        n = node_name.lower()
        if "embed" in n:
            return pos_embeds_color if "pos" in n else tok_embeds_color
        if n == "output":
            return resid_post_color
        if n.startswith("m"):
            return mlp_color
        if n.endswith(".q"):
            return q_color
        if n.endswith(".k"):
            return k_color
        if n.endswith(".v"):
            return v_color
        return o_color
    return decide_color

def sanitize_edges(edges):
    # (same as before)
    new_edges_ = set()
    for edge in edges:
        if edge[0][0] == "a" and edge[0][-1] not in ["q","k","v"]:
            new_edges_.add(edge[0])
    for to in new_edges_:
        for suffix in [".q",".k",".v"]:
            edges.append((to+suffix, to))
    while True:
        orig_len = len(edges)
        froms = {e[0] for e in edges}
        tos   = {e[1] for e in edges if e[1] != "resid_post"}
        banned = tos - froms
        edges = [e for e in edges if e[1] not in banned]
        qkv = {e[1] for e in edges if e[1].endswith((".q",".k",".v"))}
        edges = [
            e for e in edges
            if not (
                (e[0].endswith((".q",".k",".v")) and e[0] not in qkv)
            )
        ]
        if len(edges) == orig_len:
            break
    return edges

def rename(name):
    if isinstance(name, (list,tuple)):
        return [rename(n) for n in name]
    name_l = name.lower()
    if "embeds" in name_l:
        return "Embeddings"
    if name_l == "resid_post":
        return "Output"
    if name_l.startswith("m"):
        return f"MLP {int(name_l[1:])}"
    if any(name_l.endswith(s) for s in [".q",".k",".v"]):
        layer, head = map(int, name_l.replace("."," ").split()[0][1:]), int(name_l.split(".")[1][1:])
        typ = name_l[-1].upper()
        return f"Head {layer}.{head}.{typ}"
    # else must be O
    layer, head = map(int, name_l.replace("."," ").split()[0][1:]), int(name_l.split(".")[1][1:])
    return f"Head {layer}.{head}.O"

def main():
    args = parse_args()
    edges = json.load(open(args.in_path))
    if not args.no_sanitize:
        edges = sanitize_edges(edges)
    edges = [tuple(rename(e) for e in edge) for edge in edges]

    # pick up every node
    constants = args.constants
    nodes = set(constants)
    for u,v in edges:
        nodes.add(u); nodes.add(v)

    # build graph
    coloring_fn = get_circuit_colors()
    g = graphviz.Digraph(format=args.format,
        graph_attr={"nodesep":"0.02","ranksep":"0.02","ratio":"1:6"},
        node_attr={"shape":"box","style":"rounded,filled"}
    )

    # add nodes
    for n in nodes:
        g.node(n, color="black", fillcolor=coloring_fn(n))

    # add edges
    const_color = "gray66"
    for u,v in edges:
        g.edge(u, v, color=coloring_fn(u))
    # connect constants
    for n in nodes - set(constants):
        for c in constants:
            g.edge(c, n, color=const_color)

    # render
    out_base = os.path.splitext(args.out_path)[0]
    print(g.source)
    output_path = g.render(filename=out_base, cleanup=True)
    print(f"Wrote graph to {output_path}")

if __name__ == "__main__":
    main()
