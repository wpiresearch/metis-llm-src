from itertools import pairwise

import networkx as nx
from bokeh.models import (
    Arrow,
    Circle,
    CustomJS,
    GraphRenderer,
    HoverTool,
    MultiLine,
    StaticLayoutProvider,
    VeeHead,
)
from bokeh.plotting import figure

from system_two_model import NodeResponse, NodeRole, SystemTwoResponse


def create_system_two_node_graph(
    system_two_state: SystemTwoResponse,
) -> tuple[figure, list[NodeResponse]]:
    """Create the Bokeh graph"""
    # Create a graph with multiple nodes and edges
    G = nx.Graph()
    system_two_nodes: list[SystemTwoResponse] = []
    has_synthesizer_role = False
    for node_response in system_two_state.node_responses:
        if not has_synthesizer_role and node_response.node_role == NodeRole.Synthesizer:
            has_synthesizer_role = True
        system_two_nodes.append(node_response)
    if not has_synthesizer_role:
        system_two_nodes.append(
            NodeResponse(
                node_role=NodeRole.Synthesizer,
                node_response=system_two_state.system_two_response,
                node_msv=system_two_state.metacognitive_vector,
            )
        )
    system_two_edges = list(pairwise(system_two_nodes))
    G.add_nodes_from(system_two_nodes)
    G.add_edges_from(system_two_edges)
    # G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])

    # Create the Bokeh plot
    plot = figure(
        width=600,
        height=400,
        x_range=(-1, len(system_two_nodes) + 0.025),
        y_range=(-3, 3),
        title="System Two Internal Roles",
        tools="tap,pan,wheel_zoom,reset",
    )

    # Create graph renderer
    graph = GraphRenderer()

    node_to_index = {node: i for i, node in enumerate(system_two_nodes)}

    # Node data
    # node_indices = list(G.nodes())
    graph.node_renderer.data_source.data = dict(
        index=list(range(len(system_two_nodes))),
        name=[
            node.node_role.replace("_", " ").title() for node in system_two_nodes
        ],  # Extract string from Pydantic model
        node_response=[
            node.node_response for node in system_two_nodes
        ],  # Extract string
        # node_type=[node.type for node in system_two_nodes],  # Extract string
        # value=[node.value for node in system_two_nodes],  # Extract float
        node_id=[
            idx for idx, node in enumerate(system_two_nodes)
        ],  # Extract string ID for HTMX
    )
    # graph.node_renderer.data_source.data = dict(
    #     index=node_indices,
    #     name=[f"Node {i}" for i in node_indices],
    #     description=[f"This is node {i}" for i in node_indices],
    #     node_id=node_indices,
    # )

    # Edge data
    edge_start = [edge[0].model_dump() for edge in G.edges()]
    edge_end = [edge[1].model_dump() for edge in G.edges()]
    graph.edge_renderer.data_source.data = dict(start=edge_start, end=edge_end)

    # Use spring layout for automatic positioning
    # graph_layout_nodes = nx.spring_layout(G, scale=2, seed=42)
    # # graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    # # Convert node object keys to integer indices for Bokeh
    # graph_layout = {
    #     node_to_index[node]: pos for node, pos in graph_layout_nodes.items()
    # }

    # use linear layout for now, to match the system 2 process
    graph_layout = {}
    x_spacing = 1.0  # Adjust this to control horizontal spacing between nodes

    for i, node in enumerate(system_two_nodes):
        node_idx = node_to_index[node]
        graph_layout[node_idx] = (
            i * x_spacing,
            0,
        )  # All nodes at y=0, spaced horizontally
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # Style nodes
    circle_radius = 0.25

    graph.node_renderer.glyph = Circle(
        radius=circle_radius, fill_color="lightblue", line_color="navy", line_width=2
    )
    graph.node_renderer.selection_glyph = Circle(
        radius=circle_radius, fill_color="yellow", line_color="navy", line_width=2
    )
    graph.node_renderer.hover_glyph = Circle(
        radius=0.3, fill_color="lightgreen", line_color="navy", line_width=2
    )

    # Style edges with arrows
    graph.edge_renderer.glyph = MultiLine(
        line_color="gray", line_alpha=0.8, line_width=2
    )

    # Add arrows for each edge (aligned to circle edges)
    circle_radius = 0.25
    for edge in system_two_edges:
        start_idx = node_to_index[edge[0]]
        end_idx = node_to_index[edge[1]]
        start_pos = graph_layout[start_idx]
        end_pos = graph_layout[end_idx]

        # Calculate direction vector
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = (dx**2 + dy**2) ** 0.5

        # Normalize and offset by circle radius
        if distance > 0:
            dx_norm = dx / distance
            dy_norm = dy / distance

            x_start = start_pos[0] + dx_norm * circle_radius
            y_start = start_pos[1] + dy_norm * circle_radius
            x_end = end_pos[0] - dx_norm * circle_radius
            y_end = end_pos[1] - dy_norm * circle_radius

            arrow = Arrow(
                end=VeeHead(size=15, fill_color="gray"),
                x_start=x_start,
                y_start=y_start,
                x_end=x_end,
                y_end=y_end,
                line_color="gray",
                line_width=2,
            )
            plot.add_layout(arrow)

    # Add hover tooltip with complex object properties
    hover = HoverTool(
        renderers=[graph.node_renderer],
        tooltips=[
            ("Name", "@name"),
            # ("Type", "@node_type"),
            # ("Description", "@description"),
            # ("Value", "@value"),
            ("ID", "@node_id"),
        ],
        sort_by="distance",  # Options: "distance", "value", or field name like "@node_id"
    )
    plot.add_tools(hover)

    # Add tap/click event
    tap_callback = CustomJS(
        args=dict(source=graph.node_renderer.data_source),
        code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            const nodeId = source.data['node_id'][indices[0]];
            htmx.ajax('GET', '/node/' + nodeId, {
                target: '#node-info',
                swap: 'innerHTML'
            });
        }
    """,
    )

    graph.node_renderer.data_source.selected.js_on_change("indices", tap_callback)

    # Add graph to plot
    plot.renderers.append(graph)

    return plot, system_two_nodes
