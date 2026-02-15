"""
role_visualization.py

Bokeh visualizations for the five-role dialectical pipeline.
Extends the existing app_graph.py pattern — uses the same Bokeh
infrastructure, same styling conventions, same HTMX integration.

Three visualizations:
1. Pipeline graph: 5 role nodes with arrows (extends existing node graph)
2. Fitness matrix heatmap: agents × roles with assignment highlighted
3. MSV evolution chart: how MSV dimensions change across pipeline stages

All functions return Bokeh figure objects compatible with the existing
bokeh.embed.components() call pattern in app.py.
"""

import math
from itertools import pairwise

import numpy as np
from bokeh.embed import components
from bokeh.models import (
    Arrow,
    BasicTicker,
    Circle,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    GraphRenderer,
    HoverTool,
    Label,
    LabelSet,
    LinearColorMapper,
    MultiLine,
    StaticLayoutProvider,
    VeeHead,
)
from bokeh.palettes import Blues9, RdYlGn11
from bokeh.plotting import figure
from bokeh.transform import transform

from system_two_model import NodeResponse, NodeRole


# ============================================================
# 1. PIPELINE GRAPH (extends existing app_graph.py pattern)
# ============================================================

# Role-specific colors for visual differentiation
ROLE_COLORS = {
    NodeRole.Domain_Expert: "#4A90D9",   # Blue
    NodeRole.Critic: "#E74C3C",          # Red
    NodeRole.Evaluator: "#F39C12",       # Orange
    NodeRole.Synthesizer: "#27AE60",     # Green
    NodeRole.Generalist: "#8E44AD",      # Purple
}

ROLE_SHORT_LABELS = {
    NodeRole.Domain_Expert: "Expert",
    NodeRole.Critic: "Critic",
    NodeRole.Evaluator: "Evaluator",
    NodeRole.Synthesizer: "Synthesizer",
    NodeRole.Generalist: "Generalist",
}


def create_role_pipeline_graph(
    role_responses: list[NodeResponse],
    assignment: dict[str, int] | None = None,
    generalist_annotation: str | None = None,
) -> tuple[figure, list[NodeResponse]]:
    """
    Create the Bokeh pipeline graph for the five-role architecture.

    This extends create_system_two_node_graph() from app_graph.py with:
    - 5 nodes (one per role) instead of 2-3
    - Color-coded nodes by role
    - Agent assignment labels
    - Generalist shown as a separate node (branching off main chain)
    - Hover shows MSV summary and response preview

    Args:
        role_responses: List of NodeResponse from the dialectical pipeline
        assignment: Dict mapping role -> agent_index (from Hungarian)
        generalist_annotation: If present, Generalist was triggered

    Returns:
        (plot, node_list) — same signature as existing create_system_two_node_graph()
    """
    # Separate dialectical chain from Generalist
    chain_roles = [NodeRole.Domain_Expert, NodeRole.Critic,
                   NodeRole.Evaluator, NodeRole.Synthesizer]
    chain_responses = [r for r in role_responses if r.node_role in chain_roles]

    # Sort chain_responses by dialectical order
    role_order = {role: i for i, role in enumerate(chain_roles)}
    chain_responses.sort(key=lambda r: role_order.get(r.node_role, 99))

    # Build node list: chain nodes + optional Generalist
    all_nodes = list(chain_responses)

    # Create the plot — wider than existing to fit 5 nodes
    n_chain = len(chain_responses)
    plot = figure(
        width=800,
        height=450,
        x_range=(-1, n_chain + 0.5),
        y_range=(-3, 3),
        title="Five-Role Dialectical Pipeline",
        tools="tap,pan,wheel_zoom,reset",
    )

    # --- Layout: chain nodes in a horizontal line ---
    graph = GraphRenderer()
    node_to_index = {node: i for i, node in enumerate(all_nodes)}

    x_spacing = 1.0
    graph_layout = {}
    for i, node in enumerate(all_nodes):
        idx = node_to_index[node]
        if i < n_chain:
            # Chain nodes: horizontal line at y=0
            graph_layout[idx] = (i * x_spacing, 0)
        else:
            # Generalist: above the Evaluator (index 2) position
            graph_layout[idx] = (2 * x_spacing, 1.5)

    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # --- Node data ---
    node_colors = []
    node_labels = []
    agent_labels = []
    msv_summaries = []
    response_previews = []

    for node in all_nodes:
        role = node.node_role
        node_colors.append(ROLE_COLORS.get(role, "lightblue"))
        node_labels.append(ROLE_SHORT_LABELS.get(role, role.replace("_", " ").title()))

        # Agent assignment label
        if assignment and role in assignment:
            agent_labels.append(f"Agent {assignment[role]}")
        else:
            agent_labels.append("")

        # MSV summary for hover
        msv = node.node_msv
        msv_summaries.append(
            f"ER={msv.emotional_response.calculated_value}, "
            f"CE={msv.correctness.calculated_value}, "
            f"EM={msv.experiential_matching.calculated_value}, "
            f"CI={msv.conflict_information.calculated_value}, "
            f"PI={msv.problem_importance.calculated_value}"
        )

        # Response preview (first 200 chars)
        preview = node.node_response[:200]
        if len(node.node_response) > 200:
            preview += "..."
        response_previews.append(preview)

    graph.node_renderer.data_source.data = dict(
        index=list(range(len(all_nodes))),
        name=node_labels,
        color=node_colors,
        agent=agent_labels,
        msv=msv_summaries,
        response_preview=response_previews,
        node_id=list(range(len(all_nodes))),
    )

    # --- Node styling (color-coded by role) ---
    circle_radius = 0.25
    graph.node_renderer.glyph = Circle(
        radius=circle_radius, fill_color="color", line_color="navy", line_width=2
    )
    graph.node_renderer.selection_glyph = Circle(
        radius=circle_radius, fill_color="yellow", line_color="navy", line_width=2
    )
    graph.node_renderer.hover_glyph = Circle(
        radius=0.3, fill_color="lightgreen", line_color="navy", line_width=2
    )

    # --- Edges: chain is sequential ---
    chain_edges = list(pairwise(chain_responses))
    edge_start = []
    edge_end = []
    for e in chain_edges:
        edge_start.append(node_to_index[e[0]])
        edge_end.append(node_to_index[e[1]])

    graph.edge_renderer.data_source.data = dict(start=edge_start, end=edge_end)
    graph.edge_renderer.glyph = MultiLine(
        line_color="gray", line_alpha=0.8, line_width=2
    )

    # --- Arrows along edges ---
    for edge in chain_edges:
        start_idx = node_to_index[edge[0]]
        end_idx = node_to_index[edge[1]]
        start_pos = graph_layout[start_idx]
        end_pos = graph_layout[end_idx]

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = (dx**2 + dy**2) ** 0.5

        if distance > 0:
            dx_norm = dx / distance
            dy_norm = dy / distance

            arrow = Arrow(
                end=VeeHead(size=15, fill_color="gray"),
                x_start=start_pos[0] + dx_norm * circle_radius,
                y_start=start_pos[1] + dy_norm * circle_radius,
                x_end=end_pos[0] - dx_norm * circle_radius,
                y_end=end_pos[1] - dy_norm * circle_radius,
                line_color="gray",
                line_width=2,
            )
            plot.add_layout(arrow)

    # --- Role labels below nodes ---
    label_source = ColumnDataSource(data=dict(
        x=[graph_layout[node_to_index[n]][0] for n in all_nodes],
        y=[graph_layout[node_to_index[n]][1] - 0.5 for n in all_nodes],
        text=node_labels,
    ))
    labels = LabelSet(
        x="x", y="y", text="text", source=label_source,
        text_align="center", text_baseline="top",
        text_font_size="10pt", text_font_style="bold",
    )
    plot.add_layout(labels)

    # --- Agent assignment labels above nodes ---
    if assignment:
        agent_source = ColumnDataSource(data=dict(
            x=[graph_layout[node_to_index[n]][0] for n in all_nodes],
            y=[graph_layout[node_to_index[n]][1] + 0.45 for n in all_nodes],
            text=agent_labels,
        ))
        agent_label_set = LabelSet(
            x="x", y="y", text="text", source=agent_source,
            text_align="center", text_baseline="bottom",
            text_font_size="8pt", text_color="gray",
        )
        plot.add_layout(agent_label_set)

    # --- Hover tooltip ---
    hover = HoverTool(
        renderers=[graph.node_renderer],
        tooltips=[
            ("Role", "@name"),
            ("Agent", "@agent"),
            ("MSV", "@msv"),
            ("Response", "@response_preview"),
        ],
    )
    plot.add_tools(hover)

    # --- Tap/click callback (same pattern as existing) ---
    tap_callback = CustomJS(
        args=dict(source=graph.node_renderer.data_source),
        code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            const nodeId = source.data['node_id'][indices[0]];
            htmx.ajax('GET', '/role_node/' + nodeId, {
                target: '#node-info',
                swap: 'innerHTML'
            });
        }
    """,
    )
    graph.node_renderer.data_source.selected.js_on_change("indices", tap_callback)

    # Hide axes (same as existing)
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.xgrid.visible = False
    plot.ygrid.visible = False

    plot.renderers.append(graph)
    return plot, all_nodes


# ============================================================
# 2. FITNESS MATRIX HEATMAP
# ============================================================

def create_fitness_heatmap(
    matrix: np.ndarray,
    assignment: dict[str, int],
    title: str = "Fitness Matrix",
    role_names: list[str] | None = None,
    agent_model_names: list[str] | None = None,
) -> figure:
    """
    Create a heatmap showing the agent × role fitness matrix.

    Assigned cells are highlighted with a border. This lets the viewer
    see both the fitness landscape and the Hungarian assignment result.

    Args:
        matrix: numpy array of shape (n_agents, n_roles)
        assignment: dict mapping role_name -> agent_index
        title: plot title
        role_names: column labels (defaults to standard 5 roles)

    Returns:
        Bokeh figure
    """
    if role_names is None:
        role_names = ["Expert", "Critic", "Evaluator", "Synthesizer", "Generalist"]

    n_agents, n_roles = matrix.shape
    if agent_model_names and len(agent_model_names) == n_agents:
        agent_names = [f"Agent {i} ({agent_model_names[i]})" for i in range(n_agents)]
    else:
        agent_names = [f"Agent {i}" for i in range(n_agents)]

    # Flatten matrix for Bokeh rect glyph
    agents = []
    roles = []
    values = []
    is_assigned = []
    value_text = []

    for i in range(n_agents):
        for j in range(n_roles):
            agents.append(agent_names[i])
            roles.append(role_names[j])
            values.append(float(matrix[i, j]))
            value_text.append(f"{matrix[i, j]:.1f}")

            # Check if this cell is the assigned one
            # Need to map role_name back to original enum-style key
            full_role_names_map = {
                "Expert": NodeRole.Domain_Expert,
                "Critic": NodeRole.Critic,
                "Evaluator": NodeRole.Evaluator,
                "Synthesizer": NodeRole.Synthesizer,
                "Generalist": NodeRole.Generalist,
            }
            role_enum = full_role_names_map.get(role_names[j])
            assigned_agent = assignment.get(role_enum, -1) if role_enum else -1
            is_assigned.append(assigned_agent == i)

    source = ColumnDataSource(data=dict(
        agent=agents,
        role=roles,
        value=values,
        value_text=value_text,
        is_assigned=is_assigned,
        line_width=[3 if a else 0 for a in is_assigned],
        line_color=["#E74C3C" if a else "#ffffff" for a in is_assigned],
    ))

    # Determine color range
    min_val = float(matrix.min())
    max_val = float(matrix.max())

    # Use blue palette (lighter = lower, darker = higher)
    mapper = LinearColorMapper(
        palette=list(reversed(Blues9)),
        low=min_val,
        high=max_val,
    )

    p = figure(
        width=500,
        height=350,
        title=title,
        x_range=role_names,
        y_range=list(reversed(agent_names)),
        toolbar_location=None,
        tools="hover",
    )

    p.rect(
        x="role",
        y="agent",
        width=1,
        height=1,
        source=source,
        fill_color=transform("value", mapper),
        line_color="line_color",
        line_width="line_width",
    )

    # Add value labels in each cell
    labels = LabelSet(
        x="role", y="agent", text="value_text", source=source,
        text_align="center", text_baseline="middle",
        text_font_size="9pt", text_color="black",
    )
    p.add_layout(labels)

    # Color bar
    color_bar = ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(),
        label_standoff=8,
        width=20,
        location=(0, 0),
    )
    p.add_layout(color_bar, "right")

    # Hover
    p.select_one(HoverTool).tooltips = [
        ("Agent", "@agent"),
        ("Role", "@role"),
        ("Fitness", "@value_text"),
        ("Assigned", "@is_assigned"),
    ]

    p.xaxis.axis_label = "Roles"
    p.yaxis.axis_label = "Agents"

    return p


# ============================================================
# 3. MSV EVOLUTION CHART
# ============================================================

def create_msv_evolution_chart(
    role_responses: list[NodeResponse],
) -> figure:
    """
    Show how MSV dimensions evolve across dialectical pipeline stages.

    X-axis: pipeline stages (Expert → Critic → Evaluator → Synthesizer)
    Y-axis: MSV dimension values (0-100)
    Lines: one per dimension (ER, CE, EM, CI, PI)

    This is the visualization that should show CI decreasing and CE
    increasing across stages (the hypothesized convergence pattern).
    """
    # Sort by dialectical order
    chain_roles = [NodeRole.Domain_Expert, NodeRole.Critic,
                   NodeRole.Evaluator, NodeRole.Synthesizer]
    role_order = {role: i for i, role in enumerate(chain_roles)}

    chain = [r for r in role_responses if r.node_role in chain_roles]
    chain.sort(key=lambda r: role_order.get(r.node_role, 99))

    if not chain:
        # Return empty figure
        return figure(width=600, height=300, title="MSV Evolution (no data)")

    stage_labels = [ROLE_SHORT_LABELS.get(r.node_role, r.node_role) for r in chain]
    stage_indices = list(range(len(chain)))

    # Extract dimensions at each stage
    er_vals = [float(r.node_msv.emotional_response.calculated_value) for r in chain]
    ce_vals = [float(r.node_msv.correctness.calculated_value) for r in chain]
    em_vals = [float(r.node_msv.experiential_matching.calculated_value) for r in chain]
    ci_vals = [float(r.node_msv.conflict_information.calculated_value) for r in chain]
    pi_vals = [float(r.node_msv.problem_importance.calculated_value) for r in chain]

    p = figure(
        width=600,
        height=350,
        title="MSV Evolution Across Pipeline Stages",
        x_range=stage_labels,
        y_range=(0, 105),
        toolbar_location=None,
        tools="hover",
    )

    # Plot each dimension as a line
    dims = [
        ("ER (Emotional)", er_vals, "#E74C3C"),
        ("CE (Correctness)", ce_vals, "#3498DB"),
        ("EM (Experiential)", em_vals, "#27AE60"),
        ("CI (Conflict)", ci_vals, "#F39C12"),
        ("PI (Importance)", pi_vals, "#8E44AD"),
    ]

    for label, vals, color in dims:
        source = ColumnDataSource(data=dict(
            x=stage_labels,
            y=vals,
            label=[label] * len(vals),
        ))
        p.line(x="x", y="y", source=source, line_width=2, color=color, legend_label=label)
        p.scatter(x="x", y="y", source=source, size=8, color=color, legend_label=label)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "8pt"
    p.xaxis.axis_label = "Pipeline Stage"
    p.yaxis.axis_label = "MSV Value (0-100)"

    p.select_one(HoverTool).tooltips = [
        ("Dimension", "@label"),
        ("Stage", "@x"),
        ("Value", "@y"),
    ]

    return p


# ============================================================
# 4. ROUTING GAUGE
# ============================================================

def create_routing_gauge(
    paper_activation: float,
    paper_threshold: float,
    sigmoid_value: float,
    sigmoid_threshold: float,
) -> figure:
    """
    Simple bar chart comparing the two routing methods.

    Shows activation value vs threshold for both sigmoid and paper routing.
    Green bar if above threshold (System 2), red if below (System 1).
    """
    methods = ["Sigmoid (existing)", "Paper (Eq 8)"]
    activations = [sigmoid_value, paper_activation]
    thresholds = [sigmoid_threshold, paper_threshold]
    colors = [
        "#27AE60" if activations[i] >= thresholds[i] else "#E74C3C"
        for i in range(2)
    ]
    decisions = [
        "→ System 2" if activations[i] >= thresholds[i] else "→ System 1"
        for i in range(2)
    ]

    source = ColumnDataSource(data=dict(
        method=methods,
        activation=activations,
        threshold=thresholds,
        color=colors,
        decision=decisions,
        activation_text=[f"{a:.4f}" for a in activations],
        threshold_text=[f"τ={t}" for t in thresholds],
    ))

    p = figure(
        width=400,
        height=250,
        x_range=methods,
        y_range=(0, 1.0),
        title="Routing Decision",
        toolbar_location=None,
        tools="hover",
    )

    # Activation bars
    p.vbar(
        x="method", top="activation", width=0.5,
        source=source, color="color", alpha=0.8,
    )

    # Threshold line indicators
    for i, (method, threshold) in enumerate(zip(methods, thresholds)):
        p.segment(
            x0=[method], y0=[threshold], x1=[method], y1=[threshold],
            line_color="black", line_width=2, line_dash="dashed",
        )

    p.select_one(HoverTool).tooltips = [
        ("Method", "@method"),
        ("Activation", "@activation_text"),
        ("Threshold", "@threshold_text"),
        ("Decision", "@decision"),
    ]

    p.xaxis.axis_label = ""
    p.yaxis.axis_label = "Activation"

    return p


# ============================================================
# 5. COMBINED COMPONENTS HELPER
# ============================================================

def create_role_pipeline_components(
    role_responses: list[NodeResponse],
    assignment_result=None,
    routing_result=None,
    generalist_annotation: str | None = None,
) -> dict[str, tuple]:
    """
    Create all visualizations and return as Bokeh component dict.

    Returns dict of (script, div) tuples keyed by visualization name,
    ready for template rendering via the existing pattern in app.py.

    Usage in app.py:
        role_components = create_role_pipeline_components(...)
        # Then pass to template context
    """
    result = {}

    # 1. Pipeline graph
    if role_responses:
        assignment_map = assignment_result.assignment if assignment_result else None
        pipeline_plot, nodes = create_role_pipeline_graph(
            role_responses, assignment_map, generalist_annotation
        )
        result["pipeline_graph"] = components(pipeline_plot)

    # 2. Fitness heatmaps (degenerate and heterogeneous)
    if assignment_result:
        degenerate_heatmap = create_fitness_heatmap(
            assignment_result.degenerate_matrix,
            assignment_result.assignment,
            title="Degenerate Fitness Matrix (identical rows — from System 1 MSV)",
            agent_model_names=assignment_result.agent_model_names,
        )
        result["degenerate_heatmap"] = components(degenerate_heatmap)

        if assignment_result.heterogeneous_matrix is not None:
            hetero_heatmap = create_fitness_heatmap(
                assignment_result.heterogeneous_matrix,
                assignment_result.assignment,
                title="Heterogeneous Fitness Matrix (from preliminary response MSVs)",
                agent_model_names=assignment_result.agent_model_names,
            )
            result["heterogeneous_heatmap"] = components(hetero_heatmap)

    # 3. MSV evolution
    if role_responses:
        evolution_chart = create_msv_evolution_chart(role_responses)
        result["msv_evolution"] = components(evolution_chart)

    # 4. Routing gauge
    if routing_result:
        gauge = create_routing_gauge(
            routing_result.paper_activation,
            routing_result.paper_threshold,
            routing_result.sigmoid_value,
            routing_result.sigmoid_threshold,
        )
        result["routing_gauge"] = components(gauge)

    return result
