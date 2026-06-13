#!/usr/bin/env python3
"""Generate a customer-facing decision guide PPTX for ML engineers
training diffusion models on AWS Trainium/Neuron."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---- palette -----------------------------------------------------------
NEURON_NAVY   = RGBColor(0x16, 0x21, 0x3B)   # title bar / headings
PROBLEM_RED   = RGBColor(0xC0, 0x39, 0x2B)   # problem label
ORACLE_GOLD   = RGBColor(0xF5, 0xC5, 0x18)   # oracle box fill (consistent, stands out)
ORACLE_TEXT   = RGBColor(0x16, 0x21, 0x3B)
ACTION_GREEN  = RGBColor(0x1E, 0x7A, 0x46)   # action label
WHITE         = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT     = RGBColor(0x22, 0x22, 0x22)
LIGHT_GREY    = RGBColor(0xF2, 0xF2, 0xF2)
CODE_BG       = RGBColor(0x2B, 0x2B, 0x2B)
CODE_TEXT     = RGBColor(0xF8, 0xF8, 0xF2)

MONO = "Consolas"
SANS = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def add_box(slide, left, top, width, height, fill=None, line=None):
    box = slide.shapes.add_textbox(left, top, width, height)
    if fill is not None:
        box.fill.solid()
        box.fill.fore_color.rgb = fill
    else:
        box.fill.background()
    if line is not None:
        box.line.color.rgb = line
        box.line.width = Pt(1)
    else:
        box.line.fill.background()
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.08)
    tf.margin_bottom = Inches(0.08)
    return box


def set_text(tf, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, clear=True):
    """runs: list of (text, size, bold, color, font) tuples -> single paragraph."""
    if clear:
        tf.clear()
    p = tf.paragraphs[0]
    p.alignment = align
    tf.vertical_anchor = anchor
    for (text, size, bold, color, font) in runs:
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = font
    return p


def add_para(tf, text, size, bold, color, font, align=PP_ALIGN.LEFT, bullet=False, space_after=4):
    p = tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = ("•  " + text) if bullet else text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return p


# ---- title slide -------------------------------------------------------
def title_slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape  # not needed; use full-bleed rect via textbox fill
    band = add_box(s, 0, 0, SLIDE_W, SLIDE_H, fill=NEURON_NAVY)
    tf = band.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    set_text(tf, [("Training Diffusion Models on AWS Trainium / Neuron",
                   40, True, WHITE, SANS)], align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
    add_para(tf, "A Decision Guide for ML Engineers", 24, False,
             ORACLE_GOLD, SANS, align=PP_ALIGN.CENTER, space_after=18)
    add_para(tf, "Each slide: one problem  →  a self-diagnosis oracle  →  a recommended action",
             18, False, WHITE, SANS, align=PP_ALIGN.CENTER, space_after=6)
    add_para(tf, "Covers all 9 diffusion-on-Neuron training / compile blockers, grouped by who clears them",
             16, False, RGBColor(0xB9, 0xC2, 0xD6), SANS, align=PP_ALIGN.CENTER, space_after=4)
    add_para(tf, "Agent skill  ·  Customer  ·  AWS-SDK     —     still flat, one problem per slide (no branching)",
             14, False, RGBColor(0xB9, 0xC2, 0xD6), SANS, align=PP_ALIGN.CENTER)
    return s


# ---- provenance pill colors (who clears the blocker) -------------------
WHO_AGENT    = RGBColor(0x2E, 0x6D, 0xA4)   # Agent skill / config
WHO_CUSTOMER = RGBColor(0x7A, 0x5A, 0x1E)   # Customer decision
WHO_AWS      = RGBColor(0xC0, 0x39, 0x2B)   # AWS / Neuron SDK (hard wall)

WHO_COLORS = {
    "Agent skill": WHO_AGENT,
    "Customer":    WHO_CUSTOMER,
    "AWS-SDK":     WHO_AWS,
}


# ---- landscape / category overview slide -------------------------------
def overview_slide():
    """Flat summary table — the whole map of the 6 categories before detail.
    No branching logic; the 'Cleared by' column reuses the provenance pill colors."""
    s = prs.slides.add_slide(BLANK)

    # title bar (same style as problem slides)
    bar = add_box(s, 0, 0, SLIDE_W, Inches(1.05), fill=NEURON_NAVY)
    tf = bar.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    set_text(tf, [("THE LANDSCAPE   ", 16, True, ORACLE_GOLD, SANS),
                  ("14 Common Blockers in 6 Categories", 26, True, WHITE, SANS)],
             anchor=MSO_ANCHOR.MIDDLE)

    margin = Inches(0.6)
    table_w = SLIDE_W - 2 * margin
    top = Inches(1.35)

    # rows: (category, common problems (count), cleared-by who-key)
    rows = [
        ("P0 · Source-script env",     "HF token, Python 3.10 typing.Self (2)",                    "Customer"),
        ("P1 · Dependency coherence",  "CUDA-torch overwrite, accelerate mismatch, diffusers missing/too-new (3)", "Agent skill"),
        ("P2 · transformers assumption","tie_weights on diffusers model (1)",                       "Agent skill"),
        ("P3 · XLA execution model",   "inference_mode, per-step host-sync (2)",                    "Customer"),
        ("P4 · Memory & topology",     "host-RAM OOM, core-count topology, 16GB HBM, pinned-memory (4)", "Customer"),
        ("P5 · Architecture / compiler","dynamic-shape (set-dimension-size) wall, compile-cache NotImpl (2)", "AWS-SDK"),
    ]

    n_rows = len(rows) + 1            # + header
    n_cols = 3
    col_w = [Inches(3.2), table_w - Inches(3.2) - Inches(2.6), Inches(2.6)]
    header_h = Inches(0.5)
    row_h = Inches(0.72)

    gfx = s.shapes.add_table(n_rows, n_cols, margin, top, table_w,
                             header_h + row_h * len(rows))
    table = gfx.table
    table.first_row = False
    table.horz_banding = False
    table.columns[0].width = col_w[0]
    table.columns[1].width = col_w[1]
    table.columns[2].width = col_w[2]
    table.rows[0].height = header_h
    for i in range(1, n_rows):
        table.rows[i].height = row_h

    def style_cell(cell, runs, fill, align=PP_ALIGN.LEFT, size=12, bold=False,
                   color=DARK_TEXT, font=SANS):
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
        cell.margin_left = Inches(0.1)
        cell.margin_right = Inches(0.1)
        cell.margin_top = Inches(0.04)
        cell.margin_bottom = Inches(0.04)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        if runs is None:
            runs = []
        set_text(tf, runs, align=align, anchor=MSO_ANCHOR.MIDDLE)

    # header row
    headers = ["Category", "Common problems (count)", "Cleared by"]
    for c, h in enumerate(headers):
        al = PP_ALIGN.CENTER if c == 2 else PP_ALIGN.LEFT
        style_cell(table.cell(0, c),
                   [(h, 13, True, WHITE, SANS)],
                   fill=NEURON_NAVY, align=al)

    # data rows
    for r, (cat, probs, who) in enumerate(rows, start=1):
        band = WHITE if r % 2 else LIGHT_GREY
        style_cell(table.cell(r, 0), [(cat, 13, True, NEURON_NAVY, SANS)], fill=band)
        style_cell(table.cell(r, 1), [(probs, 12, False, DARK_TEXT, SANS)], fill=band)
        # "Cleared by" cell color-coded with the provenance pill colors
        style_cell(table.cell(r, 2), [(who, 12.5, True, WHITE, SANS)],
                   fill=WHO_COLORS[who], align=PP_ALIGN.CENTER)

    # one short takeaway line under the table
    note_top = top + header_h + row_h * len(rows) + Inches(0.25)
    note = add_box(s, margin, note_top, table_w, Inches(0.9))
    ntf = note.text_frame
    set_text(ntf,
             [("Most blockers are environment / integration (P0–P2) and are clearable up front.  ",
               13.5, False, DARK_TEXT, SANS),
              ("The only hard wall is P5", 13.5, True, WHO_AWS, SANS),
              (" — dynamic-shape architecture, which needs AWS Neuron SDK support.",
               13.5, False, DARK_TEXT, SANS)])
    return s


# ---- problem slide -----------------------------------------------------
def problem_slide(tag, title, problem_text, oracle_lines, action_lines, who):
    s = prs.slides.add_slide(BLANK)

    # title bar
    bar = add_box(s, 0, 0, SLIDE_W, Inches(1.05), fill=NEURON_NAVY)
    tf = bar.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    set_text(tf, [(tag + "   ", 16, True, ORACLE_GOLD, SANS),
                  (title, 26, True, WHITE, SANS)],
             anchor=MSO_ANCHOR.MIDDLE)

    # provenance pill (top-right) — a flat provenance label, not branching
    pill_w = Inches(2.7)
    pill = add_box(s, SLIDE_W - pill_w - Inches(0.25), Inches(0.22),
                   pill_w, Inches(0.6), fill=WHO_COLORS[who])
    ptf = pill.text_frame
    ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
    set_text(ptf, [("Cleared by:  ", 11, False, WHITE, SANS),
                   (who, 13, True, WHITE, SANS)],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    margin = Inches(0.6)
    width = SLIDE_W - 2 * margin
    y = Inches(1.35)

    # 1. THE PROBLEM
    lbl = add_box(s, margin, y, width, Inches(0.4))
    set_text(lbl.text_frame, [("1.  THE PROBLEM", 16, True, PROBLEM_RED, SANS)])
    y = y + Inches(0.42)
    pbox = add_box(s, margin, y, width, Inches(1.0), fill=LIGHT_GREY)
    tf = pbox.text_frame
    set_text(tf, [(problem_text, 15, False, DARK_TEXT, SANS)],
             anchor=MSO_ANCHOR.MIDDLE)
    y = y + Inches(1.12)

    # 2. ORACLE  (consistent gold box)
    lbl = add_box(s, margin, y, width, Inches(0.4))
    set_text(lbl.text_frame,
             [("2.  ORACLE", 16, True, ORACLE_TEXT, SANS),
              ("   — you ARE in this situation if:", 13, False, RGBColor(0x66, 0x66, 0x66), SANS)])
    y = y + Inches(0.42)
    obox = add_box(s, margin, y, width, Inches(1.45), fill=ORACLE_GOLD)
    tf = obox.text_frame
    first = True
    for (text, mono) in oracle_lines:
        font = MONO if mono else SANS
        if first:
            set_text(tf, [(text, 13.5, mono, ORACLE_TEXT, font)])
            first = False
        else:
            add_para(tf, text, 13.5, mono, ORACLE_TEXT, font, space_after=3)
    y = y + Inches(1.57)

    # 3. RECOMMENDED ACTION
    lbl = add_box(s, margin, y, width, Inches(0.4))
    set_text(lbl.text_frame, [("3.  RECOMMENDED ACTION", 16, True, ACTION_GREEN, SANS)])
    y = y + Inches(0.42)
    abox = add_box(s, margin, y, width, Inches(1.1),
                   fill=RGBColor(0xE8, 0xF2, 0xEB))
    tf = abox.text_frame
    first = True
    for text in action_lines:
        if first:
            set_text(tf, [(text, 14, False, DARK_TEXT, SANS)])
            first = False
        else:
            add_para(tf, text, 14, False, DARK_TEXT, SANS, bullet=True, space_after=3)
    return s


# ---- checklist slide ---------------------------------------------------
def checklist_slide():
    s = prs.slides.add_slide(BLANK)
    bar = add_box(s, 0, 0, SLIDE_W, Inches(1.05), fill=NEURON_NAVY)
    tf = bar.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    set_text(tf, [("Decision Checklist", 28, True, WHITE, SANS)],
             anchor=MSO_ANCHOR.MIDDLE)

    margin = Inches(0.8)
    width = SLIDE_W - 2 * margin
    box = add_box(s, margin, Inches(1.3), width, Inches(6.0))
    tf = box.text_frame
    set_text(tf, [("Who clears what  —  a flat provenance list (not branching):",
                   18, True, NEURON_NAVY, SANS)])

    def section(label, color, items):
        p = tf.add_paragraph()
        p.space_before = Pt(8)
        p.space_after = Pt(3)
        r = p.add_run()
        r.text = label
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = color
        r.font.name = SANS
        for it in items:
            ip = tf.add_paragraph()
            ip.space_after = Pt(4)
            ir = ip.add_run()
            ir.text = "☐  " + it
            ir.font.size = Pt(13.5)
            ir.font.color.rgb = DARK_TEXT
            ir.font.name = SANS

    section("Agent skill clears  —  dependency coherence + transformers assumption", WHO_AGENT, [
        "Single eager install of optimum-neuron[neuronx,training]; diffusers/peft/torchvision --no-deps.",
        "Pin accelerate==1.8.1; pin diffusers==0.35.* for 0.4.5 inference.",
        "Inject the no-op tie_weights shim before accelerator.prepare().",
        "Pass disable_neuron_cache=True for FLUX DiT inference compile.",
    ])
    section("Customer decides  —  source-script env, precision, cores, host-syncs", WHO_CUSTOMER, [
        ".env with HF_TOKEN (gated model); typing_extensions on Python 3.10.",
        "Precision: load in bf16 (torch_dtype=torch.bfloat16).",
        "Cores: NUM_CORES in {1, 4, 8, 16, 32} (2 is not a valid topology).",
        "Single-core fit / host RAM per DP worker; tensor-parallel if model exceeds 16 GB HBM.",
        "Replace inference_mode() with no_grad(); remove per-step host syncs.",
    ])
    section("AWS must provide  —  the architectural wall (no shim clears it)", WHO_AWS, [
        "A static-shape VLM encoder (Qwen2.5-VL emits set-dimension-size; neuronx-cc rejects it).",
        "Tracked upstream as aws-neuron-sdk #1144 — needs model rewrite or SDK support.",
    ])
    return s


# ===== BUILD ============================================================
title_slide()

# Landscape / category overview — the whole map before per-problem detail.
overview_slide()

# Ordered most-commonly-hit-first. Each slide is one problem; the "Cleared by"
# pill is a provenance label only (not branching logic).

# --- P0: stock-script env (Customer) ---
problem_slide(
    "P0 · SOURCE-SCRIPT ENV",
    "HF_TOKEN missing for gated model",
    "The training script reads the Hugging Face token from the environment at import time. "
    "The model is gated, so without the token the script dies before anything starts.",
    [("KeyError: 'HF_TOKEN'", True),
     ("raised at import / startup.", False)],
    ["Create a .env file containing your Hugging Face token (HF_TOKEN=hf_...).",
     "The token is required because the model is gated; request access on the model page first."],
    who="Customer",
)

problem_slide(
    "P0 · SOURCE-SCRIPT ENV",
    "typing.Self ImportError on Python 3.10",
    "The script (or a dependency) imports Self from typing, which only exists in Python 3.11+. "
    "On a Python 3.10 interpreter the import fails at startup.",
    [("ImportError: cannot import name 'Self' from", True),
     ("'typing'", True)],
    ["The repo falls back to typing_extensions — ensure typing_extensions is installed.",
     "pip install typing_extensions (provides Self on Python 3.10)."],
    who="Customer",
)

# --- P1: dependency coherence (Agent skill / config) ---
problem_slide(
    "P1 · DEPENDENCY COHERENCE",
    "CUDA torch overwrote the Neuron stack",
    "A pip install pulled a CUDA build of torch on top of the Neuron stack, breaking the "
    "torch / torch-neuronx pairing. optimum-neuron can no longer find NeuronAccelerator.",
    [("pip show torch  →  2.x.x+cu124  (a +cu build), OR", True),
     ("torch major.minor != torch-neuronx (e.g. torch 2.12 vs torch-neuronx 2.8), OR", False),
     ("ImportError: cannot import name 'NeuronAccelerator'", True)],
    ["Install optimum-neuron[neuronx,training] in one eager resolve (lets it pick the matched torch).",
     "Install diffusers / peft / torchvision with --no-deps so they cannot drag in CUDA torch.",
     "For optimum-neuron 0.4.5 inference, pin diffusers==0.35.*."],
    who="Agent skill",
)

problem_slide(
    "P1 · DEPENDENCY COHERENCE",
    "accelerate version mismatch",
    "The installed accelerate is the wrong version for optimum-neuron 0.4.5. Constructing "
    "NeuronAccelerator() fails because the shared-state plumbing it expects is missing.",
    [("AttributeError: 'functools.partial' object has no", True),
     ("attribute '_shared_state'", True),
     ("raised at NeuronAccelerator().", False)],
    ["Pin accelerate==1.8.1 — the version optimum-neuron 0.4.5 expects.",
     "Reinstall after pinning so the matched accelerate is the one that loads."],
    who="Agent skill",
)

problem_slide(
    "P1 · DEPENDENCY COHERENCE",
    "diffusers missing or too new",
    "diffusers is absent, or a too-new version removed/moved modules the pipeline imports "
    "(e.g. the controlnet path), so the import fails.",
    [("ModuleNotFoundError: No module named 'diffusers', OR", True),
     ("No module named 'diffusers.models.controlnet'", True)],
    ["Install diffusers.",
     "For optimum-neuron 0.4.5 inference, pin diffusers==0.35.* (newer versions move modules)."],
    who="Agent skill",
)

# --- P2: transformers assumption (Agent skill / config) ---
problem_slide(
    "P2 · TRANSFORMERS ASSUMPTION",
    "tie_weights AttributeError at prepare()",
    "optimum-neuron's accelerator.prepare() assumes a transformers model and calls tie_weights(). "
    "A diffusers DiT model (FluxTransformer2DModel) has no such method, so prepare() crashes.",
    [("AttributeError: 'FluxTransformer2DModel' object", True),
     ("has no attribute 'tie_weights'", True),
     ("raised inside accelerator.prepare().", False)],
    ["Inject a no-op tie_weights shim on the XLA path before prepare() (diffusers != transformers).",
     "e.g. model.tie_weights = lambda *a, **k: None  prior to accelerator.prepare(model)."],
    who="Agent skill",
)

# --- P3: XLA execution model (Customer) ---
problem_slide(
    "P3 · XLA EXECUTION MODEL",
    "torch.inference_mode() breaks on XLA",
    "Your training/validation code calls torch.inference_mode() (common in stock diffusion "
    "pipelines). On Neuron/XLA this raises immediately — the run never starts.",
    [("RuntimeError: Cannot set version_counter for", True),
     ("inference tensor", True),
     ("… appearing in the traceback (raised under an inference_mode context).", False)],
    ["Replace torch.inference_mode() with torch.no_grad() on Neuron.",
     "Search the pipeline / eval code for inference_mode and swap every occurrence."],
    who="Customer",
)

problem_slide(
    "P3 · XLA EXECUTION MODEL",
    "Per-step host syncs make training crawl or 'hang'",
    "Training starts but each step is unusably slow or appears frozen. A host-sync op in your "
    "custom loss/sampling code (.item(), .tolist(), .nonzero().item()) forces the lazy XLA graph "
    "to materialize every step, destroying performance.",
    [("Step time is orders of magnitude slower than expected, OR", False),
     ("the process hangs right after the log line:", False),
     ("Compiler status PASS", True)],
    ["Vectorize the computation and remove per-step host syncs from custom loss / sampling code.",
     "Avoid .item() / .tolist() / .nonzero().item() inside the training loop; keep tensors on device.",
     "Move any unavoidable host reads outside the hot loop (e.g. log every N steps)."],
    who="Customer",
)

# --- P4: memory & topology (Customer) ---
problem_slide(
    "P4 · MEMORY & TOPOLOGY",
    "Host-RAM OOM with data parallelism",
    "With more than one core, each data-parallel worker loads its own full fp32 copy of the model "
    "into host RAM. The OS kills a worker before training stabilizes.",
    [("exitcode: -9", True),
     ("(SIGKILL by the OOM killer) when NUM_CORES > 1.", False)],
    ["Load the model in bf16 (torch_dtype=torch.bfloat16) to halve host-RAM per worker.",
     "And/or reduce the core count — data-parallel replicates the full model on every core."],
    who="Customer",
)

problem_slide(
    "P4 · MEMORY & TOPOLOGY",
    "NeuronCore count must be a valid topology",
    "You set NUM_CORES=2 and collective initialization fails. Two cores is not a valid Neuron "
    "collective topology.",
    [("Collective init fails with:", False),
     ("BuildGlobalComm error", True),
     ("when NUM_CORES=2.", False)],
    ["Use only 1, 4, 8, 16, or 32 cores. 2 is not a valid topology.",
     "Use 1 core for single-step validation; scale to 4/8/16/32 for real training."],
    who="Customer",
)

problem_slide(
    "P4 · MEMORY & TOPOLOGY",
    "Model too big for one core's 16 GB HBM",
    "Even on a single core the model will not fit in device memory. HBM sits at the limit and the "
    "allocator fails.",
    [("Failed to allocate … on ND 0:NC 0", True),
     ("with HBM near 16/16 GB on a single core.", False)],
    ["Tensor-parallel: shard the model across cores (requires framework support for the model).",
     "And/or load in bf16 to halve the footprint.",
     "Data-parallel will NOT help — it replicates the full model on each core."],
    who="Customer",
)

problem_slide(
    "P4 · MEMORY & TOPOLOGY",
    "Pinned host-memory exhaustion",
    "fp32 full-model load plus host-to-device transfer buffers exhaust pinned host memory, and "
    "the transfer fails.",
    [("pinned_host_memory … errno=9", True),
     ("(Bad file descriptor)", True)],
    ["Load in bf16 to halve the transfer size.",
     "Raise file-descriptor and shared-memory (shm) limits on the host."],
    who="Customer",
)

# --- P5: architectural wall (AWS / Neuron SDK) ---
problem_slide(
    "P5 · ARCHITECTURAL WALL",
    "Dynamic-shape unsupported op — the hard wall",
    "The compiler is static-shape. A model whose encoder emits data-dependent shapes (e.g. a VLM "
    "encoder like Qwen2.5-VL) emits set-dimension-size, which neuronx-cc structurally rejects. "
    "No shim or flag fixes this.",
    [("[NCC_VRF001] An unsupported operator was found:", True),
     ("set-dimension-size   (neuronx-cc exit 70), OR", True),
     ("the model uses a VLM encoder (Qwen2.5-VL).", False)],
    ["NOT clearable by config — the encoder must be rewritten to static shapes, or use a "
     "static-shape model (e.g. FLUX).",
     "Requires AWS Neuron SDK support — tracked in aws-neuron-sdk #1144.",
     "No tie_weights / no_grad / pin shim will make this compile."],
    who="AWS-SDK",
)

problem_slide(
    "P5 · ARCHITECTURAL WALL",
    "MultiModelCacheEntry NotImplementedError (inference compile)",
    "On inference compile, optimum-neuron 0.4.5's compile cache only supports unet-based Stable "
    "Diffusion, not the FLUX DiT. The cache layer raises NotImplementedError.",
    [("NotImplementedError from", True),
     ("optimum/neuron/cache/entries/multi_model.py", True)],
    ["Pass disable_neuron_cache=True at compile.",
     "The 0.4.5 compile cache only models unet-based SD; disabling it lets the FLUX DiT compile."],
    who="Agent skill",
)

checklist_slide()

OUT = "/Users/jaebin/AWS/repositories/neuron-agentic-development/FLUX.1-Kontext-dev-Training/customer-decision-guide.pptx"
prs.save(OUT)
print("saved:", OUT)
print("slides:", len(prs.slides._sldIdLst))
