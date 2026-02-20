# app.py
# Streamlit HP lattice model explorer:
# - Enumerates non-clashing (self-avoiding) lattice conformations for a short HP sequence (2D square lattice)
# - Computes energies from contact model (H-H favorable; optional penalties for P- contacts)
# - Computes Boltzmann factors and probabilities
# - Visualizes the most probable conformations and provides an interactive viewer
#
# How to run:
#   pip install streamlit matplotlib numpy
#   streamlit run app.py

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

Coord = Tuple[int, int]
Dirs = [(1,0), (0,1), (-1,0), (0,-1)]  # +x, +y, -x, -y (square lattice)

@dataclass
class Conformation:
    coords: List[Coord]  # length N, coords[i] is lattice position of residue i
    energy: float        # model energy
    boltz_weight: float  # exp(-E/kT), with k_B=1 reduced units
    probability: float   # normalized over microstates
    contacts: List[Tuple[int,int]]  # list of non-bonded contact pairs (i,j), i<j


def parse_sequence(seq_raw: str) -> str:
    """Normalize user input into a clean HP sequence."""
    seq = seq_raw.strip().upper().replace(" ", "")
    allowed = set(["H", "P"])
    seq = "".join([c for c in seq if c in allowed])
    return seq


def enumerate_saws(seq: str, max_confs: int) -> List[List[Coord]]:
    """
    Enumerate self-avoiding walks (SAWs) for the given sequence length on 2D square lattice.
    - Fix the first move to +x to break rotational symmetry.
    - Avoid immediate backtracking to reduce trivial branches.
    - Hard cap at max_confs to keep UI responsive.
    Returns a list of coordinate lists.
    """
    N = len(seq)
    if N < 2:
        return [[(0,0)]]

    start = (0,0)
    path: List[Coord] = [start]
    occupied = {start}
    conformations: List[List[Coord]] = []
    count = 0

    def dfs(step: int, last_dir: Optional[Coord]):
        nonlocal count
        if count >= max_confs:
            return
        if step == N:
            conformations.append(path.copy())
            count += 1
            return

        # The first move is fixed to +x to remove rotational degeneracy
        if step == 1:
            moves = [(1,0)]
        else:
            moves = Dirs

        for d in moves:
            # no immediate backtracking
            if last_dir is not None and d[0] == -last_dir[0] and d[1] == -last_dir[1]:
                continue
            nx = path[-1][0] + d[0]
            ny = path[-1][1] + d[1]
            nxt = (nx, ny)
            if nxt in occupied:
                continue
            # place
            path.append(nxt)
            occupied.add(nxt)
            dfs(step + 1, d)
            # backtrack
            path.pop()
            occupied.remove(nxt)

    dfs(step=1, last_dir=None)
    return conformations


def compute_contacts(coords: List[Coord]) -> List[Tuple[int,int]]:
    """Return list of non-bonded contact pairs (i,j), i<j, |i-j|>1, lattice-adjacent by Manhattan distance 1."""
    pos_to_idx = {c: i for i, c in enumerate(coords)}
    contacts = []
    for i, (x, y) in enumerate(coords):
        for dx, dy in Dirs:
            j_coord = (x + dx, y + dy)
            j = pos_to_idx.get(j_coord, None)
            if j is None:
                continue
            if j <= i:
                continue  # ensure i<j to avoid double count
            if abs(i - j) <= 1:
                continue  # consecutive neighbors are covalent bonds, not contacts
            contacts.append((i, j))
    return contacts


def energy_hp_model(seq: str,
                    contacts: List[Tuple[int,int]],
                    e_hh: float = -1.0,
                    e_ph: float = 0.0,
                    e_pp: float = 0.0) -> float:
    """
    Classic HP model with optional penalties for P contacts:
    - H-H contact contributes e_hh (default -1)
    - H-P contact contributes e_ph (default 0)
    - P-P contact contributes e_pp (default 0)
    """
    E = 0.0
    for i, j in contacts:
        a, b = seq[i], seq[j]
        if a == 'H' and b == 'H':
            E += e_hh
        elif (a == 'H' and b == 'P') or (a == 'P' and b == 'H'):
            E += e_ph
        elif a == 'P' and b == 'P':
            E += e_pp
    return E


def boltzmann_weight(E: float, kT: float) -> float:
    # safe guard for very small kT
    kT_eff = max(1e-6, kT)
    return math.exp(-E / kT_eff)


def build_conformations(seq: str,
                        coords_list: List[List[Coord]],
                        kT: float,
                        e_hh: float,
                        e_ph: float,
                        e_pp: float) -> List[Conformation]:
    confs: List[Conformation] = []
    weights = []
    for coords in coords_list:
        contacts = compute_contacts(coords)
        E = energy_hp_model(seq, contacts, e_hh=e_hh, e_ph=e_ph, e_pp=e_pp)
        w = boltzmann_weight(E, kT)
        confs.append(Conformation(coords=coords, energy=E, boltz_weight=w,
                                  probability=0.0, contacts=contacts))
        weights.append(w)
    Z = sum(weights) if weights else 1.0
    for c in confs:
        c.probability = c.boltz_weight / Z
    return confs


def draw_conformation(coords: List[Coord],
                      seq: str,
                      contacts: List[Tuple[int,int]],
                      title: Optional[str] = None,
                      grow_to: Optional[int] = None,
                      dpi: int = 150):
    """
    Draw the conformation using Matplotlib.
    - coords: list of lattice coordinates for residues 0..N-1
    - seq: "H"/"P" string
    - contacts: list of non-bonded contacts (i,j)
    - grow_to: if provided, show chain only up to this residue index (inclusive)
    """
    N = len(coords)
    if grow_to is None:
        grow_to = N - 1
    grow_to = max(0, min(grow_to, N - 1))

    H_color = "#d62728"  # red
    P_color = "#1f77b4"  # blue
    bond_color = "black"
    contact_color = "#2ca02c"  # green (non-bonded contacts)
    node_size = 600

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    margin = 1
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin

    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xticks(range(xmin, xmax + 1))
    ax.set_yticks(range(ymin, ymax + 1))
    ax.tick_params(labelsize=8)

    # Draw covalent bonds up to grow_to
    for i in range(min(grow_to, N - 2) + 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        ax.plot([x1, x2], [y1, y2], color=bond_color, linewidth=2)

    # Draw non-bonded contacts for fully grown chain only
    if grow_to == N - 1:
        for (i, j) in contacts:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            ax.plot([x1, x2], [y1, y2], color=contact_color, linestyle='--', alpha=0.7)

    # Draw residues (circles) up to grow_to
    for i in range(grow_to + 1):
        x, y = coords[i]
        color = H_color if seq[i] == 'H' else P_color
        ax.scatter([x], [y], s=node_size, color=color, edgecolors='k', zorder=3)
        ax.text(x, y, f"{i}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def cached_enumeration(seq: str, max_confs: int) -> List[List[Coord]]:
    return enumerate_saws(seq, max_confs=max_confs)


def main():
    st.set_page_config(page_title="HP Lattice Protein Explorer", layout="wide")
    st.title("HP Lattice Protein Explorer (2D square lattice)")

    st.caption(
        "Enumerate non-clashing lattice conformations for a short HP sequence, "
        "compute Boltzmann probabilities, and visualize the most probable states."
    )

    with st.sidebar:
        st.header("Inputs")

        seq_default = "HPPHHPH"
        seq_raw = st.text_input("HP sequence (only H and P)", value=seq_default, help="Example: HPPHHPH")
        seq = parse_sequence(seq_raw)
        N = len(seq)

        if N == 0:
            st.error("Please enter a sequence containing only H and P.")
            st.stop()

        st.markdown(f"**Length:** {N}")

        st.divider()
        st.subheader("Energy parameters")
        e_hh = st.number_input("H–H contact energy (favorable < 0)", value=-1.0, step=0.1, format="%.3f")
        e_ph = st.number_input("P–H contact penalty (≥ 0)", value=0.0, step=0.1, format="%.3f")
        e_pp = st.number_input("P–P contact penalty (≥ 0)", value=0.0, step=0.1, format="%.3f")

        st.subheader("Thermodynamics")
        kT = st.slider("Reduced temperature kT (k_B = 1)", min_value=0.05, max_value=5.0, value=1.0, step=0.05)

        st.subheader("Enumeration")
        max_confs = st.number_input("Max conformations to enumerate (cap)", value=50000, min_value=100, step=100)
        st.caption("Enumeration grows fast with length; increase cautiously.")

        st.divider()
        st.subheader("Display")
        top_k = st.slider("Number of top conformations to display", min_value=1, max_value=50, value=9, step=1)
        columns_per_row = st.slider("Columns per row (grid view)", min_value=2, max_value=6, value=3, step=1)

        st.divider()
        st.subheader("Utilities")
        randomize = st.checkbox("Use a random sequence (50% H, 50% P) of the same length")
        if randomize:
            rng = np.random.default_rng(42)
            seq = "".join(["H" if x else "P" for x in rng.integers(0, 2, size=N)])
            st.info(f"Random sequence: {seq}")

        run = st.button("Enumerate & Analyze", type="primary")

    # Interactive illustration (build-up animation) for any current sequence
    st.subheader("Interactive Illustration: Build the chain on the lattice")
    st.caption("This shows a single conformation’s growth step-by-step. Use the slider to change the shown length.")
    # For illustration without full enumeration: build a simple 'snake' along +x then fold back
    # But better to reuse a valid enumerated conformation when available.
    # We'll enumerate a small number quickly to have a concrete path to animate.
    demo_coords_list = enumerate_saws(seq, max_confs=1)
    if demo_coords_list:
        demo_coords = demo_coords_list[0]
        grow_to = st.slider("Show residues up to index", min_value=0, max_value=len(seq)-1, value=min(len(seq)-1, 5))
        fig_demo = draw_conformation(demo_coords, seq, contacts=[], title="Chain growth (covalent bonds only)", grow_to=grow_to, dpi=150)
        st.pyplot(fig_demo, use_container_width=False)
    else:
        st.warning("Could not generate a quick demo conformation for this sequence.")

    st.divider()

    if not run:
        st.info("Set parameters in the sidebar and click **Enumerate & Analyze** to compute all non-clashing conformations and probabilities.")
        st.stop()

    # Enumerate
    with st.spinner("Enumerating non-clashing conformations (self-avoiding walks)..."):
        coords_list = cached_enumeration(seq, max_confs=int(max_confs))
    num_confs = len(coords_list)
    if num_confs == 0:
        st.error("No conformations found (unexpected for very short sequences). Try reducing penalties or increasing max_confs.")
        st.stop()

    # Compute energies and probabilities
    with st.spinner("Computing energies, Boltzmann factors, and probabilities..."):
        confs = build_conformations(seq, coords_list, kT=kT, e_hh=e_hh, e_ph=e_ph, e_pp=e_pp)
        # sort by probability descending
        confs.sort(key=lambda c: (-c.probability, c.energy))

    # Summary stats
    energies = np.array([c.energy for c in confs], dtype=float)
    probs = np.array([c.probability for c in confs], dtype=float)
    Z = float(np.sum([c.boltz_weight for c in confs]))
    E_avg = float(np.sum(energies * probs))
    S = float(-np.sum(np.where(probs > 0, probs * np.log(probs), 0.0)))  # Shannon entropy (in k_B units)
    E_min = float(np.min(energies))
    E_max = float(np.max(energies))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Non-clashing conformations", f"{num_confs:,}")
    c2.metric("Partition function Z", f"{Z:.4g}")
    c3.metric("⟨E⟩ (avg energy)", f"{E_avg:.3f}")
    c4.metric("Energy min / max", f"{E_min:.1f} / {E_max:.1f}")
    c5.metric("Entropy (k_B units)", f"{S:.3f}")

    # Show top conformations grid
    st.subheader(f"Top {min(top_k, num_confs)} most probable conformations")
    grid = confs[:min(top_k, num_confs)]

    rows = (len(grid) + columns_per_row - 1) // columns_per_row
    idx = 0
    for _ in range(rows):
        cols = st.columns(columns_per_row)
        for col in cols:
            if idx >= len(grid):
                break
            c = grid[idx]
            title = f"Rank {idx+1} | E={c.energy:.2f} | p={c.probability:.3f}"
            fig = draw_conformation(c.coords, seq, c.contacts, title=title, grow_to=None, dpi=150)
            with col:
                st.pyplot(fig, use_container_width=True)
            idx += 1

    st.divider()

    # Interactive viewer for any conformation by rank
    st.subheader("Interactive Viewer")
    st.caption("Pick a rank to view the conformation and step through its growth.")
    rank = st.slider("Choose rank (1 = most probable)", min_value=1, max_value=num_confs, value=1)
    chosen = confs[rank - 1]
    grow_to2 = st.slider("Show residues up to index (interactive)", min_value=0, max_value=len(seq)-1, value=len(seq)-1)
    title2 = f"Rank {rank} | E={chosen.energy:.2f} | p={chosen.probability:.4f}"
    fig2 = draw_conformation(chosen.coords, seq, chosen.contacts, title=title2, grow_to=grow_to2, dpi=150)
    st.pyplot(fig2, use_container_width=False)

    # Energy histogram (optional visualization of distribution)
    st.subheader("Energy distribution (over microstates)")
    # Bin by unique energies and sum probabilities per bin
    unique_E, idxs = np.unique(energies, return_inverse=True)
    prob_by_E = np.zeros_like(unique_E, dtype=float)
    for k, p in zip(idxs, probs):
        prob_by_E[k] += p

    fig_hist, axh = plt.subplots(figsize=(5, 3), dpi=150)
    axh.bar(unique_E, prob_by_E, width=0.6, color="#9467bd")
    axh.set_xlabel("Energy")
    axh.set_ylabel("Probability mass")
    axh.set_title("P(E) aggregated by energy level")
    axh.grid(True, axis='y', linestyle=':', alpha=0.4)
    st.pyplot(fig_hist, use_container_width=False)

    st.caption(
        "Notes: Enumeration fixes the first step to +x to remove rotational degeneracy. "
        "Mirror images may still be present. Energies reflect non-bonded lattice-adjacent contacts only."
    )


if __name__ == "__main__":
    main()