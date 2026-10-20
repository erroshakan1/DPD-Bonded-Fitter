"""Write a small united-atom (TraPPE-UA) n-dodecane system as a LAMMPS data file.

    python make_system.py            -> dodecane.data   (then: lmp -in in.aa)

100 straight chains with random orientations in a dilute box; ``in.aa`` compresses them
to liquid density (NPT, 300 K, 1 atm) and writes the production trajectory ``aa.lammpstrj``.
"""
import numpy as np

N_MOL, N_C, BOX = 100, 12, 70.0
rng = np.random.default_rng(7)
coords = []
for _ in range(N_MOL):
    while True:
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v = np.cross(u, rng.normal(size=3))
        v /= np.linalg.norm(v)
        c = rng.uniform(10, BOX - 10, 3)
        # zig-zag chain: 1.54 A bonds, 114 deg angles
        x = np.array([c + (k - N_C / 2) * 1.29 * u + (0.43 if k % 2 else -0.43) * v for k in range(N_C)])
        if all(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1)) > 3.5 for y in coords):
            coords.append(x)
            break

lines = ["TraPPE-UA n-dodecane (cgdpd-fit example)", "",
         f"{N_MOL * N_C} atoms", f"{N_MOL * (N_C - 1)} bonds", f"{N_MOL * (N_C - 2)} angles",
         f"{N_MOL * (N_C - 3)} dihedrals", "2 atom types", "1 bond types", "1 angle types",
         "1 dihedral types", "", f"0 {BOX} xlo xhi", f"0 {BOX} ylo yhi", f"0 {BOX} zlo zhi", "",
         "Masses", "", "1 15.035  # CH3", "2 14.027  # CH2", "", "Atoms  # full", ""]
aid = 0
for m, x in enumerate(coords):
    for k in range(N_C):
        aid += 1
        t = 1 if k in (0, N_C - 1) else 2
        lines.append(f"{aid} {m + 1} {t} 0.0 {x[k, 0]:.4f} {x[k, 1]:.4f} {x[k, 2]:.4f}")
for sec, n in (("Bonds", 2), ("Angles", 3), ("Dihedrals", 4)):
    lines += ["", sec, ""]
    i = 0
    for m in range(N_MOL):
        base = m * N_C + 1
        for k in range(N_C - n + 1):
            i += 1
            lines.append(f"{i} 1 " + " ".join(str(base + k + j) for j in range(n)))
open("dodecane.data", "w").write("\n".join(lines) + "\n")
print("wrote dodecane.data")
