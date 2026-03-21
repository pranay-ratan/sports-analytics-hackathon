import nbformat

notebook_path = "/Users/pranayratan/Desktop/AdvancedLevel_SSAC Sport Med Hackathon 2026 copy/pitcher_injury_risk.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

new_cell_content = """### 3. Biomechanical Fatigue and In-Game Stress (Advanced Analysis)
A purely volumetric approach inherently misses the physical toll exacted per-pitch. To push our injury-risk metric further, we extracted raw biomechanics data directly from the Trackman CSVs at a pitch-by-pitch level. 

Specifically, we added:
1. **High Stress Innings**: The number of innings a pitcher was forced to throw 25+ pitches (a well documented spike in acute injury tissue stress).
2. **Velocity Drop (`mean_velo_drop`)**: The difference in MPH between the first 10 pitches and the last 10 pitches of an outing.

**The Insight**: By aggregating `mean_velo_drop` over the season for our flagger pitchers, we gain a quantifiable proxy for **in-game, late-inning mechanical compensation**. When pitchers fatigue and lose velocity, kinematic efficiency degrades, heavily transferring force load directly onto the UCL and shoulder capsule. Pitchers with high `acwr_spikes` AND high `mean_velo_drop` represent the most absolute critical risk tier before entering the playoffs."""

# Find the 'Determine...' cell and append right after or at the very end
found = False
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and 'Determine' in cell.source:
        cell.source += "\n\n" + new_cell_content
        found = True
        break

if not found:
    new_cell = nbformat.v4.new_markdown_cell(new_cell_content)
    nb.cells.append(new_cell)

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Updated notebook successfully!")
