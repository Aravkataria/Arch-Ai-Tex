import bpy
import sys
import os

# ---------------------------------------------------------
# Parse arguments after -- : input PNG and output GLB
# ---------------------------------------------------------
args = sys.argv
try:
    sep = args.index("--")
    input_png = args[sep + 1]
    output_glb = args[sep + 2]
except Exception as e:
    print("ERROR: Could not read arguments. Usage: blender --python script.py -- input.png output.glb")
    sys.exit(1)

# ---------------------------------------------------------
# Clean existing objects
# ---------------------------------------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ---------------------------------------------------------
# PNG → SVG (placeholder)
# REAL tracing must be done BEFORE Blender.
# ---------------------------------------------------------
svg_path = input_png.replace(".png", ".svg")
if not os.path.exists(svg_path):
    print(f"[WARN] SVG missing. Creating dummy square at: {svg_path}")
    with open(svg_path, "w") as f:
        f.write("""
        <svg height="100" width="100">
            <path d="M 0 0 L 200 0 L 200 200 L 0 200 Z" fill="black"/>
        </svg>
        """)

# ---------------------------------------------------------
# Import SVG
# ---------------------------------------------------------
print("[INFO] Importing SVG:", svg_path)
bpy.ops.import_curve.svg(filepath=svg_path)

# After import: collect all curve objects
curve_objs = [obj for obj in bpy.context.scene.objects if obj.type == "CURVE"]

if not curve_objs:
    print("[ERROR] No curves imported from SVG!")
    sys.exit(1)

# ---------------------------------------------------------
# Convert all Curves → 3D extruded walls
# ---------------------------------------------------------
for obj in curve_objs:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Wall thickness (bevel depth)
    obj.data.bevel_depth = 0.1       # 10 cm walls
    obj.data.extrude = 2.8           # height in meters

    # Convert curve + extrusion → mesh
    bpy.ops.object.convert(target='MESH')
    obj.select_set(False)

print("[INFO] Extrusion + conversion complete")

# ---------------------------------------------------------
# Export GLB
# ---------------------------------------------------------
print("[INFO] Exporting GLB to:", output_glb)
bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB')

print("[SUCCESS] 3D model generated:", output_glb)
