import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np
from mathutils import Vector, Euler
import mathutils
from math import pi
import math

class ArgumentParserForBlender(argparse.ArgumentParser):
    def _get_argv_after_doubledash(self):
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:]
        except ValueError as e:
            return []

    def parse_args(self):
        return super().parse_args(args=self._get_argv_after_doubledash())

parser = ArgumentParserForBlender()
##parser.add_argument('--blend_file', type=str, default='')
parser.add_argument('--mesh_file', type=str, default='')
parser.add_argument('--mtl_file', type=str, default='')
parser.add_argument('--results_path', type=str, default='')
##parser.add_argument('--text_file', type=str, default='')
args = parser.parse_args()

MESH_FILE = args.mesh_file
MTL_FILE = args.mtl_file
##TEXTURE_FILE = args.text_file
##BLEND_FILE = args.blend_file

DEBUG = False

VIEWS = 500
#VIEWS = 420
RESOLUTION = 256
RESULTS_PATH = args.results_path
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
RANDOM_VIEWS = True
#RANDOM_VIEWS = False
UPPER_VIEWS = False
CIRCLE_FIXED_START = (.3, 0, 0)

#bpy.ops.import_scene.obj(filepath=MESH_FILE, filter_glob=MTL_FILE)
#bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")
##bpy.ops.wm.open_mainfile(filepath=BLEND_FILE)
objs = bpy.data.objects
objs.remove(objs["Cube"], do_unlink=True)
objs.remove(objs["Light"], do_unlink=True)
bpy.ops.import_scene.obj(filepath=MESH_FILE, filter_glob=MTL_FILE, split_mode="OFF")

scene = bpy.context.scene
obs = []
for ob in scene.objects:
    if ob.type == "MESH":
        obs.append(ob)

ctx = bpy.context.copy()
ctx['active_object'] = obs[0]
ctx['selected_editable_objects'] = obs
bpy.ops.object.join(ctx)

light_data = bpy.data.lights.new(name='light', type='POINT')
light_object = bpy.data.objects.new(name='light', object_data=light_data)
scene.collection.objects.link(light_object)
light_object.location = (0.0, 0.0, 1.0)
light_object.data.energy = 20.0
light_object.data.color = (1, 1, 1)
light_object.data.shadow_soft_size = 0.25
light_object.select_set(state = True)
bpy.context.view_layer.objects.active = light_object
"""
key_light_data = bpy.data.lights.new(name='key light', type='POINT')
fill_light_data = bpy.data.lights.new(name='fill light', type='POINT')
back_light_data = bpy.data.lights.new(name='back light', type='POINT')

key_light_object = bpy.data.objects.new(name='key light', object_data=key_light_data)
fill_light_object = bpy.data.objects.new(name='fill light', object_data=fill_light_data)
back_light_object = bpy.data.objects.new(name='back light', object_data=back_light_data)

scene.collection.objects.link(key_light_object)
scene.collection.objects.link(fill_light_object)
scene.collection.objects.link(back_light_object)

key_light_object.location = (1.0, 1.0, 1.0)
fill_light_object.location = (-1.0, 1.0, 1.0)
back_light_object.location = (-1.0, -1.0, 1.0)

key_light_object.data.energy = 100
fill_light_object.data.energy = 100
back_light_object.data.energy = 100

key_light_object.data.color = (1, 1, 1)
fill_light_object.data.color = (1, 1, 1)
back_light_object.data.color = (1, 1, 1)

key_light_object.select_set(state = True)
fill_light_object.select_set(state = True)
back_light_object.select_set(state = True)

#scene.objects.active = key_light_object
#scene.objects.active = fill_light_object
#scene.objects.active = back_light_object

bpy.context.view_layer.objects.active = key_light_object
bpy.context.view_layer.objects.active = fill_light_object
bpy.context.view_layer.objects.active = back_light_object
"""
#bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")
mesh = bpy.context.selected_objects[0]

#mesh = bpy.context.selected_objects
#for O in bpy.data.meshes:
#    print(bpy.data.meshes[O.name])
#mesh = bpy.ops.object.select_all(action="SELECT").data

bpy.ops.object.transform_apply( rotation = True, scale = True )
minX = min( [vertex.co[0] for vertex in mesh.data.vertices] )
minY = min( [vertex.co[1] for vertex in mesh.data.vertices] )
minZ = min( [vertex.co[2] for vertex in mesh.data.vertices] )
minXYZ = min(minX, minY, minZ)
vMin = Vector( [minX, minY, minZ] )
maxDim = max(mesh.dimensions)
#
# Use mat_rot for GoogleScannedObjects and YCB
#mat_rot = mathutils.Matrix(((0, 0, -1), (1, 0, 0), (0, -1, 0)))
for v in mesh.data.vertices:
    v.co -= vMin
    v.co /= maxDim
    v.co -= Vector( [0.5, 0.5, 0.5] )
    #v.co = mat_rot @ v.co
bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")

##bpy.ops.image.open(filepath = TEXTURE_FILE)
fp = bpy.path.abspath("//{}".format(RESULTS_PATH))

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def get_extrinsics(cam):
    R_bcam2cv = mathutils.Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1 * R_world2bcam @ location
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam
    RT = mathutils.Matrix(
        (
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],),
        )
    )
    return RT

def get_intrinsics(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camd.sensor_fit == "VERTICAL":
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels
    K = mathutils.Matrix(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))
    return K

if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
#out_data = {
#    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
#}
out_data = []

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
#bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if not DEBUG:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.offset = [-0.7]
        map.size = [DEPTH_SCALE]
        map.use_min = True
        map.min = [0]
        links.new(render_layers.outputs['Depth'], map.inputs[0])

        links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background


objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    return b_empty

scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

cam = scene.objects['Camera']
cam.location = (2.5, 0.0, 0.0)
#cam.location = (1.0, 0.0, 0.0)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians

stepsize = 360.0 / VIEWS
rotation_mode = 'XYZ'

if not DEBUG:
    for output_node in [depth_file_output, normal_file_output]:
        output_node.base_path = ''

#out_data['frames'] = []

if not RANDOM_VIEWS:
    b_empty.rotation_euler = CIRCLE_FIXED_START

for i in range(0, VIEWS):
    if DEBUG:
        i = np.random.randint(0,VIEWS)
        b_empty.rotation_euler[2] += radians(stepsize*i)
    if RANDOM_VIEWS:
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        light_x_loc = np.cos(theta) * np.sin(phi)
        light_y_loc = np.sin(theta) * np.sin(phi)
        light_z_loc = np.cos(phi)
        light_object.location = (light_x_loc, light_y_loc, light_z_loc)
        scene.render.filepath = fp + '/r_' + str(i) + '.png'
        if UPPER_VIEWS:
            rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
            b_empty.rotation_euler = rot
        else:
            b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
    else:
        print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
        light_x_loc = 0.0
        light_y_loc = 0.0
        light_z_loc = 1.0
        light_object.location = (light_x_loc, light_y_loc, light_z_loc)
        scene.render.filepath = fp + '/r_{0:03d}.png'.format(int(i * stepsize))

    # depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
    # normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'filename': scene.render.filepath,
        'RT': np.array(get_extrinsics(cam)).tolist(),
        'K': np.array(get_intrinsics(cam.data)).tolist(),
        'resolution': RESOLUTION,
        'light_pos': [light_x_loc, light_y_loc, light_z_loc],
        'light_rgb': None,
        'objects': {}
    }
    #out_data['frames'].append(frame_data)
    out_data.append(frame_data)

    if RANDOM_VIEWS:
        if UPPER_VIEWS:
            rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
            b_empty.rotation_euler = rot
        else:
            b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
    else:
        b_empty.rotation_euler[2] += radians(stepsize)

if not DEBUG:
    with open(fp + '/' + 'anno.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

#bpy.ops.wm.save_as_mainfile(filepath="results_500/normalized.blend")
