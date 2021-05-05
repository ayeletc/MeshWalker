import os
import numpy as np
import pyvista as pv
import pandas as pd
import trimesh
from matplotlib import pyplot as plt


def plot_heat_map(mesh_path='', attention=None):
    if len(mesh_path) == 0:
        return -1
    mesh = load_model_from_npz(npz_path=mesh_path)
    file_id = mesh_path.split('/')[-1]
    file_id = file_id.split('_')[1]

    vertices = np.asarray(mesh['vertices'])
    faces = np.asarray(mesh['faces'])
    mesh_to_show = vf_to_pd(vertices, faces)
    color_scalar_vector = set_colors(vertices, attention)
    attention_raw = attention.iloc[0].to_numpy()[1:]
    attention_val = attention_raw[:100]
    attention_ver = attention_raw[100:].reshape((100, 3))
    attention_full_len = np.zeros(vertices.shape[0])
    for ii in range(attention_full_len.shape[0]):
        for jj in range(attention_val.shape[0]):
            if (vertices[0, :] == attention_ver[jj, :]).all():
                attention_full_len[ii] = attention_val[jj]
    # color_scalar_vector = set_colors(attention_ver, attention_val)
    color_scalar_vector = set_colors(vertices, attention_full_len)
    red_to_blue_cmap = plt.cm.get_cmap("seismic", len(color_scalar_vector))

    plotter = pv.Plotter()
    plotter.add_mesh(mesh_to_show, scalars=color_scalar_vector, cmap=red_to_blue_cmap)
    plotter.add_text(file_id)
    plotter.plot()


def load_model_from_npz(npz_path):
    if npz_path.find(':') != -1:
        npz_path = npz_path.split(':')[1]
    mesh_data = np.load(npz_path, encoding='latin1', allow_pickle=True)
    return mesh_data


def set_colors(vertices, dist_measure):
    """
    Creates a list of colors between red and blue, by the order of the the size of the change.
    Red - most changed vertex
    Blue - least changed vertex
    :param vertices: The list of vertices
    :param dist_measure: The list of the changes in every vertex
    :return: A list of numbers, according to the distance.
    The higher the distant the closer the number will be to 1.
    """

    if len(vertices) != len(dist_measure):
        return None
    else:
        len_of_vertex_list = len(vertices)

    colors_vector = np.zeros(shape=(len_of_vertex_list))
    ordered_dist_measure = np.argsort(dist_measure)

    for i in range(len(dist_measure)):
        color = 1 - i/len(dist_measure)
        colors_vector[ordered_dist_measure[i]] = color

    return colors_vector



def vf_to_pd(v, f=None, nd=3):
    """
    converting (v,f) triangular mesh to PolyData structure
    currently supports a single type of face - triangles
    :param v: vertices array
    :param f: faces array
    :param nd: number of vertices in a face (3 for triangles)
    :return: a PolyData structure
    """
    if f is None:
        return pv.PolyData(v)
    f_pad = nd * np.ones((f.shape[0], f.shape[1]+1))
    f_pad[:, 1:] = f
    return pv.PolyData(v, f_pad)


# def visualize_model(vertices, faces_, title=' ', walk=None, opacity=1.0,
#                     all_colors='white', face_colors=None, cmap=None, edge_colors=None, edge_color_a='white',
#                     line_width=1, show_edges=True):
#     p = pv.Plotter()
#     faces = np.hstack([[3] + f.tolist() for f in faces_])
#     surf = pv.PolyData(vertices, faces)
#     p.add_mesh(surf, show_edges=show_edges, edge_color=edge_color_a, color=all_colors, opacity=opacity, smooth_shading=True,
#                scalars=face_colors, cmap=cmap, line_width=line_width)
#     all_edges = [[2, walk[i], walk[i + 1]] for i in range(len(walk) - 1)]
#     walk_edges = np.hstack([edge for edge in all_edges])
#     walk_mesh = pv.PolyData(vertices, walk_edges)
#
#     p.add_mesh(walk_mesh, show_edges=True, line_width=line_width * 4, edge_color=edge_colors)
#     cpos = p.show(title=title)

def visualize_model(vertices, faces_, title=' ', walk=None, opacity=1.0,
                    all_colors='white', face_colors=None, cmap=None, edge_colors=None, edge_color_a='white',
                    line_width=1, show_edges=True , mesh_path='', attention=None):
    if len(mesh_path) == 0:
        return -1
    mesh = load_model_from_npz(npz_path=mesh_path)
    file_id = mesh_path.split('/')[-1]
    file_id = file_id.split('_')[1]

    vertices = np.asarray(mesh['vertices'])
    faces = np.asarray(mesh['faces'])
    mesh_to_show = vf_to_pd(vertices, faces)
    color_scalar_vector = set_colors(vertices, attention)
    attention_raw = attention.iloc[0].to_numpy()[1:]
    attention_val = attention_raw[:100]
    attention_ver = attention_raw[100:].reshape((100, 3))
    attention_full_len = np.zeros(vertices.shape[0])
    for ii in range(attention_full_len.shape[0]):
        for jj in range(attention_val.shape[0]):
            if (vertices[0, :] == attention_ver[jj, :]).all():
                attention_full_len[ii] = attention_val[jj]
    # color_scalar_vector = set_colors(attention_ver, attention_val)
    color_scalar_vector = set_colors(vertices, attention_full_len)
    red_to_blue_cmap = plt.cm.get_cmap("seismic", len(color_scalar_vector))

    p = pv.Plotter()
    faces = np.hstack([[3] + f.tolist() for f in faces_])
    surf = pv.PolyData(vertices, faces)
    p.add_mesh(surf, show_edges=show_edges, edge_color=edge_color_a, color=all_colors, opacity=opacity, smooth_shading=True,
               scalars=face_colors, cmap=cmap, line_width=line_width)
    all_edges = [[2, walk[i], walk[i + 1]] for i in range(len(walk) - 1)]
    walk_edges = np.hstack([edge for edge in all_edges])
    walk_mesh = pv.PolyData(vertices, walk_edges)

    p.add_mesh(walk_mesh, show_edges=True, line_width=line_width * 4, edge_color=red_to_blue_cmap)
    cpos = p.show(title=title)

def get_mesh(mesh_path):
  # mesh = trimesh.load_mesh('/home/alonla/mesh_walker/datasets_raw/sig17_seg_benchmark/meshes/test/shrec/2.off')
  mesh = trimesh.load_mesh(mesh_path)
  mesh_data = {'vertices': mesh.vertices, 'faces': mesh.faces, 'n_vertices': mesh.vertices.shape[0]}
  prepare_edges_and_kdtree(mesh_data)
  return mesh_data


def prepare_edges_and_kdtree(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
  mesh['edges'] = [set() for _ in range(vertices.shape[0])]
  for i in range(faces.shape[0]):
    for v in faces[i]:
      mesh['edges'][v] |= set(faces[i])
  for i in range(vertices.shape[0]):
    if i in mesh['edges'][i]:
      mesh['edges'][i].remove(i)
    mesh['edges'][i] = list(mesh['edges'][i])
  max_vertex_degree = np.max([len(e) for e in mesh['edges']])
  for i in range(vertices.shape[0]):
    if len(mesh['edges'][i]) < max_vertex_degree:
      mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
  mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)

  mesh['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  n_nbrs = min(10, vertices.shape[0] - 2)
  for n in range(vertices.shape[0]):
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
  assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(mesh['kdtree_query'].shape[1])

###### ---------------------------------------------------------------- ######

def jump_to_closest_unviseted(model_kdtree_query, model_n_vertices, walk, enable_super_jump=True):
  for nbr in model_kdtree_query[walk[-1]]:
    if nbr not in walk:
      return nbr

  if not enable_super_jump:
    return None

  # If not fouind, jump to random node
  node = np.random.randint(model_n_vertices)

  return node


def get_seq_random_walk_no_jumps(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    if len(nodes_to_consider):
      to_add = np.random.choice(nodes_to_consider)
      jump = False
    else:
      if i > backward_steps:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        to_add = np.random.randint(n_vertices)
        jump = True
    seq[i] = to_add
    jumps[i] = jump
    visited[to_add] = 1

  return seq, jumps


def get_seq_random_walk_random_global_jumps(mesh_extra, f0, seq_len):
  MEIR_WALK = 0
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_prob = 1 / 100
  dont_check_visited_prob = 5 / 100
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    if MEIR_WALK and np.random.binomial(1, dont_check_visited_prob):
      nodes_to_consider = this_nbrs
    else:
      nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob)
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = False
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps


def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['n_vertices']
  kdtr = mesh_extra['kdtree_query']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  for i in range(1, seq_len + 1):
    b = min(0, i - 20)
    to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
    if len(to_consider):
      seq[i] = np.random.choice(to_consider)
      jumps[i] = False
    else:
      seq[i] = np.random.randint(n_vertices)
      jumps[i] = True
      visited = np.zeros((n_vertices + 1,), dtype=np.bool)
      visited[-1] = True
    visited[seq[i]] = True

  return seq, jumps


#### -------------------------------------------------- #########


def show_walk_on_mesh():
    mesh_path = r'/home/amirayellet/repo/shrec11/raw/T32.off'
    mesh = get_mesh(mesh_path)
    walk, jumps = get_seq_random_walk_no_jumps(mesh, f0=0, seq_len=400)
    visualize_model(mesh['vertices'], mesh['faces'], line_width=1, show_edges=1,
                    walk=walk, edge_colors='red')


def show_attention_on_mesh(pickle_path, type='heatmap'):
    walk_id = 5
    mesh_df = pd.read_pickle(pickle_path)
    attention = mesh_df.iloc[walk_id][1:].values
    if type == 'heatmap':
        mesh_name = mesh_df.loc[walk_id, 'name']
        shrec_datasets_dir = r'/home/amirayellet/repo/versions/MeshWalker-master'
    elif type == 'visualize':
        mesh_name = mesh_df.loc[walk_id, 'name'].split('/')[-1].split('_')[0] + '.off'
        shrec_datasets_dir = r'/home/amirayellet/repo/shrec11/raw'
    mesh_path = os.path.join(shrec_datasets_dir, mesh_name)

    if type == 'visualize':
        mesh = get_mesh(mesh_path)
        walk, jumps = get_seq_random_walk_no_jumps(mesh, f0=0, seq_len=100)
        visualize_model(mesh['vertices'], mesh['faces'], line_width=1, show_edges=1,
                    walk=walk, edge_colors='red', mesh_path=mesh_path, attention=mesh_df)
    elif type == 'heatmap':
        plot_heat_map(mesh_path, mesh_df)

if __name__ == '__main__':
    # show_walk_on_mesh()
    attention_path = r'/home/amirayellet/repo/versions/MeshWalker-master/runs/0163-05.05.2021..14.08__shrec11_16-04_A/attention/50/120'
    show_attention_on_mesh(attention_path, type='heatmap')
