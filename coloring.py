import pyvista as pv
from pyvista import examples
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import glob, os, copy, sys
import distance_metric
#import utils


def pyvista_example():
    mesh = examples.download_st_helens().warp_by_scalar()
    # Add scalar array with range (0, 100) that correlates with elevation
    mesh['values'] = pv.plotting.normalize(mesh['Elevation']) * 100

    # Define the colors we want to use
    blue = np.array([12/256, 238/256, 246/256, 1])
    black = np.array([11/256, 11/256, 11/256, 1])
    grey = np.array([189/256, 189/256, 189/256, 1])
    yellow = np.array([255/256, 247/256, 0/256, 1])
    red = np.array([1, 0, 0, 1])

    mapping = np.linspace(mesh['values'].min(), mesh['values'].max(), 256)
    newcolors = np.empty((256, 4))
    newcolors[mapping >= 80] = red
    newcolors[mapping < 80] = grey
    newcolors[mapping < 55] = yellow
    newcolors[mapping < 30] = blue
    newcolors[mapping < 1] = black

    # Make the colormap from the listed colors
    my_colormap = ListedColormap(newcolors)

    mesh.plot(scalars='values', cmap=my_colormap)


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


def render_surface(vertices, faces, scalar_func, paintby='faces', gain=100):
    """
    renders both faces and edges of the mesh
    :param vertices: np.array of the vertices
    :param faces: np.array of the faces
    :param scalar_func: coloring vector for the faces or color values for the vertices
    :param centering: Boolean value - center the shape around the (0,0,0) or not
    :param paintby: flag to color 'faces' or 'edges'
    :param gain: gain for arrows
    :param clip: clipping for curvature values
    :return: void
    """
    surf = vf_to_pd(vertices, faces)
    pv.set_plot_theme("night")  # white = "document"
    plotter = pv.Plotter()
    if paintby == 'faces':
        plotter.add_mesh(surf, style='surface', scalars=scalar_func,
                         show_edges=True, stitle='Colormap')  # style='wireframe' will show only edges

    elif paintby == 'vertices':
        plotter.add_mesh(surf, style='surface', scalars=scalar_func,
                         show_edges=True, stitle='Colormap')
    else:
        plotter.add_mesh(surf, style='surface', show_edges=True, stitle='Colormap')
    plotter.show_axes()
    plotter.show()


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


def red_or_blue(vertices, dist_measure):
    """
    Creates a list of colors between red and blue, by the order of the the size of the change.
    Red - 20% most changed vertices.
    Blue - 80% least changed vertices
    :param vertices: The list of vertices
    :param dist_measure: The list of the changes in every vertex
    :return: A list of numbers, according to the distance.
    The higher the distant the closer the number will be to 1.
    """

    if len(vertices) != len(dist_measure):
        return None
    else:
        len_of_vertex_list = len(vertices)

    colors_vector = np.zeros(shape=(len(vertices), 4))
    ordered_dist_measure = np.argsort(dist_measure)
    most_changed_prec = 0.2

    # The number of vertices that will be colored red
    last_red_idx = int(len(vertices)*most_changed_prec)

    blue = np.array([0, 1, 0, 1])
    red = np.array([1, 0, 0, 1])

    for i in range(len(dist_measure)):
        color = red if i <= last_red_idx else blue #np.array([1 - i / len(dist_measure), i / len(dist_measure), 0, 1])
        colors_vector[ordered_dist_measure[i]] = color

    return colors_vector



def show_model(model):
    utils.visualize_model(model['vertices'], model['faces'])
    return


def load_model_from_npz(npz_path):
    if npz_path.find(':') != -1:
        npz_path = npz_path.split(':')[1]
    mesh_data = np.load(npz_path, encoding='latin1', allow_pickle=True)
    return mesh_data


def show_model_old_main():
    print("in main")

    npz_paths = ['datasets_processed/shrec11/16-04_a/train/T500_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T520_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T358_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T546_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T80_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T96_not_changed_500.npz']
    dog1 = ['datasets_processed/shrec11/16-04_a/train/T476_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T144_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T331_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T125_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T393_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T436_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T197_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T367_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T409_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T373_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T193_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T309_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T99_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T136_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T354_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T203_not_changed_500.npz']
    dog2 = ['datasets_processed/shrec11/16-04_a/train/T504_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T93_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T207_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T507_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T8_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T189_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T467_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T468_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T267_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T40_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T182_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T445_not_changed_500.npz'	,'datasets_processed/shrec11/16-04_a/train/T582_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T178_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T250_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T459_not_changed_500.npz']

    npz_paths = []
    for d1, d2 in zip(dog1, dog2):
        npz_paths += [d1]
        npz_paths += [d2]

    for npz_path in npz_paths:
        model = load_model_from_npz(npz_path)
        if model is not None:
            show_model(model)

    return 0


def plot_heat_map(orig_mesh_path = '', attacked_mesh_path = ''):
    if len(orig_mesh_path) == 0 or len(attacked_mesh_path) == 0:
        return -1
    orig_mesh = load_model_from_npz(npz_path=orig_mesh_path)
    file_id = orig_mesh_path.split('/')[-1]
    file_id = file_id.split('_')[1]

    orig_vertices = np.asarray(orig_mesh['vertices'])
    orig_faces = np.asarray(orig_mesh['faces'])
    mesh_to_show = vf_to_pd(orig_vertices, orig_faces)
    # Get the correct dist measure
    _, _, dist_measure = distance_metric.check_dist_between_2_models(model_before_attack=orig_mesh_path, model_after_attack=attacked_mesh_path)
    color_scalar_vector = set_colors(orig_vertices, dist_measure)
    red_to_blue_cmap = plt.cm.get_cmap("seismic", len(color_scalar_vector))
    #render_surface(vertices=vertices, faces=faces, paintby='vertices', gain=100)
    #my_colormap = ListedColormap(newcolors)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_to_show, scalars=color_scalar_vector, cmap=red_to_blue_cmap)
    plotter.add_text(file_id)
    plotter.plot()
    #plotter.show(screenshot='heat_mpa_pictures/'+file_id+'.png')
    #plotter.show()



    return 0



def show_randomness(orig_mesh_path = '', attacked_mesh_path = ''):
    if len(orig_mesh_path) == 0 or len(attacked_mesh_path) == 0:
        return -1
    orig_mesh = load_model_from_npz(npz_path=orig_mesh_path)
    file_id = orig_mesh_path.split('/')[-1]
    file_id = file_id.split('_')[1]

    orig_vertices = np.asarray(orig_mesh['vertices'])
    orig_faces = np.asarray(orig_mesh['faces'])
    mesh_to_show = vf_to_pd(orig_vertices, orig_faces)
    # Get the correct dist measure
    _, _, dist_measure = distance_metric.check_dist_between_2_models(model_before_attack=orig_mesh_path, model_after_attack=attacked_mesh_path)
    red_or_blue_vector = red_or_blue(orig_vertices, dist_measure)
    my_colormap = ListedColormap(red_or_blue_vector)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_to_show, scalars=red_or_blue_vector)
    plotter.add_text(file_id)
    plotter.plot()
    return


def main():

    # cat - almost the same
    #attacked_mesh_path = 'datasets_processed/shrec11/test_T555_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/rabbit_T555/last_model.npz'
    #orig_mesh_path = 'datasets_processed/shrec11/test_T555_not_changed_500.npz'


    # centaur - almost the same
    #attacked_mesh_path = 'datasets_processed/shrec11/test_T565_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/centaur_T565_ditto_meshCNN/last_model.npz'
    #orig_mesh_path = 'datasets_processed/shrec11/test_T565_not_changed_500.npz'

    #two balls - almost the same
    #attacked_mesh_path = 'datasets_processed/shrec11/test_T327_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/glasses_T327/last_model.npz'
    #orig_mesh_path = 'datasets_processed/shrec11/test_T327_not_changed_500.npz'

    #man - an ok example
    #attacked_mesh_path = 'datasets_processed/shrec11/test_T519_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/man_T519_ditto_meshCNN/last_model.npz'
    #orig_mesh_path = 'datasets_processed/shrec11/test_T519_not_changed_500.npz'

    #dino - ok example
    #attacked_mesh_path = 'datasets_processed/shrec11/test_dinosaur_T361_simplified_to_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/bird2_T361/last_model.npz'
    #orig_mesh_path = 'datasets_processed/shrec11/test_T361_not_changed_500.npz'

    #bird 2 - almost the same
    #attacked_mesh_path = 'datasets_processed/shrec11/test_T80_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/dog2_T80/last_model.npz'
    #orig_mesh_path = 'datasets_processed/shrec11/test_T80_not_changed_500.npz'

    # alien - ok example
    attacked_mesh_path = 'datasets_processed/shrec11/test_T124_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/alien_T124_ditto_meshCNN/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T124_not_changed_500.npz'

    # flamingo - ok example
    #attacked_mesh_path = 'datasets_processed/shrec11/test_T131_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/laptop_T131/last_model.npz'
    #orig_mesh_path = 'datasets_processed/shrec11/test_T131_not_changed_500.npz'


    # octopus - good example
    #attacked_mesh_path = 'datasets_processed/shrec11/test_T562_not_changed_500_attacked.npz'
    attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/octopus_T562_ditto_meshCNN/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T562_not_changed_500.npz'

    # camel - almost the same
    #attacked_mesh_path = 'datasets_processed/shrec11/test_T497_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/ants_T497/last_model.npz'
    #orig_mesh_path = 'datasets_processed/shrec11/test_T497_not_changed_500.npz'
    #orig_mesh_path = 'datasets_processed/shrec11/test_T497_not_changed_500_attacked.npz'

    # dino skel - good example
    attacked_mesh_path = 'datasets_processed/shrec11/test_T554_not_changed_500_attacked.npz'
    attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/snake_T554/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T554_not_changed_500.npz'

    """""
    # dog1 - almost the same
    attacked_mesh_path = 'datasets_processed/shrec11/test_T125_not_changed_500_attacked.npz'
    attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/shark_T125/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T125_not_changed_500.npz'

    # gorila - almost the same
    attacked_mesh_path = 'datasets_processed/shrec11/test_T105_not_changed_500_attacked.npz'
    attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/gorilla_T105_ditto_meshCNN/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T105_not_changed_500.npz'

    # pliers - almost the same
    attacked_mesh_path = 'datasets_processed/shrec11/test_T107_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/pliers_T107_ditto_meshCNN/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T107_not_changed_500.npz'

    # hand - almost the same
    attacked_mesh_path = 'datasets_processed/shrec11/test_T150_not_changed_500_attacked.npz'
    attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/hand_T150_ditto_meshCNN/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T150_not_changed_500.npz'

    # spiders - almost the same
    #attacked_mesh_path = 'datasets_processed/shrec11/test_T108_not_changed_500_attacked.npz'
    attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/myScissor_T108/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T108_not_changed_500.npz'

    # lamp - ok example
    attacked_mesh_path = 'datasets_processed/shrec11/test_T113_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/lamp_T113/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T113_not_changed_500.npz'

    # snake - ok example
    attacked_mesh_path = 'datasets_processed/shrec11/test_T350_not_changed_500_attacked.npz'
    attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/snake_T350_ditto_meshCNN/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T350_not_changed_500.npz'

    # dog2 - ok example
    attacked_mesh_path = 'datasets_processed/shrec11/test_T178_not_changed_500_attacked.npz'
    #attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/dog2_T178_ditto_meshCNN/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T178_not_changed_500.npz'

    # horse - almost the same
    attacked_mesh_path = 'datasets_processed/shrec11/test_T117_not_changed_500_attacked.npz'
    attacked_mesh_path = '../Downloads/ditto attacked meshCNN files/hand_T117/last_model.npz'
    orig_mesh_path = 'datasets_processed/shrec11/test_T117_not_changed_500.npz'
    """
    # modelnet
    #attacked_mesh_path = '../Downloads/models_to_heat/last_model.npz'
    #orig_mesh_path = '../Downloads/models_to_heat/test_bathtub_0118_simplified_to_4000.npz'
    #attacked_mesh_path = '../Downloads/models_to_heat(1)/models_to_heat/last_model.npz'
    #orig_mesh_path = '../Downloads/models_to_heat(1)/models_to_heat/test_plant_0274_simplified_to_2000.npz'

    plot_heat_map(orig_mesh_path=orig_mesh_path, attacked_mesh_path=attacked_mesh_path)
    show_randomness(orig_mesh_path=orig_mesh_path, attacked_mesh_path=attacked_mesh_path)

if __name__ == '__main__':
    print("amir")
    main()
