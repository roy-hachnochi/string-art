from collections import defaultdict, OrderedDict
from typing import List, Tuple, Any, Union, Dict, Optional, get_origin, get_args
import bisect
import pickle
from copy import deepcopy

import numpy as np

from string_art.canvas import MulticolorCanvas
from string_art.configs import CanvasConfig
from string_art.utils import load_image
from string_art.globals import PATH_TYPE, CONTINUOUS_PATH_TYPE, LINES_TYPE, COLOR_LINES_TYPE, COLOR_TYPE
from string_art.utils.pdf_generator import save_pdf_instructions

_TYPE_ERR_MSG = 'lines is of wrong type, should be Union[LINES_TYPE, COLOR_LINES_TYPE, Dict[COLOR_TYPE, LINES_TYPE]].'


class FibersPath:
    def __init__(self, lines: Union[str, LINES_TYPE, COLOR_LINES_TYPE, Dict[COLOR_TYPE, LINES_TYPE]]):
        """
        Saves a solution for the string-art optimization problem.
        :param lines: fibers to reconstruct the image, one of these options:
            - List[int] - a list of nails that form lines using a continuous fiber.
            - List[Tuple[int, int]] - a list of nail pairs indicating lines, unnecessarily continuous.
            - List[Tuple[Tuple[int, int, int], LINES_TYPE]] - a list of ordered lines (either of the above) and their
            color, indicated by an RGB tuple.
            - Dict[Tuple[int, int, int], LINES_TYPE] - a dictionary of unordered lines per color.
        """
        if isinstance(lines, str):
            self.load_fibers(lines)
        else:
            self.lines = deepcopy(lines)
            self._set_properties()
            self.original_lines = deepcopy(self.lines)

    def _set_properties(self):
        """
        Check data type validity, and store data type indicators.
        """
        if check_type(self.lines, CONTINUOUS_PATH_TYPE):
            self.is_color, self.is_continuous, self.is_dict = False, True, False
        elif check_type(self.lines, PATH_TYPE):
            self.is_color, self.is_continuous, self.is_dict = False, False, False
        elif check_type(self.lines, Dict[COLOR_TYPE, LINES_TYPE]):
            self.is_color, self.is_dict = True, True
            self.is_continuous = all(check_type(val, CONTINUOUS_PATH_TYPE) for val in self.lines.values())
        elif check_type(self.lines, COLOR_LINES_TYPE):
            self.is_color, self.is_dict = True, False
            self.is_continuous = all(check_type(val[1], CONTINUOUS_PATH_TYPE) for val in self.lines)
        else:
            raise ValueError(_TYPE_ERR_MSG)

    def convert_lines_to_path(self, nails: int):
        """
        If lines are given as a list of lines (PATH_TYPE) rather than a continuous path (CONTINUOUS_PATH_TYPE),
        convert them to a continuous path. If necessary, adds lines around the perimeter of the canvas to connect
        unconnected components.
        :param nails: number of nails in the canvas.
        """
        if self.is_continuous:  # already continuous, nothing to do
            return
        if self.is_dict:  # apply on every value of the dict
            for color in self.lines:
                paths = cover_eulerian_paths(self.lines[color])
                self.lines[color] = connect_paths(paths, nails)
        elif self.is_color:  # apply on every path in each color-lines tuple
            for i, color_path in enumerate(self.lines):
                paths = cover_eulerian_paths(color_path[1])
                path = connect_paths(paths, nails)
                self.lines[i] = (color_path[0], path)
        else:
            paths = cover_eulerian_paths(self.lines)
            self.lines = connect_paths(paths, nails)
        self.is_continuous = True

    def trim_lines(self, max_lines: int):
        """
        Trim path to use no more than max_lines fibers.
        :param max_lines: Maximal number of lines to use.
        # TODO: it's more elegant to implement __iter__ and __get_item__ methods.
        """
        n_lines = len(self)
        if n_lines <= max_lines:
            return
        if self.is_color and self.is_dict:  # dict of {color: path}
            p = max_lines / n_lines
            n_cum = 0
            for i, color in enumerate(self.lines):
                if i < len(self.lines) - 1:
                    n = int(len(self.lines[color]) * p)
                    self.lines[color] = self.lines[color][:n]
                    n_cum += n
                else:
                    self.lines[color] = self.lines[color][:(max_lines - n_cum)]
        elif self.is_color and not self.is_dict:  # list of (color, path) tuples
            lines = []
            it = 0
            for color, path in self.lines:
                lines.append((color, []))
                for nail in path:
                    lines[-1][1].append(nail)
                    it += 1
                    if it >= max_lines:
                        break
                if it >= max_lines:
                    break
            self.lines = lines
        else:  # simple list of lines
            self.lines = self.lines[:max_lines]

    def interweave_color_paths(self, interval: float = 0.3):
        """
        Combine a dictionary of per-color paths to a single continuous path traversing between colors (COLOR_LINES_TYPE).
        The combined path will interweave the colors, each interval starting from lightest to darkest color.
        Assuming that each per-color path is ordered by its line importance, make the final path have the most important
        lines on top (in the end of the path).
        :param interval: interweaving interval to switch between colors (0 < interval <= 1).
        :return: list of tuples representing the combined path, each tuple is (color, path segment), each color is a
            3-tuple representing the color in RGB uint8, and each path segment is a list representing the lines in the
            path of this color (see COLOR_LINES_TYPE).
        """
        if not (self.is_dict and self.is_color):
            return

        # order colors from lightest to darkest
        colors = [color for color in self.lines.keys()]
        colors_np = np.array(colors, dtype=np.uint8)
        color_paths = [path for path in self.lines.values()]
        luminance = 0.2126 * colors_np[:, 0] + 0.7152 * colors_np[:, 1] + 0.0722 * colors_np[:, 2]
        sorted_indices = list(np.argsort(luminance)[::-1])
        colors = [colors[i] for i in sorted_indices]
        # assuming each path is ordered by its line importance, so flip paths to have the most important line on top
        color_paths = [color_paths[i][::-1] for i in sorted_indices]

        # interweave paths by interval
        path_interweaved = []
        segment = 0
        start_inds = [0] * len(colors)
        while segment < 1:
            segment += interval
            for color_i in range(len(colors)):
                end_ind = min(int(segment * len(color_paths[color_i])), len(color_paths[color_i]))
                cur_path = color_paths[color_i][start_inds[color_i]:end_ind]
                path_interweaved.append((colors[color_i], cur_path))
                start_inds[color_i] = end_ind

        self.lines = path_interweaved
        self.is_dict = False

    def combine_color_paths(self, image: Union[str, np.array], canvas_cfg: CanvasConfig, min_path_length: int = 1):
        """
        Combine a dictionary of per-color paths to a single continuous path traversing between colors (COLOR_LINES_TYPE).
        The combined path will interweave the colors based on their importance for the image reconstruction (by
        simulating the error).
        :param image: either a path to an image or a (c, h, w) array representing an image (scaled to [0, 1]).
        :param canvas_cfg: config of canvas to use for error simulation, resolution will be set according to image.
        :param min_path_length: minimal intermediate path length per color.
        :return: list of tuples representing the combined path, each tuple is (color, path segment), each color is a
            3-tuple representing the color in RGB uint8, and each path segment is a list representing the lines in the
            path of this color (see COLOR_LINES_TYPE).
        """
        if not (self.is_dict and self.is_color):
            return

        if isinstance(image, str):
            image = load_image(image, grayscale=False)
        canvas_cfg = deepcopy(canvas_cfg)
        canvas_cfg.resolution = image.shape[-2:]
        canvas = MulticolorCanvas(canvas_cfg)

        # initializations
        start_pointer = 1 if self.is_continuous else 0
        cur_pointer_per_color = {color: start_pointer for color in self.lines}
        color_path_lengths = {color: len(lines) for color, lines in self.lines.items()}
        path_interweaved = []

        # iterate colors and choose best one next
        while not all([cur_pointer_per_color[color] >= color_path_lengths[color] for color in cur_pointer_per_color]):
            # find next best color
            scores = defaultdict(float)
            for color, lines in self.lines.items():
                if cur_pointer_per_color[color] < color_path_lengths[color]:
                    # scores of adding next lines from this color
                    for ind in range(cur_pointer_per_color[color],
                                     min(cur_pointer_per_color[color] + min_path_length, color_path_lengths[color])):
                        line = (lines[ind - 1], lines[ind]) if self.is_continuous else lines[ind]
                        color_np = np.array(color, dtype=np.float32) / 255.
                        scores[color] += canvas.simulate_line_improvement(image, *line, color_np)
                else:  # already finished this color
                    scores[color] = float('-inf')
            next_color = max(scores, key=scores.get)
            color_np = np.array(next_color, dtype=np.float32) / 255.

            # append min_path_length lines from next color to path
            path = []
            if self.is_continuous and cur_pointer_per_color[next_color] == 1:  # add starting nail
                path.append(self.lines[next_color][0])
            for _ in range(min_path_length):
                if cur_pointer_per_color[next_color] >= color_path_lengths[next_color]:  # finished path for this color
                    break
                path.append(self.lines[next_color][cur_pointer_per_color[next_color]])

                # update canvas
                if self.is_continuous:
                    canvas.add_fiber(self.lines[next_color][cur_pointer_per_color[next_color] - 1],
                                     self.lines[next_color][cur_pointer_per_color[next_color]],
                                     color_np,
                                     opaque=False)
                else:
                    canvas.add_fiber(*self.lines[next_color][cur_pointer_per_color[next_color]], color_np, opaque=False)

                cur_pointer_per_color[next_color] += 1

            # combine with prev path if same color, or add new color path
            if len(path_interweaved) > 0 and path_interweaved[-1][0] == next_color:
                path_interweaved[-1][1].extend(path)
            else:
                path_interweaved.append((next_color, path))

        # reverse lines because they're ordered by decreasing importance
        self.lines = [(color, path[::-1]) for color, path in reversed(path_interweaved)]
        self.is_dict = False

    def save_instructions(self, pdf_path: str, canvas_cfg: CanvasConfig, image_path: Optional[str]):
        """
        Create and save a PDF with instructions on how to create a string-art using the stored lines.
        :param pdf_path: path to save PDF to.
        :param canvas_cfg: Canvas config for canvas information.
        :param image_path: optional path to preview image to add to PDF.
        """
        if not self.is_continuous:
            print('Path is not continuous, please call self.convert_lines_to_path() and run again.')
            return
        if self.is_dict:
            print('Color paths are not combined, please call self.combine_color_paths() or self.interweave_color_paths() and run again.')
            return
        if self.is_color:
            color_lines = self.lines
        else:
            color_lines = [((0, 0, 0), self.lines)]
        save_pdf_instructions(color_lines, pdf_path, canvas_cfg, image_path)

    def load_fibers(self, path: str):
        with open(path, 'rb') as f:
            self.lines = pickle.load(f)
        self.original_lines = deepcopy(self.lines)
        self._set_properties()

    def save_fibers(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.lines, f)

    def reset(self):
        self.lines = deepcopy(self.original_lines)

    def __len__(self):
        if self.is_dict:
            length = sum([len(lines) for lines in self.lines.values()])
        elif self.is_color:
            length = sum([len(lines) for color, lines in self.lines])
        else:
            length = len(self.lines)
        return length

    @property
    def n_colors(self):
        if self.is_color:
            if self.is_dict:
                return len(self.lines)
            else:
                return len(set([color for color, lines in self.lines]))
        else:
            return 1


# ======================================================================================================================
def check_type(value, expected_type) -> bool:
    """Check if a given value matches an expected type."""
    origin = get_origin(expected_type)  # get the base type (e.g., list, dict, tuple)
    args = get_args(expected_type)  # get the inner types
    if origin is None:  # non-generic types (int, str, etc.)
        return isinstance(value, expected_type)
    
    if origin is Union:
        return any(check_type(value, subtype) for subtype in args)
    if origin is list:
        if not isinstance(value, list):
            return False
        inner_type = args[0]
        return all(check_type(item, inner_type) for item in value)
    if origin is tuple:
        if not isinstance(value, tuple) or len(value) != len(args):
            return False
        return all(check_type(item, arg_type) for item, arg_type in zip(value, args))
    if origin is dict:
        if not isinstance(value, dict):
            return False
        key_type, value_type = args
        return all(check_type(k, key_type) and check_type(v, value_type) for k, v in value.items())
    if isinstance(expected_type, type):
        return isinstance(value, expected_type)
    return False


def cover_eulerian_paths(edges: PATH_TYPE) -> List[CONTINUOUS_PATH_TYPE]:
    """
    Find all eulerian paths in an undirected graph. An eulerian path is a path that covers all edges exactly once.
    The general graph might not have an eulerian path (see Euler's theorem on graphs), so we relax the problem to
    finding multiple eulerian paths that when combined cover the entire graph.
    :param edges: a list of tuples representing start node and end node of edges in graph.
    :return: paths: a list of eulerian paths which covers all edges, each is a list of nodes representing the path.
    """
    # construct adjacency graph
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    # heuristic: sort adjacency lists by ascending ranks, and graph by ascending ranks with odd ranks coming first
    rank = {u: len(graph[u]) for u in graph}
    for u in graph:
        graph[u] = list(sorted(graph[u], key=lambda u: rank[u]))
    graph = OrderedDict(sorted(graph.items(), key=lambda item: (rank[item[0]] % 2 == 0, rank[item[0]])))
    rank = OrderedDict(sorted(rank.items(), key=lambda item: (rank[item[0]] % 2 == 0, rank[item[0]])))
    n_edges = sum(x for x in rank.values())

    # find eulerian paths in the graph
    seen_edges = set()
    paths = []
    while len(seen_edges) < n_edges:  # cover all edges
        node_iter = iter(rank)
        u = next((x for x in node_iter if rank[x] > 0))  # start from first node that still has edges
        path = []

        # find eulerian path from u
        while rank[u] > 0:
            path.append(u)
            v = graph[u].pop()  # continue from neighbor with the highest rank
            if (u, v) not in seen_edges:
                seen_edges.add((u, v))
                seen_edges.add((v, u))
                graph[v].remove(u)
                rank[u] -= 1
                rank[v] -= 1
            u = v
        else:
            path.append(u)
        paths.append(path)
    return paths


def connect_paths(paths: List[CONTINUOUS_PATH_TYPE], n_nodes: int) -> CONTINUOUS_PATH_TYPE:
    """
    Given paths on a circular graph (last edge is adjacent to first), connect them to a single path by adding edges
    only around the perimeter (between adjacent nodes). Attempt to do it with minimal added edges.
    :param paths: list of paths on the graph, each is a list of nodes representing the path.
    :param n_nodes: number of nodes in the graph.
    :return: path: a single continuous path connecting all paths, represented by a list of nodes.
    """
    # take a heuristic approach by sorting paths based on their start and end nodes
    n_paths = len(paths)
    start_nodes = [(i, path[0]) for i, path in enumerate(paths)]
    start_nodes = sorted(start_nodes, key=lambda x: x[1])
    paths_by_start, start_nodes = map(list, zip(*start_nodes))
    end_nodes = [(i, path[-1]) for i, path in enumerate(paths)]
    end_nodes = sorted(end_nodes, key=lambda x: x[1])
    paths_by_end, end_nodes = map(list, zip(*end_nodes))

    # greedy approach - connect current path to the nearest path available
    k = 0
    path = []
    path.extend(paths[paths_by_start[k]])
    for _ in range(n_paths - 1):
        del start_nodes[k], end_nodes[k], paths_by_start[k], paths_by_end[k]
        end_to_start_distance, end_to_start_pos = _find_nearest_circular(start_nodes, path[-1], n_nodes)
        end_to_end_distance, end_to_end_pos = _find_nearest_circular(end_nodes, path[-1], n_nodes)
        start_to_end_distance, start_to_end_pos = _find_nearest_circular(end_nodes, path[0], n_nodes)
        start_to_start_distance, start_to_start_pos = _find_nearest_circular(start_nodes, path[0], n_nodes)
        min_dist = min(end_to_start_distance, end_to_end_distance, start_to_end_distance, start_to_start_distance)
        if end_to_start_distance == min_dist:  # append to end
            k = end_to_start_pos
            _connect_on_perimeter(path, paths[paths_by_start[k]], n_nodes, prepend=False)
        elif end_to_end_distance == min_dist:  # need to flip next path and append to end
            k = end_to_end_pos
            _connect_on_perimeter(path, paths[paths_by_end[k]][::-1], n_nodes, prepend=False)
        elif start_to_end_distance == min_dist:  # prepend to start
            k = start_to_end_pos
            _connect_on_perimeter(path, paths[paths_by_end[k]], n_nodes, prepend=True)
        else:  # need to flip next path and prepend to start
            k = start_to_start_pos
            _connect_on_perimeter(path, paths[paths_by_start[k]][::-1], n_nodes, prepend=True)
    return path


def _find_nearest_circular(values: List[Any], x: Any, p: float) -> Tuple[Any, int]:
    """
    Given a sorted array and a number, find the nearest element on the perimeter of a circle.
    This has two meanings:
        1) If the nearest element is on the edges of the array, then it might also be to the other edge.
        2) The "nearest" metric should take into account both directions on the perimeter.
    :param values: sorted array.
    :param x: object to find.
    :param p: perimeter (length) of circle.
    :return: nearest element, and its position in the array.
    """
    pos = bisect.bisect_left(values, x)

    # compare the two neighbors, support circular adjacency
    before = values[(pos - 1) % len(values)]
    after = values[pos % len(values)]
    if min(abs(x - before), p - abs(x - before)) <= min(abs(x - after), p - abs(x - after)):
        pos = pos - 1
    pos = pos % len(values)
    return values[pos], pos


def _connect_on_perimeter(list1: List[int], list2: List[int], p: int, prepend: bool = False) -> List[int]:
    """
    Connect two lists such that the first and last elements are connected along the perimeter of a circle.
    Updates the first list.
    :param list1: first list to connect.
    :param list2: second list to connect.
    :param p: perimeter (length) of circle.
    :param prepend: append second to first at start instead of at end.
    :return: updated first list.
    """
    if prepend:
        index = len(list2)
        list1[:0] = list2
    else:
        index = len(list1)
        list1.extend(list2)

    # now connect them
    n_add = (list1[index] - list1[index - 1]) % p
    list1[index:index] = [x % p for x in range(list1[index - 1], list1[index - 1] + n_add)]
    del list1[index]  # remove duplicate
    return list1


def vec_to_lines(x: np.ndarray, lines: PATH_TYPE, threshold: float = 1.) -> PATH_TYPE:
    """
    Transform a vector of values to lines based on a threshold.
    :param x: vector with line values.
    :param lines: mapping from vector element to line.
    :param threshold: each value above this will be added to the lines list.
    :return: list of tuples representing start node and end node of each line.
    """
    return [lines[i] for i in range(len(x)) if x[i] >= threshold]
