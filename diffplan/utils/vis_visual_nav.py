import matplotlib.pyplot as plt

import io
import base64
from collections import defaultdict

from dash import dcc, html, Input, Output, no_update
from jupyter_dash import JupyterDash

import plotly.express as px

from PIL import Image

from diffplan.utils.vis_pano import plot_pano, get_plt_np_array


def array2base64(img_array, to_format="jpeg"):
    im = Image.fromarray(img_array)

    buffer = io.BytesIO()
    im.save(buffer, format=to_format)
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = f"data:image/{to_format};base64," + encoded_image

    return im_url


def render_all_ego_obs(pos2pano, pos2top=None):
    pos2url = defaultdict()

    for x in range(pos2pano.shape[0]):
        for y in range(pos2pano.shape[1]):
            _pano_obs = pos2pano[x, y]
            _top_obs = (
                None if (pos2top is None or (x, y) not in pos2top) else pos2top[x, y]
            )

            _fig = plot_pano(
                pano_obs=_pano_obs, top_obs=_top_obs, title=f"Position = ${x, y}$"
            )
            img_array = get_plt_np_array(_fig)
            img_src = array2base64(img_array)

            plt.close("all")

            pos2url[x, y] = img_src

    return pos2url


def setup_interactive_ego_obs(map_img, pos2obs_src, map_source="grid"):
    assert map_source in ["grid", "topdown"]

    # > Plot map
    fig = px.imshow(img=map_img, color_continuous_scale="Blues_r")

    # > turn off native plotly.js hover effects - make sure to use
    # hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        xaxis=dict(title="X axis"),
        yaxis=dict(title="Y axis"),
    )

    # > start Dash in Jupyter
    app = JupyterDash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip"),
        ]
    )

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]

        x = pt["x"]
        y = pt["y"]

        if map_source == "grid":
            src = pos2obs_src[x, y]
        elif map_source == "topdown":
            src = pos2obs_src[x // 5, y // 5]
        else:
            raise ValueError

        children = [
            html.Img(
                src=src,
                style=dict(width="300px"),
            ),
            html.Center(
                html.H3(f"Position = (x={x}, y={y})"),
            ),
        ]

        # > Note: correspond to Output bound in @app.callback decorator
        return True, bbox, children

    # app.run_server(debug=True, mode='inline')

    return app


def run_pano_obs(map_img, all_obs):
    pos2obs_src = render_all_ego_obs(pos2pano=all_obs)
    app = setup_interactive_ego_obs(map_img=map_img, pos2obs_src=pos2obs_src)
    app.run_server(debug=True, mode="inline")
