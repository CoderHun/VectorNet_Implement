import matplotlib.pyplot as plt
import scipy.interpolate as interp
import numpy as np
import matplotlib.patches as patches
from collections import defaultdict

def get_local_drivable_area(avm, df):
    city_name = df["CITY_NAME"].values[0]
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]
    x_offset = 0
    y_offset = 0
    x_min = int(max(min(df["X"]) - x_offset/2,0))
    x_max = int(max(df["X"]) + x_offset/2)
    y_min = int(max(min(df["Y"]) - y_offset/2,0))
    y_max = int(max(df["Y"]) + y_offset/2)
    drivable_area = avm.find_local_driveable_areas((x_min,x_max,y_min,y_max),city_name)
    return drivable_area

def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i].astype(float), polyline[i - 1].astype(float)): # 정지해있으면
            duplicates.append(i) # 정지해있는 부분의 index를 저장
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T.astype(float), s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

def viz_sequence_rasterize(dict_, agent, df, avm, gt=None, dir = None):
    _ZORDER = {"AGENT": 15, "AV": 10, "OTHERS": 5}

    city_name = df["CITY_NAME"].values[0]
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]
    da_mat = avm.city_rasterized_da_roi_dict[city_name]['da_mat']
    x_min = min(df["X"])
    x_max = max(df["X"])
    y_min = min(df["Y"])
    y_max = max(df["Y"])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    lane_centerlines = []
    for lane_id, lane_props in seq_lane_props.items():
        lane_cl = lane_props.centerline
        if (
            np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min
        ):
            lane_centerlines.append(lane_cl)
    for lane_id, lane_props in seq_lane_props.items():
        lane_cl = lane_props.centerline
        if (
            np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min
        ):
            lane_centerlines.append(lane_cl)
            
    # 먼저 drivable area를 흰색으로 칠해준다
    ax = plt.gca()
    ax.set_facecolor((1.0, 1.0, 1.0))
    drivable_area = get_local_drivable_area(avm, df)
    for idx, area in enumerate(drivable_area):
        dv = np.delete(area,2,axis=1)
        if idx == 0:
            color = 'white'
            continue
        else:
            color = 'black'
        ax.add_patch(
         patches.Polygon(
            (dv),
            closed=True,
            linewidth = 1,
            edgecolor = 'white',
            facecolor = color
         ))

    # center line들을 그려준다
    for lane_cl in lane_centerlines:
        plt.plot(lane_cl[:, 0], lane_cl[:, 1], "--", color="green", alpha=1, linewidth=1, zorder=0)
    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    color_dict = {"AGENT": "RED", "OTHERS": "BLUE", "AV": "#007672"}
    object_type_tracker: Dict[int, int] = defaultdict(int)
    # 원래 API에서는 여기서 무엇을 나타낼지 쿼리를 하지만 이미 무엇을 그릴지 결정했기 때문에 바로 그리면 된다.
    #dict_['agent'] = agent
    # actors 과거 Trajectory를 그려준다
    for key, actor in dict_['actor'].items():
        cor_x = actor["x"]
        cor_y = actor["y"]
        polyline = np.column_stack((cor_x, cor_y))
        num_points = cor_x.shape[0] * 3
        if key == "agent":
            object_type = "AGENT"
        else:
            object_type = "OTHERS"
        smooth_polyline = interpolate_polyline(polyline, num_points)
        cor_x = smooth_polyline[:, 0]
        cor_y = smooth_polyline[:, 1]
        plt.plot(
            cor_x,
            cor_y,
            "-",
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            linewidth=1,
            zorder=_ZORDER[object_type],
        )
        final_x = cor_x[-1]
        final_y = cor_y[-1]
        plt.plot(
            final_x,
            final_y,
            "o",
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            markersize=7,
            zorder=_ZORDER[object_type],
        )
    #Agent 그려준다.


    if gt !=None:
        #print(gt)
        x, y = gt
        #plt.plot(gt[:, 0], gt[:, 1], "--", color="green", alpha=1, linewidth=1, zorder=0)
        plt.plot(x, y, "--", color="red", alpha=1, linewidth=1, zorder=0)
        plt.savefig( dir + '/savefig_default.png')


def draw_connected_lane(_dict):
    # usage : draw_lane(trainable_set['0'], trainable_set['0']['connected_lane'])
    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    lane_set = _dict['connected_lane']
    lane_centerlines = []
    for lanes in lane_set:
        for lane in lanes:
            lane_centerlines.append(_dict['lanes'][lane]['centerline'])
    for lane_cl in lane_centerlines:
        plt.plot(lane_cl[:, 0], lane_cl[:, 1], "--", color="green", alpha=1, linewidth=1, zorder=0)

def draw_test_fig(_dict, pred, gt, name):
    color_dict = {"AGENT": "RED", "OTHERS": "BLUE", "AV": "#007672"}
    _ZORDER = {"AGENT": 15, "AV": 10, "OTHERS": 5}
    object_type_tracker: Dict[int, int] = defaultdict(int)
    ax = plt.gca()
    ax.set_facecolor((1.0, 1.0, 1.0))
    df = _dict['df']
    draw_connected_lane(_dict)
    # draw agent
    agent = _dict['agent']
    cor_x = agent["x"][:20]
    cor_y = agent["y"][:20]
    polyline = np.column_stack((cor_x, cor_y))
    num_points = cor_x.shape[0] * 3
    object_type = "AGENT"
    smooth_polyline = interpolate_polyline(polyline, num_points)
    cor_x = smooth_polyline[:, 0]
    cor_y = smooth_polyline[:, 1]
    plt.plot(cor_x,cor_y,"-",color=color_dict[object_type],
        label=object_type if not object_type_tracker[object_type] else "",
        alpha=1,linewidth=1,zorder=_ZORDER[object_type],
    )
    final_x = cor_x[-1]
    final_y = cor_y[-1]
    plt.plot(final_x,final_y,"o",color=color_dict[object_type],
        label=object_type if not object_type_tracker[object_type] else "",
        alpha=1,markersize=7,zorder=_ZORDER[object_type],
    )
    # draw GT
    
    gt_x = np.array(gt[:30], dtype=np.float64).tolist()
    gt_y = np.array(gt[30:], dtype=np.float64).tolist()
    #gt_x = agent["x"][20:]
    #gt_y = agent["y"][20:]
    #plt.plot(gt[:, 0], gt[:, 1], "--", color="green", alpha=1, linewidth=1, zorder=0)
    plt.plot(gt_x, gt_y, "-", color="blue", alpha=1, linewidth=1, zorder=0)

    # draw pred
    x = np.array(pred[:30], dtype=np.float64).tolist()
    y = np.array(pred[30:], dtype=np.float64).tolist()
    plt.plot(x, y, "-", color="orange", alpha=1, linewidth=1, zorder=0)
    plt.savefig(name)
    plt.cla()

def draw_simple_fig(pred, gt, name):
    pd_x = np.array(pred[:30], dtype=np.float64)
    pd_y = np.array(pred[30:], dtype=np.float64)
    polyline = np.column_stack((pd_x, pd_y))
    #num_points = pd_x.shape[0] * 3
    #smooth_polyline = interpolate_polyline(polyline, num_points)
    #cor_x = smooth_polyline[:, 0]
    #cor_y = smooth_polyline[:, 1]
    plt.plot(pd_x, pd_y, "-", color="blue", alpha=1, linewidth=1, zorder=0)

    pd_x = np.array(gt[:30], dtype=np.float64)
    pd_y = np.array(gt[30:], dtype=np.float64)
    polyline = np.column_stack((pd_x, pd_y))
    num_points = pd_x.shape[0] * 3
    #smooth_polyline = interpolate_polyline(polyline, num_points)
    #cor_x = smooth_polyline[:, 0]
    #cor_y = smooth_polyline[:, 1]
    plt.plot(pd_x, pd_y, "-", color="red", alpha=1, linewidth=1, zorder=0)

    final_x = pd_x[0]
    final_y = pd_y[0]
    plt.plot(final_x,final_y,"o", color="red", alpha=1, linewidth=1, zorder=0,)

    plt.savefig(name)
    plt.cla()